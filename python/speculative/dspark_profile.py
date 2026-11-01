"""DSpark startup memory probes, independent of the model runner."""

from typing import TYPE_CHECKING, Callable

import torch
from logger import logger

from gllm.runtime.input_data import InputData
from gllm.runtime.memory_manager import DSparkRequestCache
from gllm.runtime.sequence import GenerationSequence

if TYPE_CHECKING:
    from gllm.models.qwen38_dspark import Qwen38DSpark


@torch.inference_mode()
def profile_dspark_prefill(
    target: torch.nn.Module, draft: "Qwen38DSpark", input_data: InputData,
    *, max_running_seqs: int = 1, uses_mrope: bool = False,
    prepare_attention: Callable[[InputData], None] | None = None,
) -> dict[str, int]:
    """Include taps, projection/noise and both heads before sizing the arena.

    Dense K/V is temporary profiling storage only. A nonempty one-row
    suffix avoids the validation path's unsupported zero-row RMSNorm. The
    startup chunk must contain at least eight rows to also probe all target
    verify logits; this minimum does not restrict real request prompt lengths.
    Paged attention uses four FP32 split accumulators; the explicit extra
    allowance also covers its merge/noise tensors. The real paged path is
    exercised separately after arena allocation. A configured B-request
    runtime also probes B noise blocks and B*8 target rows independently of
    the maximum single prompt chunk; actual request prompts may be shorter.
    """
    if input_data.tokens_cpu.shape[0] < 8:
        raise ValueError("DSpark startup profiling requires a prefill chunk of at least 8 tokens")
    if (
        max_running_seqs < 1 or input_data.tokens_cpu.shape[0] < 8 * max_running_seqs
        or (max_running_seqs > 1 and prepare_attention is None)
    ):
        raise ValueError("DSpark batch profile requires 8 * maxd rows and prepared attention")
    torch.cuda.reset_peak_memory_stats()
    hidden, taps = target.forward_with_hidden_taps(
        input_data, target_layer_ids=draft.config.target_layer_ids
    )
    # Keep the full-chunk projection alive to account for the largest
    # actual append shape, even if RoPE bounds shorten the noise probe.
    projected = draft._project_context(taps)
    anchor = input_data.get_tokens()[:1]
    rows = min(taps.shape[1], draft.rotary_emb.cos_sin_cache.shape[0] - 7)
    cache = draft.prefill(taps[:, :rows - 1], anchor)
    heads, dim = draft.config.num_attention_heads, draft.config.head_dim
    workspace = torch.empty(
        (7, heads, 4 * (dim + 1) + 8 * dim + 4 * 7 + 4),
        dtype=torch.float32, device=taps.device,
    )
    draft_hidden, _ = draft.forward_draft(
        taps[:, rows - 1:rows], anchor, start_pos=rows, caches=cache
    )
    draft_output = draft.forward_head(draft_hidden, anchor)
    target_logits = target.logits_from_hidden(hidden[-8:])
    profile = {
        "prefill_tokens": taps.shape[1],
        "projected_tokens": projected.shape[1],
        "noise_rows": draft_output[0].shape[1] - 1,
        "target_logits_rows": target_logits.shape[0],
        "prefill_peak_bytes": torch.cuda.max_memory_allocated(),
        "paged_workspace_allowance_bytes": workspace.numel() * workspace.element_size(),
    }
    if max_running_seqs > 1:
        # Release the completed maximum-chunk outputs before probing the
        # distinct B x 8/B x 7 shape. Cached allocator memory still covers
        # the prior peak when MemoryManager sizes the arena from free memory.
        del hidden, taps, projected, cache, workspace, draft_hidden, draft_output, target_logits
        seqs = [
            GenerationSequence(-2 - row, [0] * 8, [], output_len=8, temperature=0)
            for row in range(max_running_seqs)
        ]
        pages = (8 + input_data.page_size - 1) // input_data.page_size
        for row, seq in enumerate(seqs):
            seq.to_compute_token_num = 8
            # InputData still materializes slot metadata before the arena
            # exists; these dummy pages are never dereferenced by attention.
            seq.page_table = list(range(row * pages, (row + 1) * pages))
        torch.cuda.reset_peak_memory_stats()
        input_data.cal_and_set_input(seqs)
        if uses_mrope:
            input_data.set_mrope_position(
                torch.arange(8, device="cpu").repeat(max_running_seqs).repeat(3, 1)
            )
        prepare_attention(input_data)
        hidden, taps = target.forward_with_hidden_taps(
            input_data, target_layer_ids=draft.config.target_layer_ids
        )
        features = taps.reshape(max_running_seqs, 8, -1)
        anchors = input_data.get_tokens().reshape(max_running_seqs, 8)[:, 0]
        rows = min(8, draft.rotary_emb.cos_sin_cache.shape[0] - 7)
        cache = draft.prefill(features[:, :rows - 1], anchors)
        workspace = torch.empty(
            (max_running_seqs * 7, heads, 4 * (dim + 1) + 8 * dim + 4 * 7 + 4),
            dtype=torch.float32, device=taps.device,
        )
        draft_hidden, _ = draft.forward_draft(
            features[:, rows - 1:rows], anchors, start_pos=rows, caches=cache
        )
        draft_output = draft.forward_head(draft_hidden, anchors)
        for row in range(max_running_seqs * 8):
            target.logits_from_hidden(hidden[row:row + 1]).argmax(-1)
        profile.update(
            profile_batch_size=max_running_seqs, noise_rows=max_running_seqs * 7,
            target_logits_rows=max_running_seqs * 8,
            prefill_peak_bytes=max(profile["prefill_peak_bytes"], torch.cuda.max_memory_allocated()),
            paged_workspace_allowance_bytes=workspace.numel() * workspace.element_size(),
        )
    logger.info(
        "DSpark pre-arena profile: prompt=%d, noise=%d, target_logits=%d, "
        "peak_allocated=%.2f MiB, paged_allowance=%.2f MiB",
        profile["prefill_tokens"], profile["noise_rows"], profile["target_logits_rows"],
        profile["prefill_peak_bytes"] / (1 << 20),
        profile["paged_workspace_allowance_bytes"] / (1 << 20),
    )
    return profile


@torch.inference_mode()
def profile_dspark_paged_verify(
    target: torch.nn.Module,
    draft: "Qwen38DSpark",
    input_data: InputData,
    *,
    max_prompt_tokens: int,
    model_max_length: int,
    uses_mrope: bool,
    prepare_attention: Callable[[InputData], None],
    max_running_seqs: int = 1,
) -> dict[str, int]:
    """Probe the maximum prompt chunk and B eight-row target verify windows.

    Startup failures propagate to the worker's existing failure path.
    In particular, do not attempt event waits or cache recovery after a
    potentially fatal CUDA error. Cleanup below runs only after success.
    """
    if max_prompt_tokens < 8:
        raise ValueError("DSpark startup profiling requires a prefill chunk of at least 8 tokens")
    if max_running_seqs < 1 or max_prompt_tokens < 8 * max_running_seqs:
        raise ValueError("DSpark batch profile requires 8 * maxd token capacity")

    from gllm.speculative.dspark import verify_dspark_target, verify_dspark_target_batched

    manager = input_data.memory_manager
    profile = {}
    if max_running_seqs > 1:
        profile["probe_initial_physical_pages"] = manager.cache_arena.allocator.num_used_physical_pages

    def prefill(length):
        seq = GenerationSequence(
            -1, [0] * length, [], output_len=8, temperature=0, top_k=1,
        )
        cache = DSparkRequestCache()
        if not manager.reserve_dspark_pages(
            seq, cache, target_length=length, draft_length=length
        ):
            raise RuntimeError("DSpark startup probe cannot reserve target/draft/GDN storage")
        seq.to_compute_token_num = length
        input_data.cal_and_set_input([seq])
        if uses_mrope:
            input_data.set_mrope_position(
                torch.arange(length, device="cpu").repeat(3, 1)
            )
        prepare_attention(input_data)
        hidden, taps = target.forward_with_hidden_taps(
            input_data, target_layer_ids=draft.config.target_layer_ids
        )
        paged = draft.prefill_paged(
            taps, segment=manager.dspark_segment, cache=cache
        )
        seq.computed_token_num = length
        return seq, cache, hidden, taps, paged

    # Profile the maximum chunk even when it leaves no eight-row suffix.
    # A second, shorter request then probes the full verify shape.
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    length = max_prompt_tokens
    seq, cache, hidden, taps, paged = prefill(length)
    torch.cuda.synchronize()
    profile.update(
        paged_prefill_tokens=length,
        paged_prefill_peak_extra_bytes=torch.cuda.max_memory_allocated() - before,
    )
    if length + 8 > model_max_length:
        manager.free_dspark_cache(seq, cache)
        torch.cuda.synchronize()
        # Assignment evaluates the next prefill before replacing these names.
        # Drop the completed probe outputs first to avoid overlapping peaks.
        del hidden, taps, paged
        length = model_max_length - 8
        seq, cache, hidden, taps, paged = prefill(length)
    if not manager.reserve_dspark_pages(
        seq, cache, target_length=length + 8, draft_length=length + 8
    ):
        raise RuntimeError("DSpark startup probe cannot reserve the full verify window")
    anchor = target.logits_from_hidden(hidden[-1:]).argmax(-1)
    seq.computed_token_num = length
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    draft_hidden, _ = draft.forward_draft(
        taps[:, :0], anchor, start_pos=length, caches=paged
    )
    proposed, _, _ = draft.forward_head(draft_hidden, anchor)
    seq.to_compute_tokens = proposed[0].cpu().tolist()
    seq.to_compute_token_num = 8
    seq._mtp_verify = True
    input_data.cal_and_set_input([seq])
    if uses_mrope:
        input_data.set_mrope_position(
            torch.arange(length, length + 8, device="cpu").repeat(3, 1)
        )
    prepare_attention(input_data)
    verify_dspark_target(
        target, input_data, block_size=7,
        target_layer_ids=draft.config.target_layer_ids,
    )
    torch.cuda.synchronize()
    profile.update(
        verify_tokens=8, verify_context_tokens=length,
        verify_peak_extra_bytes=torch.cuda.max_memory_allocated() - before,
    )
    manager.free_dspark_cache(seq, cache)
    torch.cuda.synchronize()
    logger.info(
        "DSpark post-arena profile: prompt=%d (peak_extra=%.2f MiB), "
        "context=%d, verify=8 (peak_extra=%.2f MiB); temporary request released",
        profile["paged_prefill_tokens"],
        profile["paged_prefill_peak_extra_bytes"] / (1 << 20),
        length, profile["verify_peak_extra_bytes"] / (1 << 20),
    )
    if max_running_seqs > 1:
        # The single maximum-chunk probe above must not overlap the next
        # requests' lifetimes or retain their full raw feature buffers.
        del seq, cache, hidden, taps, paged, anchor, draft_hidden, proposed
        base, extra = divmod(max_prompt_tokens, max_running_seqs)
        lengths = [base + (row < extra) for row in range(max_running_seqs)]
        seqs = [
            GenerationSequence(-2 - row, [0] * count, [], output_len=8, temperature=0)
            for row, count in enumerate(lengths)
        ]
        caches = [DSparkRequestCache() for _ in seqs]
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        for request, state, count in zip(seqs, caches, lengths):
            if not manager.reserve_dspark_pages(
                request, state, target_length=count, draft_length=count
            ):
                raise RuntimeError("DSpark batch profile cannot reserve all request working states")
            request.to_compute_token_num = count
        input_data.cal_and_set_input(seqs)
        if uses_mrope:
            input_data.set_mrope_position(torch.cat([
                torch.arange(count, device="cpu") for count in lengths
            ]).repeat(3, 1))
        prepare_attention(input_data)
        hidden, taps = target.forward_with_hidden_taps(
            input_data, target_layer_ids=draft.config.target_layer_ids
        )
        begin, chunks, anchors = 0, [], []
        for request, count in zip(seqs, lengths):
            chunks.append(taps[:, begin:begin + count])
            anchors.append(target.logits_from_hidden(hidden[begin + count - 1:begin + count]).argmax(-1))
            request.computed_token_num = count
            begin += count
        paged = draft.prefill_paged_batched(
            chunks, segment=manager.dspark_segment, caches=caches
        )
        anchors = torch.cat(anchors)
        empty = [taps.new_empty((1, 0, taps.shape[-1])) for _ in seqs]
        torch.cuda.synchronize()
        profile.update(
            batch_prefill_tokens=sum(lengths),
            batch_prefill_peak_extra_bytes=torch.cuda.max_memory_allocated() - before,
        )
        del hidden, taps, chunks
        for request, state, count in zip(seqs, caches, lengths):
            if not manager.reserve_dspark_pages(
                request, state, target_length=count + 8, draft_length=count + 8
            ):
                raise RuntimeError("DSpark batch profile cannot reserve all verify windows")
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        draft_hidden, _ = draft.forward_draft_batched(
            empty, anchors, start_positions=lengths, caches=paged
        )
        proposed, _, _ = draft.forward_head(draft_hidden, anchors)
        proposed_ids = proposed.cpu().tolist()
        for request, tokens in zip(seqs, proposed_ids):
            request.to_compute_tokens = tokens
            request.to_compute_token_num, request._mtp_verify = 8, True
        input_data.cal_and_set_input(seqs)
        if uses_mrope:
            input_data.set_mrope_position(torch.cat([
                torch.arange(count, count + 8, device="cpu") for count in lengths
            ]).repeat(3, 1))
        prepare_attention(input_data)
        verify_dspark_target_batched(
            target, input_data, block_size=7,
            target_layer_ids=draft.config.target_layer_ids,
        )
        torch.cuda.synchronize()
        profile.update(
            verify_batch_size=max_running_seqs, verify_tokens=8 * max_running_seqs,
            verify_context_tokens=max(lengths), gdn_working_slots=8 * max_running_seqs,
            batch_verify_peak_extra_bytes=torch.cuda.max_memory_allocated() - before,
        )
        profile["verify_peak_extra_bytes"] = max(
            profile["verify_peak_extra_bytes"], profile["batch_verify_peak_extra_bytes"]
        )
        for request, state in zip(seqs, caches):
            manager.free_dspark_cache(request, state)
        torch.cuda.synchronize()
        profile["probe_final_physical_pages"] = manager.cache_arena.allocator.num_used_physical_pages
        if profile["probe_final_physical_pages"] != profile["probe_initial_physical_pages"]:
            raise RuntimeError("DSpark startup batch probe leaked temporary arena ownership")
        logger.info(
            "DSpark batch profile: requests=%d, prompt=%d, verify=%d, GDN working slots=%d; "
            "temporary requests released",
            max_running_seqs, sum(lengths), 8 * max_running_seqs, 8 * max_running_seqs,
        )
    return profile
