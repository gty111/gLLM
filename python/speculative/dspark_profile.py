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
) -> dict[str, int]:
    """Include taps, projection/noise and both heads before sizing the arena.

    Dense K/V is temporary profiling storage only. A nonempty one-row
    suffix avoids the validation path's unsupported zero-row RMSNorm.
    Paged attention uses four FP32 split accumulators; the explicit extra
    allowance also covers its merge/noise tensors. The real paged path is
    exercised separately after arena allocation.
    """
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
    logger.info(
        "DSpark pre-arena profile: prompt=%d, noise=7, target_logits=8, "
        "peak_allocated=%.2f MiB, paged_allowance=%.2f MiB",
        taps.shape[1], profile["prefill_peak_bytes"] / (1 << 20),
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
) -> dict[str, int]:
    """Probe an actual eight-row target verify with both caches in one arena.

    Startup failures propagate to the worker's existing failure path.
    In particular, do not attempt event waits or cache recovery after a
    potentially fatal CUDA error. Cleanup below runs only after success.
    """
    from gllm.speculative.dspark import verify_dspark_target

    manager = input_data.memory_manager
    profile = {}

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
    return profile
