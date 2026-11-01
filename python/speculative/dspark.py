"""Synchronous DSpark prompt steps and request-owned target verification."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Tuple

import torch
from torch import nn

from gllm.distributed.parallel_state import get_dp_size, get_pp_size, get_tp_size
from gllm.layers.attention.qkv_backends import FlashInferPagedAttentionMetadata
from gllm.runtime.forward_metadata import MetadataMaterialization
from gllm.runtime.input_data import InputData
from gllm.runtime.memory_manager import DSparkRequestCache
from gllm.runtime.sequence import GenerationSequence

if TYPE_CHECKING:
    from gllm.models.qwen38_dspark import Qwen38DSparkPagedCache


@dataclass
class DSparkStepResult:
    """Worker-local phase output, awaiting one scheduler commit.

    Prefill retains only its final prediction, not the large prompt taps.
    Bootstrap owns one input's taps and an unpublished next anchor.
    """

    seq_id: int
    cache_epoch: int
    start_pos: int
    query_len: int
    phase: str
    next_anchor: Optional[torch.Tensor] = None
    hidden: Optional[torch.Tensor] = None
    consumed: bool = False
    ready: bool = False  # GPU work submitted and covered by last_use; not GPU-complete.
    verification: Optional["DSparkVerifyResult"] = None


@dataclass
class DSparkRequestState:
    """Runner-owned tensors; never copied into a GenerationSequence.

    Target length comes solely from seq.computed_token_num. After bootstrap,
    pending_hidden covers [cache.context_length, target length) during verify, and anchor
    is unpublished at anchor_pos == target length. An outstanding result also
    blocks retry after a possibly partial forward failure.
    """

    seq_id: int
    cache_epoch: int
    cache: DSparkRequestCache = field(default_factory=DSparkRequestCache)
    phase: str = "prefill"
    paged_cache: Optional["Qwen38DSparkPagedCache"] = None
    pending_hidden: Optional[torch.Tensor] = None
    anchor: Optional[torch.Tensor] = None
    anchor_pos: Optional[int] = None
    last_use: Optional[torch.cuda.Event] = None
    inflight: Optional[DSparkStepResult] = None


def _validate_dspark_step_result(seq, state, result):
    if result.consumed:
        raise ValueError("DSpark step result was already consumed")
    if (
        not result.ready or state.inflight is not result or state.phase != result.phase
        or seq.seq_id != state.seq_id or seq.seq_id != result.seq_id
        or seq._cache_epoch != state.cache_epoch or seq._cache_epoch != result.cache_epoch
        or seq.computed_token_num != result.start_pos
        or seq.to_compute_token_num != result.query_len
        or seq.is_abort or seq.is_finish
    ):
        raise ValueError("DSpark step does not match the live request, epoch or position")


def commit_dspark_prompt_step(
    seq: GenerationSequence,
    state: DSparkRequestState,
    result: DSparkStepResult,
) -> list[int]:
    """Scheduler-side commit; return only tokens newly published this step.

    Intermediate prefill and bootstrap return no tokens. The final prompt
    chunk publishes y0 once; bootstrap later consumes that already-published
    input without appending it again. GPU work and draft append happen before
    this call; only this commit advances the authoritative target length.
    """
    _validate_dspark_step_result(seq, state, result)
    end = result.start_pos + result.query_len
    if result.query_len <= 0:
        raise ValueError("DSpark step must consume a nonempty input")
    if result.phase == "prefill":
        final = end == seq.prompt_len
        if (
            end > seq.prompt_len or len(seq) != seq.prompt_len
            or state.cache.context_length != end
            or (result.next_anchor is not None) != final
        ):
            raise ValueError("DSpark prefill result does not cover the next prompt chunk")
        # Read the token before mutating host progress. A failed device read
        # must not leave a partially committed request.
        published = [int(result.next_anchor.item())] if final else []
    elif result.phase == "bootstrap":
        if (
            result.query_len != 1 or result.start_pos != seq.prompt_len
            or len(seq) != end or state.cache.context_length != result.start_pos
            or result.next_anchor is None or result.hidden is None
            or result.hidden.ndim != 3 or result.hidden.shape[:2] != (1, 1)
        ):
            raise ValueError("DSpark bootstrap must consume the published first token once")
        published = []
    else:
        raise ValueError("DSpark prompt commit requires prefill or bootstrap")

    seq.computed_token_num = end
    seq.to_compute_token_num = 0
    if result.phase == "prefill":
        if published:
            seq.append(published[0])
            state.phase = "bootstrap"
    else:
        state.pending_hidden = result.hidden
        state.anchor, state.anchor_pos = result.next_anchor, end
        state.phase = "verify"
    result.consumed = True
    state.inflight = None
    return published


@dataclass
class DSparkVerifyResult:
    """Owned accepted inputs and taps, before target-state commit.

    accepted_drafts excludes the anchor. next_anchor has not been a target
    input at its correct position and has no taps/KV of its own yet.
    A state-commit helper must consume this record exactly once.
    """

    seq_id: int
    cache_epoch: int
    start_pos: int
    accepted_drafts: int
    committed_ids: torch.Tensor
    next_anchor: torch.Tensor
    accepted_hidden: torch.Tensor
    consumed: bool = False


def _validate_verify_input(target, data, block_size, layer_ids):
    if (get_tp_size(), get_pp_size(), get_dp_size()) != (1, 1, 1):
        raise ValueError("DSpark verify requires TP=PP=DP=1")
    seqs = data.seqs
    batch = len(seqs)
    if block_size != 7 or not batch:
        raise ValueError("DSpark verify requires a nonempty batch and seven proposals")
    if len({seq.seq_id for seq in seqs}) != batch:
        raise ValueError("DSpark verify requires distinct request IDs")
    if any(
        seq.mm_contents is not None or not seq.computed_prompt
        or seq.computed_token_num <= 0 or seq.is_abort or seq.is_finish
        or (seq.temperature != 0 and seq.top_k != 1)
        or seq.repetition_penalty != 1
        or seq.structured_output is not None
        or seq.logprobs_enabled or seq.prompt_logprobs_enabled
        for seq in seqs
    ):
        raise ValueError("DSpark verify requires a pure-text greedy decode request")
    text = getattr(target, "language_model", target)
    backbone = text.model
    if (
        not layer_ids or len(set(layer_ids)) != len(layer_ids)
        or any(
            not isinstance(i, int) or not backbone.start_layer <= i < backbone.end_layer
            for i in layer_ids
        )
    ):
        raise ValueError("target_layer_ids must be unique valid block indices")
    plan = data.forward_metadata_plan
    if (
        plan is None or plan.materialization != MetadataMaterialization.CPU
        or not plan.device_buffers_ready or plan.gpu_patch_rows
        or plan.query_lens != (8,) * batch or plan.num_decodes != 0
        or plan.num_mtp_verify_rows != batch or plan.attention_metadata is None
        or any(
            not getattr(seq, "_mtp_verify", False) or seq.to_compute_token_num != 8
            for seq in seqs
        )
        or not data.is_mtp_verify or data.num_mtp_verify_rows != batch
        or data.num_decodes != 0
    ):
        raise ValueError("DSpark verify needs prepared eight-row CPU verify metadata")

    manager = data.memory_manager
    kv, state = manager.segment, manager.ssm_segment
    tokens = data.get_tokens()
    rows = batch * 8
    if tokens.shape != (rows,) or tokens.dtype != torch.int64 or kv is None or state is None:
        raise ValueError("DSpark verify needs eight tokens per request and target KV/GDN storage")
    if torch.is_autocast_enabled(tokens.device.type):
        raise ValueError("DSpark verify does not support autocast")
    banks = [backbone.embed_tokens.weight, *kv.k_cache, *kv.v_cache, state.conv_state]
    draft = getattr(manager, "dspark_segment", None)
    if draft is not None:
        banks.extend([*draft.k_cache, *draft.v_cache])
    if any(t.dtype != torch.bfloat16 or t.device != tokens.device for t in banks):
        raise ValueError("DSpark activation, KV and conv state must share a BF16 device")
    # Verify carries FP32 recurrence between tokens. Ordinary decode writes and
    # reloads state every token, so BF16 temporal storage changes its rounding.
    if state.temporal_state.dtype != torch.float32 or state.temporal_state.device != tokens.device:
        raise ValueError("DSpark verify requires FP32 temporal state")

    gpu_table = data.get_ssm_block_table_2d()
    if gpu_table is None or gpu_table.shape != (batch, 8):
        raise ValueError("eight-column GDN device table was not materialized")
    block_table = data.get_block_table()
    if block_table is None or block_table.ndim != 2 or block_table.shape[0] != batch:
        raise ValueError("target device page table does not describe the verify batch")
    context_limit = text.max_model_len
    serving_limit = getattr(text.config, "model_max_length", None)
    if serving_limit is not None:
        context_limit = min(context_limit, serving_limit)
    positions, ends, slots, tables, rolling, page_checks = [], [], [], [], [], []
    used_states, used_write_pages = set(), set()
    for row, seq in enumerate(seqs):
        table = seq.ssm_block_table
        if (
            table is None or len(table) != 8 or len(set(table)) != 8
            or any(type(slot) is not int or not 0 < slot < state.num_blocks for slot in table)
            or seq.recurrent_state_slot != table[0] or seq.ssm_num_accepted != 1
        ):
            raise ValueError("DSpark verify needs eight distinct GDN slots and rolling column zero")
        if used_states.intersection(table):
            raise ValueError("DSpark verify requests must own disjoint GDN slots")
        used_states.update(table)
        start, end = seq.computed_token_num, seq.computed_token_num + 8
        # The runner publishes the resolved serving bound on the text config.
        # Reserved pages and the checkpoint window may both be larger.
        if end > context_limit:
            raise ValueError("target verify exceeds the model context window")
        if len(seq.page_table) * kv.page_size < end:
            raise ValueError("target KV capacity does not cover the verify window")
        first_page, last_page = start // kv.page_size, (end - 1) // kv.page_size
        pages = seq.page_table[first_page:last_page + 1]
        if any(not 0 <= page < kv.num_pages for page in pages):
            raise ValueError("invalid target page in the verify window")
        # Prefix sharing is disabled for DSpark. Concurrent verify windows
        # must never write the same physical page, even at different offsets.
        if len(set(pages)) != len(pages) or used_write_pages.intersection(pages):
            raise ValueError("DSpark verify requests must own distinct writable KV pages")
        used_write_pages.update(pages)
        if block_table.shape[1] <= last_page:
            raise ValueError("target device page table does not cover the verify window")
        page_checks.append((block_table[row, first_page:last_page + 1], pages))
        positions.extend(range(start, end))
        ends.append(end)
        slots.extend(
            seq.page_table[pos // kv.page_size] * kv.page_size + pos % kv.page_size
            for pos in range(start, end)
        )
        tables.extend(table)
        rolling.append(table[0])
    cpu_tokens = data.tokens_cpu.tolist()
    if len(cpu_tokens) != rows or any(not 0 <= token < text.config.vocab_size for token in cpu_tokens):
        raise ValueError("verify token IDs are outside the target vocabulary")
    position_buffer = data.get_position()
    # Conditional text-only requests carry three identical MRoPE axes. Check
    # every axis without replacing the model's original position buffer.
    if position_buffer is not None and position_buffer.shape == (3, rows):
        position_buffer = position_buffer.flatten()
        positions *= 3
    # One bounded read validates all requests before target forward. Historical
    # KV pages remain the InputData materializer's responsibility.
    checks = [
        (tokens, cpu_tokens),
        (position_buffer, positions),
        (data.get_seq_lens(), ends),
        (data.get_query_start_loc(), list(range(0, rows + 1, 8))),
        (data.get_slot_mapping(), slots),
        *page_checks,
        (gpu_table.flatten(), tables),
        (data.get_ssm_num_accepted(), [1] * batch),
        (data.get_recurrent_state_slot_per_seq(), rolling),
        (data.get_ssm_snapshot_write_slot_per_seq(), [-1] * batch),
    ]
    actual, expected = [], []
    for tensor, values in checks:
        if (
            tensor is None or tensor.shape != (len(values),)
            or tensor.dtype not in (torch.int32, torch.int64)
            or tensor.device != tokens.device
        ):
            raise ValueError("verify device metadata has invalid shape, dtype or device")
        actual.append(tensor.to(torch.int64))
        expected.extend(values)
    if torch.cat(actual).cpu().tolist() != expected:
        raise ValueError("verify device metadata is stale or inconsistent with the request")
    metadata = plan.attention_metadata
    if (
        metadata.max_query_len != 8 or metadata.batch_size != batch
        or metadata.block_table.data_ptr() != block_table.data_ptr()
        or metadata.seq_lens.data_ptr() != data.get_seq_lens().data_ptr()
        or metadata.query_start_loc.data_ptr() != data.get_query_start_loc().data_ptr()
    ):
        raise ValueError("attention metadata does not describe the verify buffers")
    if isinstance(metadata, FlashInferPagedAttentionMetadata) and (
        metadata.fast_path_rows != batch or metadata.fast_path_tokens != rows
        or metadata.fast_q_len_per_req != 8 or metadata.context_max_query_len != 0
    ):
        raise ValueError("FlashInfer metadata must use the eight-token verify partition")
    return seqs, tokens


@torch.no_grad()
def verify_dspark_target(
    target: nn.Module,
    input_data: InputData,
    *,
    block_size: int,
    target_layer_ids: Tuple[int, ...],
) -> DSparkVerifyResult:
    """Verify one [anchor, d1..d7] window; retain the single-result API."""
    if len(input_data.seqs) != 1:
        raise ValueError("DSpark single-request verify requires exactly one request")
    return verify_dspark_target_batched(
        target, input_data, block_size=block_size, target_layer_ids=target_layer_ids
    )[0]


@torch.no_grad()
def verify_dspark_target_batched(
    target: nn.Module,
    input_data: InputData,
    *,
    block_size: int,
    target_layer_ids: Tuple[int, ...],
) -> list[DSparkVerifyResult]:
    """Verify one packed [anchor, d1..d7] window for each request.

    The caller prepares/reserves all requests on the consuming stream, with
    BF16 activation/KV/conv and FP32 temporal state. Each consecutive eight
    rows belongs to the corresponding input_data.seqs entry, including taps.

    Target forward writes independent KV/GDN checkpoints, including column
    zero. A partial failure is not retryable without restoring/recomputing
    affected requests. No state is promoted or token published here. Consume
    each owned result separately, then prepare fresh metadata; EOS/budget
    cutoffs must select that request's actual committed checkpoint column.
    """
    layer_ids = tuple(target_layer_ids)
    seqs, tokens = _validate_verify_input(target, input_data, block_size, layer_ids)
    hidden, taps = target.forward_with_hidden_taps(
        input_data, target_layer_ids=layer_ids
    )
    # Ordinary greedy scores one row. Batched BF16 GEMM can round near-tied
    # logits differently even for the same hidden row, so retain that head
    # shape while executing the target body once for the whole packed batch.
    predictions = torch.cat([
        target.logits_from_hidden(hidden[i:i + 1]).argmax(dim=-1)
        for i in range(hidden.shape[0])
    ], dim=0).reshape(len(seqs), 8)
    windows = tokens.reshape(len(seqs), 8)
    matches = predictions[:, :-1] == windows[:, 1:]
    accepted = matches.to(torch.int32).cumprod(dim=1).sum(dim=1).cpu().tolist()
    results = []
    for row, (seq, count) in enumerate(zip(seqs, accepted)):
        length, begin = 1 + count, row * 8
        results.append(DSparkVerifyResult(
            seq_id=seq.seq_id, cache_epoch=seq._cache_epoch,
            start_pos=seq.computed_token_num, accepted_drafts=count,
            committed_ids=windows[row:row + 1, :length].clone(),
            next_anchor=predictions[row, count:count + 1].clone(),
            accepted_hidden=taps[:, begin:begin + length].clone(),
        ))
    return results


def commit_dspark_verify_state(
    seq: GenerationSequence,
    result: DSparkVerifyResult,
    *,
    committed_count: int,
) -> None:
    """Promote the post-state after exactly committed_count verified inputs.

    The input count includes the anchor, so its checkpoint column is count-1.
    All eight slots stay allocated; swapping CPU IDs promotes both conv and
    temporal state without copying either tensor. The request must serialize
    verify/commit and retain its validated checkpoint table until this call.

    This consumes result even when count=1 needs no exchange. It does not
    advance request lengths, append draft context or publish tokens. The caller
    completes those updates before any further forward and materializes fresh
    device metadata. A truncated commit must discard the original next_anchor.
    """
    if result.consumed:
        raise ValueError("DSpark verify result was already consumed")
    if (
        seq.seq_id != result.seq_id or seq._cache_epoch != result.cache_epoch
        or seq.computed_token_num != result.start_pos
    ):
        raise ValueError("DSpark verify result does not match the request, epoch or position")
    if (
        type(committed_count) is not int
        or not 0 <= result.accepted_drafts <= 7
        or not 1 <= committed_count <= result.accepted_drafts + 1
    ):
        raise ValueError("committed_count must select a nonempty accepted input prefix")
    table = seq.ssm_block_table
    if (
        table is None or len(table) != 8 or len(set(table)) != 8
        or any(type(slot) is not int or slot <= 0 for slot in table)
        or seq.recurrent_state_slot != table[0] or seq.ssm_num_accepted != 1
    ):
        raise ValueError("DSpark state commit needs the validated eight-slot rolling table")
    # Verify checked slot bounds; ownership stays with the caller. No manager
    # or allocator is involved. Validation precedes every mutation.
    column = committed_count - 1
    table[0], table[column] = table[column], table[0]
    seq.recurrent_state_slot = table[0]
    seq.ssm_num_accepted = 1
    result.consumed = True


def commit_dspark_verify_step(
    seq: GenerationSequence,
    state: DSparkRequestState,
    result: DSparkStepResult,
) -> list[int]:
    """Scheduler-side verify commit, including an EOS/output-budget cutoff.

    Every published token was a verified target input. Promote the checkpoint
    for the actual prefix before advancing target progress. The predicted next
    anchor is retained only when continuing after the complete accepted prefix.
    """
    _validate_dspark_step_result(seq, state, result)
    verified = result.verification
    if (
        result.phase != "verify" or result.query_len != 8 or verified is None
        or len(seq) != result.start_pos or state.cache.context_length != result.start_pos
        or state.pending_hidden is not None
    ):
        raise ValueError("DSpark verify commit needs an appended context and eight-row result")
    tokens = verified.committed_ids[0].cpu().tolist()
    remaining = seq.output_len - (len(seq) - seq.raw_prompt_len)
    count = min(len(tokens), remaining)
    if not seq.ignore_eos:
        count = next(
            (i + 1 for i, token in enumerate(tokens[:count]) if token in seq.finish_tokens),
            count,
        )
    if count <= 0:
        raise ValueError("DSpark verify commit has no remaining output budget")
    # All device reads and validation precede the first state-table mutation.
    commit_dspark_verify_state(seq, verified, committed_count=count)
    seq.token_ids.extend(tokens[:count])
    seq.computed_token_num += count
    seq.to_compute_token_num = 0
    if seq.is_finish:
        state.pending_hidden = state.anchor = None
        state.anchor_pos = None
    else:
        state.pending_hidden = verified.accepted_hidden
        state.anchor = verified.next_anchor
        state.anchor_pos = seq.computed_token_num
    result.consumed = True
    state.inflight = None
    return tokens[:count]


def commit_dspark_tail_step(
    seq: GenerationSequence,
    state: DSparkRequestState,
    result: DSparkStepResult,
) -> list[int]:
    """Publish the consumed relay input once; retain only its next prediction.

    Tail never appends draft context or promotes speculative GDN checkpoints:
    the ordinary single-token target forward updates rolling column zero.
    """
    _validate_dspark_step_result(seq, state, result)
    if (
        result.phase != "tail" or result.query_len != 1
        or len(seq) != result.start_pos or state.anchor_pos != result.start_pos
        or state.anchor is None or result.next_anchor is None
    ):
        raise ValueError("DSpark tail commit needs one consumed unpublished anchor")
    token = int(state.anchor.item())
    seq.append(token)
    seq.computed_token_num += 1
    seq.to_compute_token_num = 0
    state.pending_hidden = None
    state.anchor = None if seq.is_finish else result.next_anchor
    state.anchor_pos = None if seq.is_finish else seq.computed_token_num
    result.consumed = True
    state.inflight = None
    return [token]
