"""Synchronous single-request target verification for the Qwen3.8 drafter."""

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn

from gllm.distributed.parallel_state import get_dp_size, get_pp_size, get_tp_size
from gllm.layers.attention.qkv_backends import FlashInferPagedAttentionMetadata
from gllm.runtime.forward_metadata import MetadataMaterialization
from gllm.runtime.input_data import InputData
from gllm.runtime.sequence import GenerationSequence


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
    if block_size != 7 or len(data.seqs) != 1:
        raise ValueError("DSpark verify requires one request and seven proposals")
    seq = data.seqs[0]
    if (
        seq.mm_contents is not None or not seq.computed_prompt
        or seq.computed_token_num <= 0
        or (seq.temperature != 0 and seq.top_k != 1)
        or seq.repetition_penalty != 1
        or seq.structured_output is not None
        or seq.logprobs_enabled or seq.prompt_logprobs_enabled
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
        or plan.query_lens != (8,) or plan.num_decodes != 0
        or plan.num_mtp_verify_rows != 1 or plan.attention_metadata is None
        or not getattr(seq, "_mtp_verify", False)
        or not data.is_mtp_verify or data.num_mtp_verify_rows != 1
        or data.num_decodes != 0 or seq.to_compute_token_num != 8
    ):
        raise ValueError("DSpark verify needs prepared eight-row CPU verify metadata")

    manager = data.memory_manager
    kv, state = manager.segment, manager.ssm_segment
    tokens = data.get_tokens()
    if tokens.shape != (8,) or tokens.dtype != torch.int64 or kv is None or state is None:
        raise ValueError("DSpark verify needs eight tokens and target KV/GDN storage")
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

    table = seq.ssm_block_table
    if (
        table is None or len(table) != 8 or len(set(table)) != 8
        or any(not 0 < slot < state.num_blocks for slot in table)
        or seq.recurrent_state_slot != table[0] or seq.ssm_num_accepted != 1
    ):
        raise ValueError("DSpark verify needs eight distinct GDN slots and rolling column zero")
    start = seq.computed_token_num
    end = start + 8
    # The runner publishes the resolved serving bound on the text config.
    # Reserved pages and the checkpoint window may both be larger.
    context_limit = text.max_model_len
    serving_limit = getattr(text.config, "model_max_length", None)
    if serving_limit is not None:
        context_limit = min(context_limit, serving_limit)
    if end > context_limit:
        raise ValueError("target verify exceeds the model context window")
    if len(seq.page_table) * kv.page_size < end:
        raise ValueError("target KV capacity does not cover the verify window")
    first_page, last_page = start // kv.page_size, (end - 1) // kv.page_size
    pages = seq.page_table[first_page:last_page + 1]
    if any(not 0 <= page < kv.num_pages for page in pages):
        raise ValueError("invalid target page in the verify window")
    cpu_tokens = data.tokens_cpu.tolist()
    if len(cpu_tokens) != 8 or any(not 0 <= token < text.config.vocab_size for token in cpu_tokens):
        raise ValueError("verify token IDs are outside the target vocabulary")
    slots = [
        seq.page_table[pos // kv.page_size] * kv.page_size + pos % kv.page_size
        for pos in range(start, end)
    ]
    gpu_table = data.get_ssm_block_table_2d()
    if gpu_table is None or gpu_table.shape != (1, 8):
        raise ValueError("eight-column GDN device table was not materialized")
    # A single bounded read catches stale GPU mirrors after column promotion.
    # Historical KV pages remain the InputData materializer's responsibility.
    checks = [
        (tokens, cpu_tokens),
        (data.get_position(), list(range(start, end))),
        (data.get_seq_lens(), [end]),
        (data.get_query_start_loc(), [0, 8]),
        (data.get_slot_mapping(), slots),
        (data.get_block_table()[0, first_page:last_page + 1], pages),
        (gpu_table.flatten(), table),
        (data.get_ssm_num_accepted(), [1]),
        (data.get_recurrent_state_slot_per_seq(), [table[0]]),
        (data.get_ssm_snapshot_write_slot_per_seq(), [-1]),
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
        metadata.max_query_len != 8 or metadata.batch_size != 1
        or metadata.block_table.data_ptr() != data.get_block_table().data_ptr()
        or metadata.seq_lens.data_ptr() != data.get_seq_lens().data_ptr()
        or metadata.query_start_loc.data_ptr() != data.get_query_start_loc().data_ptr()
    ):
        raise ValueError("attention metadata does not describe the verify buffers")
    if isinstance(metadata, FlashInferPagedAttentionMetadata) and (
        metadata.fast_path_rows != 1 or metadata.fast_path_tokens != 8
        or metadata.fast_q_len_per_req != 8 or metadata.context_max_query_len != 0
    ):
        raise ValueError("FlashInfer metadata must use the eight-token verify partition")
    return seq, tokens


@torch.no_grad()
def verify_dspark_target(
    target: nn.Module,
    input_data: InputData,
    *,
    block_size: int,
    target_layer_ids: Tuple[int, ...],
) -> DSparkVerifyResult:
    """Verify [anchor, d1..d7] against the target's causal next-token logits.

    The caller prepares/reserves a single greedy request on the consuming
    stream. Activation/KV/conv are BF16; temporal state is FP32. This eager
    path reads bounded device metadata and the accepted count on the host.

    Target forward writes KV and GDN checkpoints, including column zero.
    It cannot be retried after a partial failure without restoring/recomputing
    the request. This call neither promotes a checkpoint nor publishes tokens
    or advances request/draft lengths. Consume/commit the result before another
    target forward, then materialize fresh metadata for the next call.
    """
    layer_ids = tuple(target_layer_ids)
    seq, tokens = _validate_verify_input(target, input_data, block_size, layer_ids)
    hidden, taps = target.forward_with_hidden_taps(
        input_data, target_layer_ids=layer_ids
    )
    predictions = target.logits_from_hidden(hidden).argmax(dim=-1)
    matches = predictions[:-1] == tokens[1:]
    accepted = int(matches.to(torch.int32).cumprod(dim=0).sum().item())
    length = 1 + accepted
    return DSparkVerifyResult(
        seq_id=seq.seq_id, cache_epoch=seq._cache_epoch,
        start_pos=seq.computed_token_num, accepted_drafts=accepted,
        committed_ids=tokens[:length].unsqueeze(0).clone(),
        next_anchor=predictions[accepted:accepted + 1].clone(),
        accepted_hidden=taps[:, :length].clone(),
    )


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
