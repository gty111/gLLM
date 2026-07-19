"""DeepSeek-V3.2: DeepSeek-V3 + DeepSeek Sparse Attention (DSA).

V3.2's config and weights are a strict superset of DeepSeek-V3 -- identical MLA
projections, MoE block, YaRN RoPE and FP8 block-quant -- so everything except
attention is inherited unchanged from :mod:`gllm.models.deepseek_v2` (which
already backs V3/R1).

The one architectural addition is a per-layer **lightning indexer** (DSA): a
lightweight side path (``self_attn.indexer.{wq_b, wk, k_norm, weights_proj}``)
that scores every cached key against each query and keeps only the top
``index_topk`` (2048) tokens, which are then fed as a sparse mask into MLA
attention. For any sequence no longer than ``index_topk`` the top-k selects
*all* keys, so DSA is mathematically identical to dense MLA there.

Staged bring-up:

* **Stage 1 (this file's initial form):** construct the indexer so its weights
  load, but run the inherited *dense* MLA forward. Exact for context
  <= ``index_topk``; an approximation (attends to all keys) beyond it.
* **Stage 3/4 (TODO):** feed the indexer's top-k selection into the sparse
  FlashMLA kernels (``flash_mla_with_kvcache(indices=...)`` /
  ``flash_mla_sparse_fwd``) for true sparse attention at long context.

Reference: HuggingFace ``transformers>=5.11`` ``modeling_deepseek_v32`` /
``modular_deepseek_v32`` (the concise diff-from-V3).
"""

from typing import Optional

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from gllm.dist_utils import get_pp_layers, is_first_pp_rank, is_last_pp_rank
from gllm.input_data import InputData
from gllm.layers.layernorm import RMSNorm
from gllm.layers.linear import ReplicatedLinear
from gllm.layers.rotary_embedding import YaRNScalingRotaryEmbedding
from gllm.layers.vocab_parallel_embedding import VocabParallelEmbedding

from .deepseek_v2 import (
    DeepseekV2DecoderLayer,
    DeepseekV2ForCausalLM,
    DeepseekV2MLAAttention,
)
from .qwen2_moe import Qwen2MoeForCausalLM
from .utils import extract_rope_config

# Position-axis tile for the DSA decode indexer score (see
# ``_select_topk_decode``). The indexer must score a decode query against its
# full, statically-sized KV history for CUDA-graph safety; scoring all of it at
# once materializes a [num_decode, max_L, dim] fp32 key gather that OOMs at large
# decode buckets. Scoring in fixed-size position tiles caps the scratch at
# [num_decode, tile, dim] while keeping the loop count static (graph-safe).
_INDEX_SCORE_TILE = 512

# Query-axis chunk for the prefill DSA indexer selection (see
# ``_select_topk_prefill``). Prefill scores many query tokens at once; the
# per-query key gather is ``[chunk, max_L, dim]``, so an unbounded query axis at
# long context OOMs. Chunking the query axis caps peak scratch. Prefill runs
# eagerly (never CUDA-graph captured), so a Python loop here is fine.
_INDEX_QUERY_CHUNK = 512

# DSA prefill indexer scoring backend. Default: exact fp32 einsum
# (``_select_topk_prefill``). With ``GLLM_DSA_FP8_SCORE=1`` the prefill selector
# instead uses SGLang's FP8 path (Hadamard -> e4m3 quant -> deep_gemm
# ``fp8_mqa_logits``): ~10-50x faster indexer scoring at long context, at the
# cost of FP8 score quantization (~13% mean rel err on the logits, but the
# indexer only *selects* top-k, so end-to-end impact is validated on RULER).
_DSA_FP8_SCORE = os.environ.get("GLLM_DSA_FP8_SCORE", "0") == "1"


class DeepseekV32Indexer(nn.Module):
    """DeepSeek Sparse Attention (DSA) lightning indexer.

    Scores every query against the (cached) keys with a lightweight
    single-head-per-token projection and returns the ``index_topk`` key indices
    with the highest scores per query. Mirrors the bf16 reference indexer:
    scores are ``ReLU(q . k) * softmax_scale`` weighted per head by
    ``weights_proj`` and summed across heads, then causally masked before the
    top-k.

    Unlike the main MLA attention (which uses interleaved RoPE), the indexer
    applies **non-interleaved (neox / half-split) RoPE** to the first
    ``qk_rope_head_dim`` dims of its head -- hence its own rotary embedding
    instance with ``is_neox_style=True``.

    NOTE: not yet wired into the attention hot path (see module docstring); the
    module exists so its weights load, and its ``forward`` is validated /
    consumed in the sparse-attention stages.
    """

    def __init__(self, config, layer_id: int):
        super().__init__()
        quant_config = getattr(config, "quantization_config", None)

        self.layer_id = layer_id
        self.hidden_size: int = config.hidden_size
        self.n_heads: int = config.index_n_heads
        self.head_dim: int = config.index_head_dim
        self.qk_rope_head_dim: int = config.qk_rope_head_dim
        self.index_topk: int = config.index_topk
        self.q_lora_rank: int = config.q_lora_rank
        self.softmax_scale = self.head_dim**-0.5

        # wq_b / wk are FP8 block-quant in released checkpoints; both are
        # replicated (not TP-sharded) -- the indexer is cheap and its per-head
        # scores are summed, so sharding buys little and complicates the top-k.
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
        )
        self.wk = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
        )
        # k_norm is a LayerNorm (has bias), unlike the RMSNorms elsewhere in the
        # model. weights_proj stays bf16/unquantized (the reference keeps it in
        # fp32; we compute the head-weighting in fp32 in ``forward``).
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = ReplicatedLinear(
            self.hidden_size,
            self.n_heads,
            bias=False,
            quant_config=None,
        )

        # Same YaRN parameters as the main MLA RoPE, but neox-style (half-split)
        # application on the rope slice.
        rope_theta, rope_scaling = extract_rope_config(config, default_theta=10000.0)
        max_pos = getattr(config, "max_position_embeddings", 8192)
        if rope_scaling is None:
            scaling = {"factor": 1.0, "original_max_position_embeddings": max_pos}
        else:
            scaling = dict(rope_scaling)
            scaling.setdefault("factor", 1.0)
            scaling.setdefault("original_max_position_embeddings", max_pos)

        extra = {
            k: v
            for k, v in scaling.items()
            if k in ("extrapolation_factor", "attn_factor", "beta_fast", "beta_slow")
        }
        if scaling:
            extra["mscale"] = float(scaling.get("mscale", 1))
            extra["mscale_all_dim"] = float(scaling.get("mscale_all_dim", 0))

        self.rotary_emb = YaRNScalingRotaryEmbedding(
            self.qk_rope_head_dim,
            rotary_dim=self.qk_rope_head_dim,
            max_position_embeddings=scaling["original_max_position_embeddings"],
            base=rope_theta,
            is_neox_style=True,
            scaling_factor=scaling["factor"],
            **extra,
        )

    @torch.no_grad()
    def compute_qk(
        self,
        position: torch.Tensor,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the indexer query and (this step's) key with norm + rope.

        Returns ``(q, k)`` where ``q`` is ``[num_tokens, n_heads, head_dim]``
        and ``k`` is ``[num_tokens, head_dim]`` (single head). The rope applied
        here is NON-interleaved (neox) on the first ``qk_rope_head_dim`` dims,
        which differs from the interleaved rope of the main MLA attention.
        """
        num_tokens = hidden_states.shape[0]
        q = self.wq_b(q_resid).view(num_tokens, self.n_heads, self.head_dim)
        k = self.k_norm(self.wk(hidden_states)).view(num_tokens, 1, self.head_dim)

        q_rot = q[..., : self.qk_rope_head_dim].contiguous()
        k_rot = k[..., : self.qk_rope_head_dim].contiguous()
        q_rot, k_rot = self.rotary_emb(position, q_rot, k_rot)
        q = torch.cat([q_rot, q[..., self.qk_rope_head_dim :]], dim=-1)
        k = torch.cat([k_rot, k[..., self.qk_rope_head_dim :]], dim=-1).squeeze(1)
        return q, k

    @torch.no_grad()
    def head_weights(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Per-head score weights ``[num_tokens, n_heads]`` (fp32).

        The reference keeps ``weights_proj`` in fp32; we compute in fp32 and
        fold in the ``n_heads ** -0.5`` normalization so callers can directly
        contract with the per-head ReLU'd scores.
        """
        return self.weights_proj(hidden_states).float() * (self.n_heads**-0.5)

    @torch.no_grad()
    def score(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Indexer logits ``[num_q, num_k]`` = Σ_h w[·,h]·ReLU(softmax_scale·q·k).

        ``q``: ``[num_q, n_heads, head_dim]``; ``k``: ``[num_k, head_dim]``;
        ``weights``: ``[num_q, n_heads]``. Computed in fp32 to match the bf16
        reference (which also accumulates the score in fp32).
        """
        scores = torch.matmul(q.float(), k.float().t()) * self.softmax_scale
        scores = F.relu(scores)  # [num_q, n_heads, num_k]
        return torch.einsum("qhk,qh->qk", scores, weights)


class DeepseekV32MLAAttention(DeepseekV2MLAAttention):
    """DeepSeek-V3 MLA attention + the DSA lightning indexer.

    Overrides the model-level attention ``forward`` to, in addition to the
    inherited MLA projection + RoPE:

    1. compute the indexer query/key (``compute_qk``, neox RoPE),
    2. store the indexer key into the parallel paged index cache
       (``memory_manager.store_index_k``) by slot mapping,
    3. score each query against the sequence's cached index keys and select the
       top ``index_topk`` token positions,
    4. pass those indices into :class:`MLAAttention` so it runs the *sparse*
       FlashMLA kernels.

    For sequences no longer than ``index_topk`` (2048) the top-k selects every
    key, so the sparse result is exactly the dense result -- this is the
    correctness oracle used to validate the sparse path.
    """

    def __init__(self, layer_id: int, config):
        super().__init__(layer_id, config)
        self.indexer = DeepseekV32Indexer(config, layer_id)

    def _topk_slots(
        self,
        flat_index_cache: torch.Tensor,
        block_table: torch.Tensor,
        valid: torch.Tensor,
        q_idx: torch.Tensor,
        weights: torch.Tensor,
        page_sz: int,
    ) -> torch.Tensor:
        """Shared indexer top-k: score every query row against its addressable
        key history and return the top ``index_topk`` **physical KV slots**.

        Used by both the decode and prefill selectors -- they differ only in the
        ``valid`` mask (decode: ``p < seq_len``; prefill: per-query causal). All
        shapes are static in ``max_L = block_table.shape[1] * page_sz`` for
        CUDA-graph safety, and scoring is tiled over the position axis so the
        peak key-gather scratch is ``[rows, tile, dim]`` rather than the full
        ``[rows, max_L, dim]`` (which OOMs at large batches / long context).

        Args:
            flat_index_cache: ``[pages * page_sz, dim]`` flattened index-K cache.
            block_table: ``[rows, max_blocks]`` per-row paged block table (row ==
                one query; for prefill it is the query's sequence's row, expanded).
            valid: ``[rows, max_L]`` bool; True where the query may attend.
            q_idx: ``[rows, n_heads, head_dim]`` indexer queries.
            weights: ``[rows, n_heads]`` per-head score weights (fp32).
            page_sz: KV page size.

        Returns:
            ``[rows, index_topk]`` int32 physical slots, ``-1``-padded.
        """
        rows = q_idx.shape[0]
        topk = self.indexer.index_topk
        device = flat_index_cache.device
        max_L = block_table.shape[1] * page_sz

        # token position -> flat physical slot, batched over all rows:
        #   slot[i, p] = block_table[i, p // page_sz] * page_sz + p % page_sz
        pos = torch.arange(max_L, device=device)  # [max_L]
        blocks = block_table[:, pos // page_sz]  # [rows, max_L]
        slots = (blocks * page_sz + (pos % page_sz)).to(torch.int32)  # [rows, max_L]

        # Score every (row, pos): logits[i,p] = Σ_h w[i,h]·ReLU(scale·q[i,h,:]·k[i,p,:]).
        # Tile the position axis (see docstring): fixed tile + static (ceil) tile
        # count keep every per-tile tensor's shape static (CUDA-graph safe) while
        # capping scratch at [rows, tile, dim].
        qf = q_idx.float()  # [rows, n_heads, head_dim]
        scale = self.indexer.softmax_scale
        logits = torch.full((rows, max_L), float("-inf"), device=device)
        tile = _INDEX_SCORE_TILE
        for start in range(0, max_L, tile):
            end = min(start + tile, max_L)
            slots_t = slots[:, start:end]  # [rows, t]
            keys_t = flat_index_cache[slots_t.long()].float()  # [rows, t, dim]
            sc = torch.einsum("ihd,ipd->ihp", qf, keys_t) * scale
            sc = F.relu(sc)
            sc = torch.einsum("ihp,ih->ip", sc, weights)  # [rows, t]
            logits[:, start:end] = sc
        logits = logits.masked_fill(~valid, float("-inf"))

        # Top-k positions per row, mapped back to physical slots. ``k`` is fixed
        # to ``topk`` (static) for CUDA-graph safety; rows with fewer valid keys
        # select padding positions, masked to -1.
        k = min(topk, max_L)
        sel = logits.topk(k, dim=-1).indices  # [rows, k] positions in [0, max_L)
        sel_slots = torch.gather(slots, 1, sel)  # [rows, k]
        sel_valid = torch.gather(valid, 1, sel)
        sel_slots = torch.where(sel_valid, sel_slots, sel_slots.new_full((), -1))

        out = q_idx.new_full((rows, topk), -1, dtype=torch.int32)
        out[:, :k] = sel_slots
        return out

    def _select_topk_decode(
        self, input_data: InputData, q_idx: torch.Tensor, weights: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Per-decode-query top-k PHYSICAL KV slots ``[num_decode, index_topk]``.

        Scores each decode query against the *entire* cached index-key history
        of its sequence (gathered from the paged index cache via the per-seq
        block table + sequence length), selects the top ``index_topk``, and
        returns their **absolute physical cache slots** (int32, padded with -1).

        Physical slots (not per-seq token positions) are what FlashMLA's sparse
        decode kernel consumes as ``indices`` — it does not re-apply the
        64-block ``block_table`` to sparse indices, so the page-table transform
        (token position -> physical slot) is folded in here. ``causal`` is
        implicit: a decode query attends to all ``seq_len`` cached keys, which
        already end at the current token.
        """
        meta = input_data.metadata
        if meta is None or meta.num_decodes == 0:
            return None
        seg = input_data.memory_manager.segment
        index_cache = seg.index_k_cache[self.layer_id]  # [pages, page_sz, D]
        pages, page_sz, dim = index_cache.shape
        flat = index_cache.view(pages * page_sz, dim)

        decode = meta.decode
        block_table = decode.block_table  # [num_decode, max_blocks]
        seq_lens = decode.seq_lens  # [num_decode] int32 (GPU)
        device = flat.device
        max_L = block_table.shape[1] * page_sz

        # Decode: query attends to all cached positions < its seq length.
        pos = torch.arange(max_L, device=device)
        valid = pos.unsqueeze(0) < seq_lens.unsqueeze(1)  # [num_decode, max_L]
        return self._topk_slots(flat, block_table, valid, q_idx, weights, page_sz)

    @torch.no_grad()
    def _select_topk_decode_fp8(
        self, input_data: InputData, q_idx: torch.Tensor, weights: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """FP8 decode selector via ``deep_gemm.fp8_paged_mqa_logits``.

        Same contract/output as :meth:`_select_topk_decode` (per-decode-query top
        ``index_topk`` physical KV slots, -1 padded) but scores with the paged
        FP8 MQA-logits kernel over the persistent FP8 index-K cache. Both the
        metadata call and the kernel are CUDA-graph-capturable (verified), and
        all shapes are static in ``max_L``/``index_topk``, so this stays
        graph-safe on the captured decode path. ``fp8_paged_mqa_logits`` applies
        ReLU internally and folds the per-head ``weights`` (carrying q_scale *
        softmax_scale) + the cache's per-token scale.
        """
        import deep_gemm
        from sgl_kernel import hadamard_transform

        meta = input_data.metadata
        if meta is None or meta.num_decodes == 0:
            return None
        seg = input_data.memory_manager.segment
        fp8_cache = seg.index_k_fp8_cache[self.layer_id]  # [pages, page_sz*132] uint8
        pages = fp8_cache.shape[0]
        page_sz = seg.page_size
        dim = seg.index_head_dim
        fp8_bytes = seg.index_fp8_bytes  # 132
        decode = meta.decode
        block_table = decode.block_table  # [num_decode, max_blocks] int32
        seq_lens = decode.seq_lens  # [num_decode] int32 (GPU)
        num_decode = q_idx.shape[0]
        topk = self.indexer.index_topk
        device = q_idx.device
        max_L = block_table.shape[1] * page_sz

        # Hadamard + FP8-quant the query (matches the stored key path).
        qh = hadamard_transform(q_idx.contiguous(), scale=dim ** -0.5)
        qf = qh.float().reshape(num_decode, self.indexer.n_heads, dim // 128, 128)
        q_scale = (qf.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448.0)
        q_fp8 = (qf / q_scale).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(
            num_decode, self.indexer.n_heads, dim
        )
        q_scale = q_scale.reshape(num_decode, self.indexer.n_heads)
        w = (weights.float() * q_scale * self.indexer.softmax_scale).contiguous()

        kv = fp8_cache.view(pages, page_sz, 1, fp8_bytes)
        sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        sched = deep_gemm.get_paged_mqa_logits_metadata(seq_lens, page_sz, sm_count)
        logits = deep_gemm.fp8_paged_mqa_logits(
            q_fp8.unsqueeze(1),   # [num_decode, next_n=1, H, D]
            kv,
            w,
            seq_lens,
            block_table,
            sched,
            max_L,
            clean_logits=False,
        )  # [num_decode, max_L] fp32 (positions beyond seq_len are -inf/invalid)

        # Positions -> physical slots via the block table (same as _topk_slots).
        pos = torch.arange(max_L, device=device)
        blocks = block_table[:, pos // page_sz]
        slots = (blocks * page_sz + (pos % page_sz)).to(torch.int32)  # [nd, max_L]
        valid = pos.unsqueeze(0) < seq_lens.unsqueeze(1)
        logits = logits.masked_fill(~valid, float("-inf"))

        k = min(topk, max_L)
        sel = logits.topk(k, dim=-1).indices  # [nd, k] positions
        sel_slots = torch.gather(slots, 1, sel)
        sel_valid = torch.gather(valid, 1, sel)
        sel_slots = torch.where(sel_valid, sel_slots, sel_slots.new_full((), -1))
        out = q_idx.new_full((num_decode, topk), -1, dtype=torch.int32)
        out[:, :k] = sel_slots
        return out

    def _select_topk_prefill(
        self, input_data: InputData, q_idx: torch.Tensor, weights: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Per-prefill-query top-k PHYSICAL KV slots ``[num_prefill_tokens, index_topk]``.

        Like :meth:`_select_topk_decode`, but prefill has many query tokens per
        sequence, each with its own **causal horizon**: the query at absolute
        position ``abs_pos`` (= its seq's cached ``context_len`` + its intra-step
        offset) may attend only to key positions ``[0, abs_pos]``. We build a
        per-query ``valid`` mask encoding that, expand each prefill sequence's
        block-table row across its query tokens, and reuse the shared
        :meth:`_topk_slots` scorer.

        For any query whose causal horizon ``abs_pos + 1 <= index_topk`` the
        top-k selects *all* its valid keys, so the sparse result is exactly the
        dense result there (the correctness oracle for prompts <= index_topk).
        """
        meta = input_data.metadata
        if meta is None or meta.num_prefills == 0:
            return None
        prefill = meta.prefill
        if prefill is None or prefill.seq_lens is None:
            return None
        seg = input_data.memory_manager.segment
        index_cache = seg.index_k_cache[self.layer_id]  # [pages, page_sz, D]
        pages, page_sz, dim = index_cache.shape
        flat = index_cache.view(pages * page_sz, dim)

        block_table = prefill.block_table  # [num_prefills, max_blocks]
        qsl = prefill.query_start_loc  # [num_prefills + 1] int32 (GPU)
        seq_lens = prefill.seq_lens  # [num_prefills] int32 (GPU)
        context_lens = prefill.context_lens  # [num_prefills] int32 (GPU)
        num_prefill_tokens = q_idx.shape[0]
        device = flat.device

        # Prefill runs EAGERLY (only decode buckets are CUDA-graph captured), so
        # -- unlike the decode selector -- we do NOT need a static, full-width
        # ``max_L``. Using the full block-table width here would allocate
        # ``[num_prefill_tokens, max_blocks*page_sz]`` scratch (multiple GB at long
        # context / large prefill batches, OOMing the profile run). Instead bound
        # the scored history to the actual longest sequence in this batch and
        # slice the block table to just those blocks. A ``.item()`` sync is fine
        # off the graph path (the pre-static decode code did the same).
        max_seq = int(seq_lens.max().item())
        n_blocks = (max_seq + page_sz - 1) // page_sz
        n_blocks = max(1, min(n_blocks, block_table.shape[1]))
        block_table = block_table[:, :n_blocks]
        max_L = n_blocks * page_sz

        # Map each prefill query token -> its sequence index, its per-seq block
        # table row, and its absolute position. ``query_start_loc`` is the
        # cumulative query-token offset per prefill seq, so a bucketize of the
        # flat token id into ``qsl`` gives its seq index.
        tok = torch.arange(num_prefill_tokens, device=device)
        # seq_of_tok[t] = index i such that qsl[i] <= t < qsl[i+1].
        seq_of_tok = torch.bucketize(tok, qsl[1:], right=True)  # [num_prefill_tokens]
        seq_of_tok = seq_of_tok.clamp_(max=block_table.shape[0] - 1)

        # abs_pos[t] = context_len[seq] + (t - qsl[seq]).
        intra = tok - qsl[seq_of_tok]
        abs_pos = context_lens[seq_of_tok] + intra  # [num_prefill_tokens]
        seq_len_of_tok = seq_lens[seq_of_tok]  # [num_prefill_tokens]

        pos = torch.arange(max_L, device=device)  # [max_L]

        # Chunk over the QUERY axis: the per-query key gather in ``_topk_slots``
        # is ``[chunk, max_L, dim]``, which at long context * many prefill tokens
        # (e.g. an 8k-token prefill) would be multiple GB. Bounding the query
        # chunk keeps peak scratch at ``[_INDEX_QUERY_CHUNK, max_L, dim]``.
        topk = self.indexer.index_topk
        out = q_idx.new_full((num_prefill_tokens, topk), -1, dtype=torch.int32)
        qchunk = _INDEX_QUERY_CHUNK
        for s in range(0, num_prefill_tokens, qchunk):
            e = min(s + qchunk, num_prefill_tokens)
            rows = seq_of_tok[s:e]
            per_tok_block_table = block_table[rows]  # [chunk, n_blocks]
            # valid[t, p] = p <= abs_pos[t]  (causal; also < seq_len, implied
            # since abs_pos < seq_len, but clamp defensively against padding).
            valid = (pos.unsqueeze(0) <= abs_pos[s:e].unsqueeze(1)) & (
                pos.unsqueeze(0) < seq_len_of_tok[s:e].unsqueeze(1)
            )  # [chunk, max_L]
            out[s:e] = self._topk_slots(
                flat, per_tok_block_table, valid, q_idx[s:e], weights[s:e], page_sz
            )
        return out

    @torch.no_grad()
    def _select_topk_prefill_fp8(
        self, input_data: InputData, q_idx: torch.Tensor, weights: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """FP8 variant of :meth:`_select_topk_prefill` using ``fp8_mqa_logits``.

        Per prefill sequence: gather its cached index-key history (physical
        slots), Hadamard-transform + FP8-quantize both the indexer query and key
        (SGLang's NSA recipe), score with ``deep_gemm.fp8_mqa_logits`` under the
        per-query causal bounds ``ks/ke``, top-k, and map back to physical slots.
        ``fp8_mqa_logits`` applies ReLU internally and folds the per-head
        ``weights`` (which carry ``q_scale * softmax_scale``), matching the exact
        ``Σ_h w·ReLU(scale·q·k)`` score. Much faster than the fp32 einsum at long
        context; the FP8 quantization error only perturbs the *selection*.

        Gathering + quantizing K per forward (rather than a persistent FP8 index
        cache) keeps this change local to the model; a persistent FP8 index-K
        cache is a later perf optimization.
        """
        import deep_gemm
        from sgl_kernel import hadamard_transform

        meta = input_data.metadata
        prefill = meta.prefill
        seg = input_data.memory_manager.segment
        index_cache = seg.index_k_cache[self.layer_id]  # [pages, page_sz, D]
        pages, page_sz, dim = index_cache.shape
        flat = index_cache.view(pages * page_sz, dim)  # [num_slots, D]

        block_table = prefill.block_table  # [num_prefills, max_blocks]
        qsl_cpu = prefill.query_start_loc.cpu().tolist()  # [num_prefills + 1]
        seq_lens_cpu = prefill.seq_lens.cpu().tolist()
        context_lens_cpu = prefill.context_lens.cpu().tolist()
        num_prefill_tokens = q_idx.shape[0]
        topk = self.indexer.index_topk
        scale = self.indexer.softmax_scale
        device = flat.device
        hdim = q_idx.shape[-1]

        out = q_idx.new_full((num_prefill_tokens, topk), -1, dtype=torch.int32)

        def _quant(x):  # x [..., D] bf16 -> (e4m3, per-D-block fp32 scale)
            xf = x.float().reshape(*x.shape[:-1], hdim // 128, 128)
            amax = xf.abs().amax(-1, keepdim=True).clamp_min(1e-4)
            s = amax / 448.0
            q = (xf / s).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(x.shape)
            return q, s.reshape(*x.shape[:-1], hdim // 128).squeeze(-1)

        for i in range(len(seq_lens_cpu)):
            q0, q1 = qsl_cpu[i], qsl_cpu[i + 1]
            qlen = q1 - q0
            if qlen == 0:
                continue
            seq_len = seq_lens_cpu[i]
            ctx_len = context_lens_cpu[i]
            # Physical slots for this seq's whole history [0, seq_len).
            posk = torch.arange(seq_len, device=device)
            blk = block_table[i, posk // page_sz]
            seq_slots = (blk * page_sz + (posk % page_sz)).to(torch.int64)  # [seq_len]
            k_bf16 = flat[seq_slots]  # [seq_len, D]

            q_seq = q_idx[q0:q1]  # [qlen, n_heads, D]
            w_seq = weights[q0:q1]  # [qlen, n_heads]

            # Hadamard + FP8 quant (SGLang NSA recipe).
            qh = hadamard_transform(q_seq.contiguous(), scale=hdim ** -0.5)
            kh = hadamard_transform(k_bf16.contiguous(), scale=hdim ** -0.5)
            q_fp8, q_scale = _quant(qh)  # [qlen,H,D], [qlen,H]
            k_fp8, k_scale = _quant(kh)  # [seq_len,D], [seq_len]
            w_folded = w_seq.float() * q_scale * scale  # [qlen, n_heads]

            # Per-query causal bounds: query at intra-seq offset j (abs pos
            # ctx_len + j) attends to keys [0, ctx_len + j].
            j = torch.arange(qlen, device=device, dtype=torch.int32)
            ks = torch.zeros(qlen, device=device, dtype=torch.int32)
            ke = (ctx_len + j + 1).clamp(max=seq_len).to(torch.int32)

            logits = deep_gemm.fp8_mqa_logits(
                q_fp8.contiguous(),
                (k_fp8.contiguous(), k_scale.contiguous()),
                w_folded.contiguous(),
                ks, ke, clean_logits=False,
            )  # [qlen, seq_len]

            k_sel = min(topk, seq_len)
            sel = logits.topk(k_sel, dim=-1).indices  # [qlen, k_sel] positions
            sel_slots = seq_slots[sel].to(torch.int32)  # [qlen, k_sel]
            # positions beyond a query's causal horizon are invalid -> -1.
            sel_valid = sel < ke.unsqueeze(1)
            sel_slots = torch.where(sel_valid, sel_slots, sel_slots.new_full((), -1))
            out[q0:q1, :k_sel] = sel_slots
        return out

    def forward(
        self,
        input_data: InputData,
        hidden_states: torch.Tensor,
    ):
        # --- inherited MLA projections + RoPE (mirrors DeepseekV2MLAAttention) ---
        assert self.q_lora_rank is not None, "V3.2 always uses q_lora_rank."
        qkv_lora = self.fused_qkv_a_proj(hidden_states)
        q_c, kv_lora = qkv_lora.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
        )
        q_c = self.q_a_layernorm(q_c)
        q = self.q_b_proj(q_c)

        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c)

        q = q.view(-1, self.num_heads, self.qk_head_dim)
        k_pe = k_pe.unsqueeze(1)
        position = input_data.get_position()
        q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
            position, q[..., self.qk_nope_head_dim :], k_pe
        )

        # --- DSA indexer: key store + top-k selection ---
        decode_topk = None
        prefill_topk = None
        mm = input_data.memory_manager
        meta = input_data.metadata
        # Skip during the profile run (no KV segment) / DP-EP zero-fill step.
        if getattr(mm, "segment", None) is not None and meta is not None:
            idx_q, idx_k = self.indexer.compute_qk(position, hidden_states, q_c)
            weights = self.indexer.head_weights(hidden_states)
            # Store this step's index keys into the paged index cache.
            mm.store_index_k(
                self.layer_id, idx_k, input_data.get_slot_mapping()
            )
            # DSA FP8 scoring also needs the key in the persistent paged FP8
            # index cache (block-contiguous 132B layout for fp8_paged_mqa_logits).
            # The FP8 score path scores Hadamard(q)·Hadamard(k) (the Hadamard
            # transform decorrelates activations so FP8 quant is accurate); the
            # decode query is Hadamard'd in ``_select_topk_decode_fp8``, so the
            # stored key MUST be Hadamard'd here to match -- otherwise the score
            # is Hadamard(q)·k, which is wrong (H is orthogonal: (Hq)·(Hk)=q·k
            # but (Hq)·k != q·k).
            if _DSA_FP8_SCORE and mm.segment.index_k_fp8_cache is not None:
                from sgl_kernel import hadamard_transform

                idx_k_had = hadamard_transform(
                    idx_k.contiguous(), scale=idx_k.shape[-1] ** -0.5
                )
                mm.store_index_k_fp8(
                    self.layer_id, idx_k_had, input_data.get_slot_mapping()
                )
            num_dec = meta.num_decode_tokens
            # DeepSeek Sparse Attention: feed the indexer's top-k selection into
            # the sparse decode kernel. MLAAttention routes by cache dtype:
            # FP8-packed cache -> FlashMLA sparse ``fwd_kvcache_mla``; bf16 cache
            # -> FA3 with the top-k physical slots as a page_size=1 page table
            # (paged bf16 sparse is rejected by FlashMLA on SM90). Either way
            # sparse decode runs regardless of cache precision.
            if meta.num_decodes > 0:
                sel_decode = (
                    self._select_topk_decode_fp8
                    if _DSA_FP8_SCORE
                    else self._select_topk_decode
                )
                decode_topk = sel_decode(
                    input_data, idx_q[:num_dec], weights[:num_dec]
                )
            # DeepSeek Sparse Attention (prefill): per-query causal top-k over the
            # cached index-key history, fed into the sparse prefill kernel. Exact
            # (== dense) for any query whose causal horizon <= index_topk, so this
            # is a no-op for prompts <= index_topk and only changes long context.
            if meta.num_prefills > 0:
                sel_prefill = (
                    self._select_topk_prefill_fp8
                    if _DSA_FP8_SCORE
                    else self._select_topk_prefill
                )
                prefill_topk = sel_prefill(
                    input_data, idx_q[num_dec:], weights[num_dec:]
                )

        output_shape = (hidden_states.shape[0], self.num_heads * self.v_head_dim)
        output = torch.zeros(output_shape, dtype=q.dtype)

        attn_out = self.mla_attn.forward(
            q,
            kv_c_normed,
            k_pe,
            input_data=input_data,
            output=output,
            decode_topk_indices=decode_topk,
            prefill_topk_indices=prefill_topk,
        )
        return self.o_proj(attn_out)


class DeepseekV32DecoderLayer(DeepseekV2DecoderLayer):
    """DeepSeek-V3 decoder layer whose MLA attention carries the DSA indexer.

    The base ``__init__`` builds the (identical) MLP/MoE + norms and a
    ``DeepseekV2MLAAttention``; we replace ``self_attn`` with the V3.2 variant
    that adds the indexer + sparse selection. V3.2 is always MLA
    (``config.use_mla`` forced True), so ``self_attn`` is always the MLA path.
    """

    def __init__(self, glb_layer_id: int, layer_id: int, config):
        super().__init__(glb_layer_id, layer_id, config)
        self.self_attn = DeepseekV32MLAAttention(layer_id, config)


class DeepseekV32Model(nn.Module):
    """Same as :class:`DeepseekV2Model` but built from
    :class:`DeepseekV32DecoderLayer` (indexer-carrying attention).

    :class:`DeepseekV2Model` hardcodes ``DeepseekV2DecoderLayer`` in its layer
    list and takes no decoder-layer-type parameter, so the layer construction is
    reproduced here with the V3.2 decoder layer. Everything else (PP layer
    range, embedding, final norm, forward) is identical.
    """

    def __init__(self, config):
        super().__init__()

        if is_first_pp_rank():
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size, config.hidden_size
            )
        self.start_layer, self.end_layer = get_pp_layers(config.num_hidden_layers)
        self.layers = nn.ModuleList(
            [
                DeepseekV32DecoderLayer(i, i - self.start_layer, config)
                for i in range(self.start_layer, self.end_layer)
            ]
        )
        if is_last_pp_rank():
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_data: InputData, hidden_states=None, residual=None):
        if is_first_pp_rank() and hidden_states is None:
            hidden_states = self.embed_tokens(input_data.get_tokens())
        for layer in self.layers:
            hidden_states, residual = layer(input_data, hidden_states, residual)
        if is_last_pp_rank():
            hidden_states, _ = self.norm(hidden_states, residual)
            return hidden_states
        return hidden_states, residual


class DeepseekV32ForCausalLM(DeepseekV2ForCausalLM):
    """DeepSeek-V3.2 causal LM.

    Inherits every V3 weight rule and MoE/MLA loading logic from
    :class:`DeepseekV2ForCausalLM`; only the backing model class differs (V3.2
    decoder layers with the DSA indexer). The indexer's replicated linears /
    norms are not matched by any MLA / MoE / dense rule, so they fall through to
    the default verbatim copy in the weight loader -- exactly right for unsharded
    parameters (FP8 ``weight`` + ``weight_scale_inv``, LayerNorm weight/bias,
    bf16 ``weights_proj``).

    ``DeepseekV2ForCausalLM.__init__`` hardcodes ``DeepseekV2Model``, so this
    bypasses it and calls the grandparent (:class:`Qwen2MoeForCausalLM`) with the
    V3.2 model class, then reproduces the tiny MLA ``head_dim`` derivation.
    """

    def __init__(self, config):
        Qwen2MoeForCausalLM.__init__(self, config, model_type=DeepseekV32Model)
        attn = self.model.layers[0].self_attn
        # V3.2 is always MLA (loader forces config.use_mla=True).
        self.head_dim = attn.kv_lora_rank + attn.qk_rope_head_dim
        # Surface the DSA indexer key-cache head dim so the ModelRunner sizes a
        # parallel paged index cache in the MemoryManager (see Segment).
        self.index_head_dim = config.index_head_dim
        # MLA rope head dim, needed to size the native FP8 MLA latent cache
        # (DSA sparse decode on SM90 requires the FP8-packed layout).
        self.qk_rope_head_dim = config.qk_rope_head_dim

