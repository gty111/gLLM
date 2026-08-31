"""Qwen3.5 hybrid GDN + full-attention text decoder and Qwen3.5-VL wrapper.

Architectural cheat-sheet (Qwen3.5-0.8B config):

* ``num_hidden_layers = 24`` interleaved as 3x``linear_attention`` followed by
  1x``full_attention`` (``full_attention_interval = 4``), so 18 GDN layers and
  6 softmax layers per stack.
* Full-attention block uses ``attn_output_gate = True`` — the qkv projection
  outputs ``[q | gate | k | v]`` and the sigmoid-gated ``q*gate`` flows into
  the kernel (sglang Qwen3.5 ``self_attention``).
* MRoPE with ``partial_rotary_factor = 0.25`` (so only the first
  ``head_dim * 0.25`` dims of q/k are rotated) and ``mrope_interleaved =
  True``. Phase D wires the interleaved MRoPE through ``MRotaryEmbedding``;
  here we just propagate the factor.
* GDN linear-attention layer (Gated DeltaNet, fused-projection variant):

      x  -> in_proj_qkvz -> [Q, K, V, Z]   (MergedColumnParallelLinear of
                                            [K, K, V, V])
      x  -> in_proj_ba   -> [B, A]         (Merged of [Nv, Nv])
      causal_conv1d(K, K, V) -> mixed_qkv  (vendored Triton kernel)
      (g, beta) = fused_gdn_gating(A_log, A, B, dt_bias)
      core    = chunk_gated_delta_rule(...)  (prefill, vendored)
              / fused_recurrent_gated_delta_rule_packed_decode(...) (decode)
      norm    = RMSNormGated(core, Z, norm_before_gate=True)
      out     = out_proj(norm)

  ``conv_state`` (Cin, kernel) and ``ssm_state`` (Nv, Hk, Hv) live in the
  :class:`gllm.runtime.memory_manager.SSMSegment` arena view; the slot id is
  ``sequence.recurrent_state_slot`` (filled by the scheduler and pushed to GPU
  by :meth:`InputData._cal_ssm_metadata`).
* Some checkpoints ship an ``mtp.*`` multi-token-prediction head for
  speculative decoding; gllm does not load or run it yet (only ``model.*``
  and ``lm_head`` are used).

KV-cache layer accounting:

* gllm's ``QKVAttention(layer_id, ...)`` indexes ``segment.k_cache[layer_id]``
  / ``v_cache[layer_id]``. With 24 hybrid layers but only 6 full-attn layers,
  we MUST hand each full-attn layer the dense *kv-layer* index ``0..5``, not
  its global decoder index. The companion ``ssm_layer_id`` (``0..17``)
  selects the slice of the GDN state tensors in :class:`SSMSegment`. Both
  mappings are computed once in :class:`Qwen3_5Model`.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn

from gllm.distributed.parallel_state import (
    get_local_rank,
    get_pp_layers,
    get_tp_rank,
    get_tp_size,
    is_first_pp_rank,
    is_last_pp_rank,
    resolve_pp_layer_idx,
)
from gllm.runtime.input_data import InputData
from gllm.layers.attention.qkv import QKVAttention
from gllm.layers.layernorm import GemmaRMSNorm, RMSNorm
from gllm.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from gllm.layers.ops.fla import (
    RMSNormGated,
    chunk_gated_delta_rule,
    fused_gdn_gating,
    fused_recurrent_gated_delta_rule,
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_recurrent_gdn_spec,
)
from gllm.layers.ops.mamba import (
    causal_conv1d_fn,
    causal_conv1d_update,
    causal_conv1d_update_paged,
)
from gllm.layers.ops.mtp_embed_norm import (
    fused_mtp_embed_hidden_gemma_norm,
)
from gllm.layers.rotary_embedding import MRotaryEmbedding, RotaryEmbedding
from gllm.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from gllm.runtime.memory_manager import SSMCacheConfig
from gllm.models.qwen2 import Qwen2MLP
from gllm.models.qwen2_moe import Qwen2MoeSparseMoeBlock
from gllm.models.weight_utils import (
    copy_qkv_proj,
    copy_single_proj_dim0,
    copy_single_proj_dim1,
    get_tensor_from_dict,
)
from gllm.models.weight_loader import (
    LoadContext,
    WeightRule,
    contains,
    h_gate_up,
    h_proj_dim0,
    h_proj_dim1,
    h_qkv_proj_gated,
    hv_proj_dim0,
    hv_proj_dim1,
    hv_qkv_fused_split,
    make_gdn_pre_pass,
    run_vision_loader,
    run_weight_loader,
)
from gllm.runtime.piecewise_cuda_graph import piecewise_dynamic_tensor
from gllm.utils import get_model_load_pbar


_GLOBAL_LAYER_TYPE_ATTRS = ("layer_types", "layers_block_type")


def _partition_query_start_loc(
    query_start_loc: torch.Tensor,
    query_start_loc_cpu: Optional[torch.Tensor],
    row_start: int,
) -> torch.Tensor:
    """Rebase a packed suffix while preserving its CPU boundary mirror.

    Mixed decode/prefill forwards split the packed query after the decode rows.
    The CUDA slice/subtraction below necessarily creates a new tensor, so the
    private ``_cpu_view`` attached by ``InputData.get_query_start_loc`` does not
    survive. FLA then falls back to a GPU ``.tolist()`` in every GDN layer to
    rebuild identical chunk metadata. Attach the equivalently rebased CPU
    boundaries to the derived tensor so the metadata paths avoid that sync.
    """
    partition = query_start_loc[row_start:] - query_start_loc[row_start]
    if query_start_loc_cpu is not None:
        cpu_partition = query_start_loc_cpu[row_start:]
        partition._cpu_view = cpu_partition - cpu_partition[0]
    return partition


def _apply_strided_attention_output_gate(
    attn_out: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Apply a fused-projection gate without first compacting its view."""
    gated = attn_out.view_as(gate) * torch.sigmoid(gate)
    return gated.reshape(*gate.shape[:-2], -1)


def _get_layer_types(text_config) -> List[str]:
    """Return the per-decoder-layer attention type strings.

    HF's Qwen3.5 config exposes ``layer_types`` while older sglang configs
    used ``layers_block_type``; accept either to be transcript-friendly.
    """
    for attr in _GLOBAL_LAYER_TYPE_ATTRS:
        value = getattr(text_config, attr, None)
        if value is not None:
            return list(value)
    raise AttributeError(
        "Qwen3.5 text_config must define `layer_types` or `layers_block_type`."
    )


def _resolve_rope_params(config) -> Tuple[float, dict, float, bool]:
    """Pull ``(theta, scaling-dict, partial_rotary_factor, mrope_interleaved)``
    out of either ``rope_parameters`` (transformers 4.57+) or the legacy
    ``rope_theta`` / ``rope_scaling`` pair.
    """
    rope_params = getattr(config, "rope_parameters", None) or {}
    if rope_params:
        theta = float(rope_params.get("rope_theta", getattr(config, "rope_theta", 1e7)))
        partial = float(rope_params.get("partial_rotary_factor", 1.0))
        mrope_interleaved = bool(rope_params.get("mrope_interleaved", False))
        scaling = dict(rope_params)
        return theta, scaling, partial, mrope_interleaved

    theta = float(getattr(config, "rope_theta", 1e7))
    scaling = getattr(config, "rope_scaling", None) or {}
    partial = float(getattr(config, "partial_rotary_factor", 1.0))
    mrope_interleaved = bool(scaling.get("mrope_interleaved", False))
    return theta, dict(scaling), partial, mrope_interleaved


def _build_rope(head_dim: int, max_position: int, config) -> nn.Module:
    """Construct the rotary embedding module for a full-attention layer.

    Supports both vanilla RoPE and MRoPE-with-partial-rotary-factor; the
    common Qwen3.5-0.8B config takes the MRoPE branch with
    ``partial_rotary_factor = 0.25``.
    """
    theta, scaling, partial, mrope_interleaved = _resolve_rope_params(config)
    rotary_dim = max(int(round(head_dim * partial)), 2)
    if rotary_dim % 2:
        rotary_dim -= 1

    mrope_section = scaling.get("mrope_section")
    if mrope_section is not None:
        return MRotaryEmbedding(
            head_size=head_dim,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position,
            base=theta,
            is_neox_style=True,
            mrope_section=list(mrope_section),
            mrope_interleaved=mrope_interleaved,
        )
    return RotaryEmbedding(
        head_size=head_dim,
        rotary_dim=rotary_dim,
        max_position_embeddings=max_position,
        base=theta,
        is_neox_style=True,
    )


class Qwen3_5GatedDeltaNet(nn.Module):
    """Gated DeltaNet linear-attention layer.

    Follows sglang's ``Qwen3_5GatedDeltaNet`` so the in-tree port reuses the
    upstream weight names verbatim — ``in_proj_qkvz`` (merged ``[Q, K, V, Z]``
    along the output dim), ``in_proj_ba`` (merged ``[B, A]``), ``conv1d``,
    ``A_log``, ``dt_bias``, ``norm``, ``out_proj``. The actual recurrence
    runs on the vendored FLA Triton kernels at
    :mod:`gllm.layers.ops.fla`. State lives in the
    :class:`gllm.runtime.memory_manager.SSMSegment` arena view, addressed via
    ``input_data.get_recurrent_state_slot_per_seq()``.
    """

    def __init__(self, config, layer_id: int, ssm_layer_id: int,
                 quant_config=None):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.ssm_layer_id = ssm_layer_id

        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.activation = config.hidden_act
        self.layer_norm_epsilon = config.rms_norm_eps
        self.conv_dim = self.key_dim * 2 + self.value_dim

        tp_size = get_tp_size()
        if self.num_v_heads % tp_size or self.num_k_heads % tp_size:
            raise ValueError(
                "Qwen3.5 GDN requires linear_num_{k,v}_heads divisible by TP "
                f"size: tp_size={tp_size}, num_k_heads={self.num_k_heads}, "
                f"num_v_heads={self.num_v_heads}"
            )
        self.tp_num_v_heads = self.num_v_heads // tp_size
        self.tp_num_k_heads = self.num_k_heads // tp_size

        # ``in_proj_a`` / ``in_proj_b`` live in ``modules_to_not_convert`` on
        # the Qwen3.5-MoE-FP8 checkpoint (per-rank size num_v_heads / tp_size
        # is below the 128-element block granularity, so the FP8 path can't
        # validate). Conv1d input dim equals ``conv_kernel_size`` (4 on
        # Qwen3.5) which is also far below ``block_k``. Keep them in bf16
        # regardless of the global quant_config.
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=False,
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        # Merged fused-projections (match sglang names exactly). For the
        # MoE-FP8 variant ``in_proj_qkvz`` and ``out_proj`` are FP8 block-
        # quantized (the checkpoint stores ``in_proj_qkv`` and ``in_proj_z``
        # separately and we fuse them at load time).
        self.in_proj_qkvz = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[self.key_dim, self.key_dim, self.value_dim, self.value_dim],
            bias=False,
            quant_config=quant_config,
        )
        self.in_proj_ba = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[self.num_v_heads, self.num_v_heads],
            bias=False,
        )

        self.dt_bias = nn.Parameter(
            torch.ones(self.tp_num_v_heads, device="cuda")
        )
        self.A_log = nn.Parameter(
            torch.empty(
                self.tp_num_v_heads, dtype=torch.float32, device="cuda"
            )
        )

        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            group_size=None,
            norm_before_gate=True,
        )

        self.out_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
        )

    @property
    def _scale(self) -> float:
        return self.head_k_dim ** -0.5

    @torch.no_grad()
    def _split_qkvzba(self, qkvz: torch.Tensor, ba: torch.Tensor):
        """Slice the merged projections into per-component tensors.

        ``qkvz`` is laid out along the last dim as
        ``[Q_tp | K_tp | V_tp | Z_tp]`` (where ``X_tp = X / tp_size`` after
        :class:`MergedColumnParallelLinear`), so a straight ``split`` plus
        ``reshape`` recovers the per-head views expected by the FLA kernels.
        """
        k_tp = self.key_dim // get_tp_size()
        v_tp = self.value_dim // get_tp_size()
        nv_tp = self.tp_num_v_heads
        q, k, v, z = qkvz.split([k_tp, k_tp, v_tp, v_tp], dim=-1)
        b, a = ba.split([nv_tp, nv_tp], dim=-1)
        # v and z are consumed by the chunk-GDN kernel / RMSNormGated as
        # ``[T, num_v_heads, head_v_dim]``, mirroring sglang's
        # ``fix_query_key_value_ordering``. Reshape here so the caller sees
        # the per-head layout uniformly.
        v = v.reshape(v.size(0), -1, self.head_v_dim)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        return q, k, v, z, b, a

    def _ssm_state_tensors(self, input_data: InputData):
        """Return the per-layer ``conv_state`` and ``ssm_state`` views.

        ``SSMSegment`` packs all linear-attention layers into a single
        ``[L, num_state_slots, ...]`` arena view (slot 0 == CUDA-graph dummy);
        the runtime-only ``self.ssm_layer_id`` selects this layer's slice.
        """
        seg = input_data.memory_manager.ssm_segment
        return (
            seg.conv_state[self.ssm_layer_id],
            seg.temporal_state[self.ssm_layer_id],
        )

    def _maybe_snapshot_state(
        self,
        input_data: InputData,
        conv_state_working: torch.Tensor,
        ssm_state_working: torch.Tensor,
        row_start: int = 0,
    ) -> None:
        """Copy this layer's working state into reclaimable snapshot slots
        designated by ``input_data.get_ssm_snapshot_write_slot_per_seq``.

        InputData derives the valid source/destination slots once per forward
        on the host and uploads them once.  All GDN layers reuse those device
        indices, avoiding a synchronizing ``nonzero`` and repeated index/cast
        kernels for identical metadata in every layer.
        """
        copy_indices = input_data.get_ssm_snapshot_copy_indices(row_start)
        if copy_indices is None:
            return
        seg = input_data.memory_manager.ssm_segment
        src_idx, dst_idx = copy_indices

        # Capture = copy this layer's working entry -> the page's cached entry;
        # both address the same arena view (``conv_state_working`` /
        # ``ssm_state_working`` are ``seg.conv_state[ssm_layer_id]`` /
        # ``seg.temporal_state[ssm_layer_id]``). ``src_idx`` (live rolling entry)
        # and ``dst_idx`` (cached entry) have distinct live ownership -> no alias.
        conv_state_working.index_copy_(
            0, dst_idx, conv_state_working.index_select(0, src_idx)
        )
        ssm_state_working.index_copy_(
            0, dst_idx, ssm_state_working.index_select(0, src_idx)
        )
        # CPU metadata for overlap batches is speculative. Make the snapshot
        # visible to prefix lookup only after the actual device copies have
        # been enqueued. The worker cannot schedule the next batch until this
        # Python forward returns, and all recurrent-state work shares the same
        # forward stream, so publication here closes the zero-snapshot window.
        input_data.mark_ssm_snapshot_writes_enqueued()

    def _is_decode_batch(self, input_data: InputData) -> bool:
        """All-decode batches have exactly one query token per sequence and
        the seq is past prompt (``computed_prompt``). gllm partitions prefill
        and decode batches in the scheduler so a single forward is
        homogeneous; we still derive the answer from the batch's max query
        length to stay robust against future fused-batch scheduling.
        """
        return getattr(input_data, "max_query_len", 1) == 1

    def _verify_conv(
        self,
        input_data: InputData,
        mixed_qkv: torch.Tensor,
        conv_state: torch.Tensor,
        conv_weights: torch.Tensor,
        conv_bias,
        block_table_2d: torch.Tensor,
        num_accepted: torch.Tensor,
        nseq: int,
        qlen: int,
    ) -> torch.Tensor:
        """Causal conv over the verify row and checkpoint it in one kernel.

        The kernel consumes the same 2D block table as the recurrent GDN
        update: it resumes from column ``num_accepted - 1`` and writes token
        ``t``'s post-window directly to column ``t``.  Keeping gLLM's narrow
        per-block window avoids changing the allocator/commit protocol while
        eliminating the previous gather, wide scratch, intermediate window,
        and scatter launches.
        """
        c_in = mixed_qkv.shape[1]
        out = causal_conv1d_update_paged(
            mixed_qkv.view(nseq, qlen, c_in).transpose(1, 2),   # [nseq, dim, T]
            conv_state,
            conv_weights,
            block_table_2d[:, :qlen],
            num_accepted,
            bias=conv_bias,
            activation=self.activation,
        )                                                       # [nseq, dim, T]
        return out.transpose(1, 2)                              # [nseq, T, dim]

    def _forward_mtp_verify(
        self,
        input_data: InputData,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        conv_weights: torch.Tensor,
        conv_bias,
        cache_indices: torch.Tensor,
    ) -> torch.Tensor:
        """MTP verify GDN forward with 2D block-table checkpointing.

        The verify batch is ``nseq`` sequences, each a uniform ``T = 1+k`` query
        ``[x1, d1..dk]`` over its cached context. Each seq holds a ``[1+k]`` SSM
        state block table; column 0 holds its committed (pre-x1) rolling state.
        We run the recurrent GDN kernel token-by-token starting from column 0 and
        write each token ``t``'s post-state into column ``t`` of the block table
        (both temporal state and conv window). The entries use the same arena
        view as rolling state -- no separate intermediate buffer. After
        sampling, the accept step copies the committed column (``na``) back to
        column 0 (see ``model_runner._mtp_decode``), so the plain decode/snapshot
        paths keep reading column 0.

        Returns ``core_attn_out`` shaped ``[nseq*T, tp_num_v_heads, head_v_dim]``.
        """
        block_table_2d = input_data.get_ssm_block_table_2d()  # [nseq, 1+k] int32
        num_accepted = input_data.get_ssm_num_accepted()       # [nseq] int32
        assert block_table_2d is not None, "MTP verify needs an SSM block table"

        # Pure verify graphs cover every row. A mixed target forward keeps its
        # MTP rows as a contiguous prefix and passes only that token prefix into
        # this helper; slice the persistent block/accept buffers to the same
        # request partition.
        nseq = int(
            getattr(input_data, "num_mtp_verify_rows", 0)
            or block_table_2d.shape[0]
        )
        block_table_2d = block_table_2d[:nseq]
        num_accepted = num_accepted[:nseq]
        total = mixed_qkv.shape[0]
        assert total % nseq == 0, (
            f"MTP verify batch not uniform: {total} tokens / {nseq} seqs"
        )
        qlen = total // nseq  # == 1 + k
        c_in = mixed_qkv.shape[1]
        conv_out = self._verify_conv(
            input_data, mixed_qkv, conv_state, conv_weights, conv_bias,
            block_table_2d, num_accepted, nseq, qlen,
        ).reshape(total, c_in)
        kd = self.key_dim // get_tp_size()
        vd = self.value_dim // get_tp_size()
        qd_, kd_, vd_ = torch.split(conv_out, [kd, kd, vd], dim=-1)
        # The recurrent kernel accepts token-major strided views, avoiding
        # three full materializations of the fused QKV projection per layer.
        # Verify is uniform by construction. Preserve the request dimension so
        # the recurrent kernel can use its fixed-length path directly instead
        # of loading two cu-seqlen entries in every program. These remain
        # strided views of ``conv_out``; no Q/K/V materialization is introduced.
        q = qd_.reshape(nseq, qlen, self.tp_num_k_heads, self.head_k_dim)
        k = kd_.reshape(nseq, qlen, self.tp_num_k_heads, self.head_k_dim)
        v = vd_.reshape(nseq, qlen, self.tp_num_v_heads, self.head_v_dim)

        # Temporal state: recurrent kernel reads column ``num_accepted-1`` and
        # writes each verify token t's post-state into column t (in the shared
        # ``ssm_state`` arena view, addressed by the 2D block table).
        core_attn_out = fused_recurrent_gdn_spec(
            A_log=self.A_log,
            a=a,
            b=b,
            dt_bias=self.dt_bias,
            q=q,
            k=k,
            v=v,
            scale=self._scale,
            state_source=ssm_state,
            ssm_state_indices=block_table_2d,
            num_accepted_tokens=num_accepted,
            cu_seqlens=None,
            use_qk_l2norm_in_kernel=True,
        )
        return core_attn_out.reshape(total, self.tp_num_v_heads, self.head_v_dim)

    def _forward_prefill_rows(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        conv_weights: torch.Tensor,
        conv_bias,
        cache_indices: torch.Tensor,
        has_initial_state: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run ordinary ragged prefill for one contiguous request partition."""
        seq_len = mixed_qkv.shape[0]
        mixed_qkv = causal_conv1d_fn(
            mixed_qkv.transpose(0, 1),
            conv_weights,
            conv_bias,
            activation=self.activation,
            conv_states=conv_state,
            has_initial_state=has_initial_state,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            seq_lens_cpu=seq_lens_cpu,
        ).transpose(0, 1)[:seq_len]

        qd, kd, vd = torch.split(
            mixed_qkv,
            [
                self.key_dim // get_tp_size(),
                self.key_dim // get_tp_size(),
                self.value_dim // get_tp_size(),
            ],
            dim=-1,
        )
        qd = qd.view(1, seq_len, self.tp_num_k_heads, self.head_k_dim)
        kd = kd.view(1, seq_len, self.tp_num_k_heads, self.head_k_dim)
        vd = vd.view(1, seq_len, self.tp_num_v_heads, self.head_v_dim)

        g, beta = fused_gdn_gating(self.A_log, a, b, self.dt_bias)
        core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
            q=qd,
            k=kd,
            v=vd,
            g=g,
            beta=beta,
            initial_state=ssm_state,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            scale=self._scale,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )[:2]
        if last_recurrent_state is not None:
            ssm_state[cache_indices] = last_recurrent_state.to(
                ssm_state.dtype, copy=False
            )
        return core_attn_out.reshape(
            seq_len, self.tp_num_v_heads, self.head_v_dim
        )

    def _forward_decode_rows(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        conv_weights: torch.Tensor,
        conv_bias,
        cache_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the exact one-token decode path for a contiguous row prefix.

        A scheduler batch may contain ordinary decode rows followed by ragged
        prefill rows.  The recurrent state of the decode prefix must still be
        updated by the packed one-token kernels; sending those rows through the
        bulk prefill implementation changes the GDN state and makes generation
        depend strongly on when new requests join the batch.
        """
        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_state,
            conv_weights,
            conv_bias,
            self.activation,
            conv_state_indices=cache_indices,
        )
        batch_size = mixed_qkv.shape[0]
        out = torch.empty(
            (batch_size, 1, self.tp_num_v_heads, self.head_v_dim),
            dtype=mixed_qkv.dtype,
            device=mixed_qkv.device,
        )
        core_attn_out, _ = fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            scale=self._scale,
            initial_state=ssm_state,
            out=out,
            ssm_state_indices=cache_indices,
            use_qk_l2norm_in_kernel=True,
        )
        return core_attn_out.reshape(
            batch_size, self.tp_num_v_heads, self.head_v_dim
        ), mixed_qkv


    def forward(self, input_data: InputData, hidden_states: torch.Tensor):
        # Profile-run path (no allocated cache). Cheap, prevents crashes
        # during ``MemoryManager.profile``.
        if not hasattr(input_data.memory_manager, "ssm_segment") or \
                input_data.memory_manager.ssm_segment is None:
            return torch.zeros_like(hidden_states)

        qkvz = self.in_proj_qkvz(hidden_states)
        ba = self.in_proj_ba(hidden_states)
        q, k, v, z, b, a = self._split_qkvzba(qkvz, ba)

        seq_len = hidden_states.shape[0]
        # ``in_proj_qkvz`` already lays out Q, K and V as one adjacent prefix.
        # Keep that strided view instead of materializing the same bytes with a
        # per-layer ``cat``.  All causal-conv paths consume explicit strides
        # and produce their own contiguous output, so the trailing Z columns
        # in the projection's physical row stride are harmless.
        qkv_width = (
            2 * (self.key_dim // get_tp_size())
            + self.value_dim // get_tp_size()
        )
        mixed_qkv = qkvz[:, :qkv_width]
        conv_state, ssm_state = self._ssm_state_tensors(input_data)
        cache_indices = input_data.get_recurrent_state_slot_per_seq()
        has_initial_state = input_data.get_has_initial_state_per_seq()
        query_start_loc = input_data.get_query_start_loc()
        # ``conv1d.weight`` is stored as ``(C_in, 1, kernel)`` so the kernel
        # consumes it as a 2-D ``(C_in, kernel)`` view (no copy).
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), -1)
        conv_bias = self.conv1d.bias  # None for Qwen3.5

        num_mtp_rows = int(getattr(input_data, "num_mtp_verify_rows", 0))
        if num_mtp_rows:
            # MTP verify rows form a contiguous token prefix. In a pure verify
            # graph that prefix is the whole batch; in a mixed target forward
            # it is followed by ordinary ragged prefill rows. Attention handles
            # both as cached-context prefill, while GDN partitions them here so
            # only the speculative rows write checkpoint columns.
            if getattr(input_data, "is_mtp_verify", False):
                num_mtp_tokens = seq_len
            else:
                qsl_cpu = getattr(input_data, "query_start_loc_cpu", None)
                if qsl_cpu is None:
                    raise RuntimeError("mixed MTP forward needs CPU query boundaries")
                num_mtp_tokens = int(qsl_cpu[num_mtp_rows])
            if not 0 < num_mtp_tokens <= seq_len:
                raise RuntimeError(
                    f"invalid mixed MTP token prefix {num_mtp_tokens}/{seq_len}"
                )

            core_mtp = self._forward_mtp_verify(
                input_data,
                mixed_qkv[:num_mtp_tokens],
                a[:num_mtp_tokens],
                b[:num_mtp_tokens],
                conv_state,
                ssm_state,
                conv_weights,
                conv_bias,
                cache_indices[:num_mtp_rows],
            )
            if num_mtp_tokens < seq_len:
                prefill_qsl = _partition_query_start_loc(
                    query_start_loc,
                    getattr(input_data, "query_start_loc_cpu", None),
                    num_mtp_rows,
                )
                seq_lens_cpu = getattr(input_data, "seq_lens_cpu", None)
                if seq_lens_cpu is not None:
                    seq_lens_cpu = seq_lens_cpu[num_mtp_rows:]
                core_prefill = self._forward_prefill_rows(
                    mixed_qkv[num_mtp_tokens:],
                    a[num_mtp_tokens:],
                    b[num_mtp_tokens:],
                    conv_state,
                    ssm_state,
                    conv_weights,
                    conv_bias,
                    cache_indices[num_mtp_rows:],
                    has_initial_state[num_mtp_rows:],
                    prefill_qsl,
                    seq_lens_cpu,
                )
                core_attn_out = torch.cat((core_mtp, core_prefill), dim=0)
                self._maybe_snapshot_state(
                    input_data,
                    conv_state,
                    ssm_state,
                    row_start=num_mtp_rows,
                )
            else:
                core_attn_out = core_mtp
        elif (
            int(getattr(input_data, "num_decodes", 0)) > 0
            and int(getattr(input_data, "num_prefills", 0)) > 0
        ):
            # Ordinary mixed batch: the scheduler orders one-token decode rows
            # first and ragged prefill rows last. Keep the exact decode state
            # transition for the prefix, then run only the suffix through the
            # bulk varlen kernels. This mirrors the partition already used by
            # FlashInfer and by the mixed MTP target path above.
            num_decode_rows = int(input_data.num_decodes)
            num_decode_tokens = num_decode_rows
            core_decode, _ = self._forward_decode_rows(
                mixed_qkv[:num_decode_tokens],
                a[:num_decode_tokens],
                b[:num_decode_tokens],
                conv_state,
                ssm_state,
                conv_weights,
                conv_bias,
                cache_indices[:num_decode_rows],
            )
            prefill_qsl = _partition_query_start_loc(
                query_start_loc,
                getattr(input_data, "query_start_loc_cpu", None),
                num_decode_rows,
            )
            seq_lens_cpu = getattr(input_data, "seq_lens_cpu", None)
            if seq_lens_cpu is not None:
                seq_lens_cpu = seq_lens_cpu[num_decode_rows:]
            core_prefill = self._forward_prefill_rows(
                mixed_qkv[num_decode_tokens:],
                a[num_decode_tokens:],
                b[num_decode_tokens:],
                conv_state,
                ssm_state,
                conv_weights,
                conv_bias,
                cache_indices[num_decode_rows:],
                has_initial_state[num_decode_rows:],
                prefill_qsl,
                seq_lens_cpu,
            )
            core_attn_out = torch.cat((core_decode, core_prefill), dim=0)
            self._maybe_snapshot_state(
                input_data,
                conv_state,
                ssm_state,
                row_start=num_decode_rows,
            )
        elif self._is_decode_batch(input_data):
            # Decode: one new token per seq updates conv_state in-place and
            # runs the recurrent kernel for a single step. ``conv_state`` and
            # ``ssm_state`` are mutated in-place; the slot id is the row
            # index.
            core_attn_out, mixed_qkv = self._forward_decode_rows(
                mixed_qkv,
                a,
                b,
                conv_state,
                ssm_state,
                conv_weights,
                conv_bias,
                cache_indices,
            )
        else:
            # Prefill: causal_conv1d over the packed varlen sequence, then
            # ``chunk_gated_delta_rule`` for the bulk and the recurrence for
            # the tail. We follow sglang's "extend" path verbatim.
            core_attn_out = self._forward_prefill_rows(
                mixed_qkv,
                a,
                b,
                conv_state,
                ssm_state,
                conv_weights,
                conv_bias,
                cache_indices,
                has_initial_state,
                query_start_loc,
                getattr(input_data, "seq_lens_cpu", None),
            )
            # Phase G.3: persist the just-computed state into a snapshot arena
            # entry for seqs whose chunk ended on an eligible page boundary.
            # ``InputData`` borrows the entry lazily for this forward. This is
            # how cross-seq prefix-cache hits later restore the GDN
            # recurrent state into a fresh working slot. ``-1`` slots are a
            # no-op so non-PrefixSegment runs / non-cacheable boundaries
            # don't pay anything beyond a cheap mask check.
            self._maybe_snapshot_state(input_data, conv_state, ssm_state)

        z_shape = z.shape
        if core_attn_out.shape != z.shape:
            pad = torch.zeros_like(z)
            pad.reshape(-1, pad.shape[-1])[
                : core_attn_out.numel() // core_attn_out.shape[-1]
            ].copy_(core_attn_out.reshape(-1, core_attn_out.shape[-1]))
            core_attn_out = pad

        # Preserve token/head strides so the fused gate view does not need to
        # be materialized before the gated RMSNorm kernel.
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape)
        core_attn_out = core_attn_out.reshape(*core_attn_out.shape[:-2], -1)

        return self.out_proj(core_attn_out)


class Qwen3_5FullAttention(nn.Module):
    """Full-attention layer with optional output-gating + Q/K RMSNorm.

    Distinct from :class:`gllm.models.qwen3.Qwen3Attention` only because of
    ``attn_output_gate`` — when enabled, the qkv projection is widened to
    ``[Q | gate | K | V]`` and ``q * sigmoid(gate)`` flows into the kernel.
    Partial RoPE is handled by ``_build_rope`` which sets ``rotary_dim`` to
    ``head_dim * partial_rotary_factor`` (defaults to 1.0 ==> full RoPE).
    """

    def __init__(self, config, kv_layer_id: int, quant_config=None):
        super().__init__()
        self.kv_layer_id = kv_layer_id
        self.hidden_size = config.hidden_size

        tp_size = get_tp_size()
        self.total_num_heads = config.num_attention_heads
        if self.total_num_heads % tp_size:
            raise ValueError(
                f"num_attention_heads ({self.total_num_heads}) must be "
                f"divisible by TP size ({tp_size})."
            )
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads % tp_size and tp_size % self.total_num_kv_heads:
            raise ValueError(
                f"num_key_value_heads ({self.total_num_kv_heads}) must "
                f"divide or be divided by TP size ({tp_size})."
            )
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.total_num_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        self.attn_output_gate = bool(getattr(config, "attn_output_gate", False))
        # When the layer uses an output gate, the qkv projection outputs
        # ``num_heads * 2`` query rows so we can split off the gate cheaply.
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads * (2 if self.attn_output_gate else 1),
            self.total_num_kv_heads,
            bias=bool(getattr(config, "attention_bias", False)),
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
        )

        self.q_norm = GemmaRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, config.rms_norm_eps)

        self.rotary_emb = _build_rope(
            self.head_dim,
            getattr(config, "max_position_embeddings", 8192),
            config,
        )
        self.attn = QKVAttention(
            kv_layer_id,
            self.scaling,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
        )

    def forward(self, input_data: InputData, hidden_states: torch.Tensor):
        qkv = self.qkv_proj(hidden_states)
        orig_shape = qkv.shape[:-1]
        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            gate = None

        q_shape = (*orig_shape, self.q_size)
        k_shape = (*orig_shape, self.kv_size)
        q = q.view(*orig_shape, self.num_heads, self.head_dim)
        k = k.view(*orig_shape, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q).reshape(q_shape)
        k = self.k_norm(k).reshape(k_shape)
        q, k = self.rotary_emb(input_data.get_position(), q, k)

        attn_out = self.attn.forward(q, k, v, input_data)

        if gate is not None:
            attn_out = _apply_strided_attention_output_gate(attn_out, gate)
        return self.o_proj(attn_out)


def _is_moe_text_config(config) -> bool:
    """Detect the Qwen3.5-MoE variant from its text config.

    The dense Qwen3.5 checkpoint (e.g. Qwen3.5-0.8B) uses ``Qwen2MLP``-style
    dense MLPs, while the Qwen3.5-MoE variant fills every (non-mlp-only)
    layer with a sparse-MoE block + shared expert. ``num_experts > 0`` is the
    canonical signal in both the standalone text config and the VL wrapper's
    ``text_config``.
    """
    return getattr(config, "num_experts", 0) > 0


class Qwen3_5DecoderLayer(nn.Module):
    """Dispatches between the linear-attn and full-attn block."""

    supports_piecewise_cuda_graph = True

    def __init__(
        self,
        config,
        layer_id: int,
        layer_type: str,
        kv_layer_id: Optional[int],
        ssm_layer_id: Optional[int],
    ):
        super().__init__()
        self.layer_id = layer_id
        self.layer_type = layer_type
        quant_config = getattr(config, "quantization_config", None)

        if layer_type in ("linear_attention", "linear_attn"):
            assert ssm_layer_id is not None
            self.linear_attn = Qwen3_5GatedDeltaNet(
                config, layer_id, ssm_layer_id, quant_config=quant_config
            )
            self.self_attn = None
        elif layer_type in ("attention", "full_attention", "full_attn"):
            assert kv_layer_id is not None
            self.self_attn = Qwen3_5FullAttention(
                config, kv_layer_id, quant_config=quant_config
            )
            self.linear_attn = None
        else:
            raise ValueError(f"Unknown layer_type: {layer_type!r}")

        if _is_moe_text_config(config):
            # Qwen3.5-MoE: top-K routed experts + (optional) shared expert.
            # ``Qwen2MoeSparseMoeBlock`` already reads ``num_experts``,
            # ``num_experts_per_tok``, ``moe_intermediate_size``,
            # ``norm_topk_prob`` and ``shared_expert_intermediate_size`` off
            # the config and propagates ``quantization_config`` into both
            # the ``FusedMoE`` experts and the shared-expert ``Qwen2MLP``.
            mlp_only_layers = getattr(config, "mlp_only_layers", []) or []
            if layer_id in mlp_only_layers:
                self.mlp = Qwen2MLP(config)
            else:
                self.mlp = Qwen2MoeSparseMoeBlock(config)
        else:
            self.mlp = Qwen2MLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(
        self,
        input_data: InputData,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if self.self_attn is not None:
            hidden_states, residual = piecewise_dynamic_tensor(
                lambda x: self.self_attn(input_data, x), hidden_states, residual
            )
        else:
            hidden_states, residual = piecewise_dynamic_tensor(
                lambda x: self.linear_attn(input_data, x), hidden_states, residual
            )
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        # MoE routing and expert kernels are GPU-only and graph-safe for a
        # static token bucket. Its large scratch/dispatch buffers are owned by
        # the layer's piecewise workspace rather than graph-pool temporaries,
        # which keeps their addresses stable across capture and replay.
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3_5Model(nn.Module):
    """Hybrid GDN + full-attention text decoder.

    The constructor builds two parallel layer indices in step with the
    HF ``layer_types`` schedule:

    * ``self._kv_layer_ids[i] = j``  when layer ``i`` is full-attention and
      this is the j-th full-attention layer (0-indexed). ``j`` is what
      ``QKVAttention`` uses to address ``segment.k_cache[j]`` /
      ``v_cache[j]``.
    * ``self._ssm_layer_ids[i] = j`` when layer ``i`` is linear-attention.
      ``j`` selects ``SSMSegment.conv_state_working[j]`` /
      ``temporal_state_working[j]``.

    ``self.num_kv_layers`` and ``self.ssm_layer_global_ids`` are then read by
    ``model_runner.init`` and surfaced to :class:`MemoryManager` so the KV
    page and recurrent-state arena layouts match the model's real shape.
    """

    def __init__(self, config, decoder_layer_type=None):
        super().__init__()
        self.config = config
        if decoder_layer_type is None:
            decoder_layer_type = Qwen3_5DecoderLayer

        if is_first_pp_rank() or (
            getattr(config, "tie_word_embeddings", False) and is_last_pp_rank()
        ):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
            )

        self.start_layer, self.end_layer = get_pp_layers(config.num_hidden_layers)

        layer_types = _get_layer_types(config)
        if len(layer_types) != config.num_hidden_layers:
            raise ValueError(
                "Length of `layer_types` does not match num_hidden_layers: "
                f"{len(layer_types)} vs {config.num_hidden_layers}"
            )

        # Build dense KV / SSM layer indices for the layers that belong to
        # this pipeline rank. The model only allocates the slice
        # ``[start_layer, end_layer)`` so the local indices reset at the PP
        # boundary — KV and recurrent cache views are also sized per-rank.
        self._layer_types: List[str] = []
        self._kv_layer_ids: List[Optional[int]] = []
        self._ssm_layer_ids: List[Optional[int]] = []

        kv_counter = 0
        ssm_counter = 0
        layers: List[Qwen3_5DecoderLayer] = []
        for global_idx in range(self.start_layer, self.end_layer):
            lt = layer_types[global_idx]
            if lt in ("linear_attention", "linear_attn"):
                self._layer_types.append("linear_attention")
                self._kv_layer_ids.append(None)
                self._ssm_layer_ids.append(ssm_counter)
                layers.append(
                    decoder_layer_type(
                        config,
                        layer_id=global_idx,
                        layer_type="linear_attention",
                        kv_layer_id=None,
                        ssm_layer_id=ssm_counter,
                    )
                )
                ssm_counter += 1
            else:
                self._layer_types.append("full_attention")
                self._kv_layer_ids.append(kv_counter)
                self._ssm_layer_ids.append(None)
                layers.append(
                    decoder_layer_type(
                        config,
                        layer_id=global_idx,
                        layer_type="full_attention",
                        kv_layer_id=kv_counter,
                        ssm_layer_id=None,
                    )
                )
                kv_counter += 1
        self.layers = nn.ModuleList(layers)
        self.num_kv_layers = kv_counter
        self.num_ssm_layers = ssm_counter
        # Global linear-attn layer indices (for diagnostics / config); the
        # SSMSegment uses a dense layer axis, so we just need the count.
        self.ssm_layer_global_ids = [
            self.start_layer + i
            for i, lt in enumerate(self._layer_types)
            if lt == "linear_attention"
        ]

        if is_last_pp_rank():
            self.norm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        input_data: InputData,
        hidden_states: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
        deepstack_input_embeds=None,
    ):
        if is_first_pp_rank() and hidden_states is None:
            hidden_states = self.embed_tokens(input_data.get_tokens())

        for local_idx, layer in enumerate(self.layers):
            global_idx = local_idx + self.start_layer
            hidden_states, residual = layer(input_data, hidden_states, residual)
            if (
                deepstack_input_embeds is not None
                and f"deepstack_input_embeds_{global_idx}" in deepstack_input_embeds
            ):
                hidden_states = (
                    hidden_states
                    + deepstack_input_embeds[f"deepstack_input_embeds_{global_idx}"]
                )

        if not is_last_pp_rank():
            return hidden_states, residual
        # The MTP head consumes this POST-norm output, matching vLLM
        # (``target_hidden_states = hidden_states``) and sglang
        # (``spec_info.hidden_states``); the pre-norm residual is not it.
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)


# ---------------------------------------------------------------------------
# weight-loading helper for the GDN layer
# ---------------------------------------------------------------------------


def _tp_slice_dim0(tensor: torch.Tensor, total_partition_size: int) -> torch.Tensor:
    """Slice a checkpoint tensor along dim 0 for the current TP rank."""
    tp_size = get_tp_size()
    rank = get_tp_rank()
    chunk = total_partition_size // tp_size
    return tensor[rank * chunk : (rank + 1) * chunk]


def _load_gdn_layer_weights(layer: Qwen3_5GatedDeltaNet, prefix: str, weights):
    """Load all parameters of one :class:`Qwen3_5GatedDeltaNet` block.

    The HF/Qwen3.5 checkpoint exposes the GDN block as four split
    projections (``in_proj_qkv``, ``in_proj_z``, ``in_proj_b``, ``in_proj_a``)
    plus ``conv1d``, ``A_log``, ``dt_bias``, ``norm.weight`` and
    ``out_proj.weight``. We collapse the splits into the merged
    ``in_proj_qkvz`` / ``in_proj_ba`` parameters that match sglang's layout.

    All slicing is TP-rank-local: the source checkpoint stores the full
    tensors and we keep only this rank's share. The slicing pattern matches
    ``mamba_v2_sharded_weight_loader`` (per-component sharding along output
    dim).

    When the linear projections are FP8 block-quantized
    (``in_proj_qkv``/``in_proj_z``/``out_proj`` on the Qwen3.5-MoE-FP8
    checkpoint), the corresponding ``weight_scale_inv`` tensors are loaded
    in the exact same shape but divided by ``block_n`` along dim 0; the
    fusion-aware per-component slicing stays valid because every
    ``key_dim``/``value_dim`` slice is a multiple of ``block_n`` for the
    target geometry (key_dim=2048, value_dim=4096, block_n=128).
    """
    src = lambda name: get_tensor_from_dict(weights, f"{prefix}.{name}")
    tp_size = get_tp_size()
    rank = get_tp_rank()

    is_fp8_qkvz = hasattr(layer.in_proj_qkvz, "weight_scale_inv")
    is_fp8_out = hasattr(layer.out_proj, "weight_scale_inv")
    block_n = (
        layer.in_proj_qkvz.weight_block_size[0] if is_fp8_qkvz else None
    )

    # ---- conv1d ----
    conv_weight = src("conv1d.weight")
    if conv_weight.ndim == 3:
        conv_weight = conv_weight.squeeze(1)
    parts = []
    cursor = 0
    for total in (layer.key_dim, layer.key_dim, layer.value_dim):
        slc = conv_weight[cursor : cursor + total]
        cursor += total
        chunk = total // tp_size
        parts.append(slc[rank * chunk : (rank + 1) * chunk])
    layer.conv1d.weight.data.copy_(
        torch.cat(parts, dim=0).unsqueeze(1).contiguous()
    )

    def _fuse_qkvz(suffix: str, scale_div: int):
        """Slice + fuse the per-component projections into the merged tensor.

        ``scale_div`` is 1 for the FP8 weight tensor (rows stay row-aligned)
        and ``block_n`` for the ``weight_scale_inv`` tensor (each row of the
        scale covers ``block_n`` rows of the weight along dim 0). The TP-
        local component sizes (``key_dim/tp_size``, ``value_dim/tp_size``)
        are multiples of ``block_n`` by construction for Qwen3.5-MoE, so
        the divisions below are exact.
        """
        parts = []
        qkv = src(f"in_proj_qkv.{suffix}")
        cursor = 0
        for sub in (layer.key_dim, layer.key_dim, layer.value_dim):
            sub_s = sub // scale_div
            chunk = sub_s // tp_size
            parts.append(qkv[cursor + rank * chunk : cursor + (rank + 1) * chunk])
            cursor += sub_s
        z = src(f"in_proj_z.{suffix}")
        z_chunk = (layer.value_dim // scale_div) // tp_size
        parts.append(z[rank * z_chunk : (rank + 1) * z_chunk])
        return torch.cat(parts, dim=0)

    layer.in_proj_qkvz.weight.data.copy_(_fuse_qkvz("weight", 1))
    if is_fp8_qkvz:
        layer.in_proj_qkvz.weight_scale_inv.data.copy_(
            _fuse_qkvz("weight_scale_inv", block_n)
        )

    # ---- in_proj_ba: merge [B, A] along dim 0 (always bf16) ----
    ba_parts = []
    for name, total in (
        ("in_proj_b.weight", layer.num_v_heads),
        ("in_proj_a.weight", layer.num_v_heads),
    ):
        w = src(name)
        chunk = total // tp_size
        ba_parts.append(w[rank * chunk : (rank + 1) * chunk])
    layer.in_proj_ba.weight.data.copy_(torch.cat(ba_parts, dim=0))

    layer.A_log.data.copy_(_tp_slice_dim0(src("A_log"), layer.num_v_heads))
    layer.dt_bias.data.copy_(_tp_slice_dim0(src("dt_bias"), layer.num_v_heads))

    layer.norm.weight.data.copy_(src("norm.weight"))

    copy_single_proj_dim1(layer.out_proj.weight.data, src("out_proj.weight"))
    if is_fp8_out:
        copy_single_proj_dim1(
            layer.out_proj.weight_scale_inv.data,
            src("out_proj.weight_scale_inv"),
        )


# ---------------------------------------------------------------------------
# Qwen3_5MTP (Multi-Token Prediction head for speculative decoding)
# ---------------------------------------------------------------------------


class Qwen3_5MTP(nn.Module):
    """A single Qwen3.5 MTP (NextN) head — the checkpoint's ``mtp.*`` block.

    Unlike DeepSeek's MLA-based MTP head, the Qwen3.5 head is ONE
    *full-attention* decoder layer (``mtp.layers.0.*``) plus the fusion
    projection ``fc`` and its two pre-norms. It reuses the base model's token
    embedding and (tied) LM head — the checkpoint ships neither a dedicated
    ``embed_tokens`` (``mtp_use_dedicated_embeddings = false``) nor a separate
    LM head (``tie_word_embeddings = true``).

    Forward (matches sglang ``Qwen3_5ForCausalLMMTP``)::

        e = pre_fc_norm_embedding(embed(input_ids))   # GemmaRMSNorm
        h = pre_fc_norm_hidden(prev_hidden)           # GemmaRMSNorm
        x = fc(cat([e, h], dim=-1))                   # [2H -> H], embed first
        x, residual = mtp_block(x)                     # one full-attn decoder layer
        x = residual + x
        x = norm(x)                                    # GemmaRMSNorm
        logits = lm_head(x)                            # tied to embed_tokens

    ``kv_layer_id`` is the dense full-attention slot the head's attention uses
    to index the shared paged KV cache (it is ``num_kv_layers`` of the base
    model — one past the last base full-attn layer). The head shares the target
    model's paged arena the same way DeepSeek's does. Draft steps extend its
    slot, while target prefill/verify passes replay the head to replace accepted
    positions with KV derived from authoritative target hidden states.
    """

    def __init__(self, config, kv_layer_id: int, parent_model: "Qwen3_5Model"):
        super().__init__()
        self.config = config
        self.kv_layer_id = kv_layer_id
        self._parent_model = [parent_model]  # list to avoid registering as submodule
        self._lm_head = []  # set by the owning ForCausalLM (base tied LM head)
        eps = config.rms_norm_eps
        hidden = config.hidden_size
        quant_config = getattr(config, "quantization_config", None)

        # Fusion of (next-token embedding, previous hidden state).
        self.pre_fc_norm_embedding = GemmaRMSNorm(hidden, eps)
        self.pre_fc_norm_hidden = GemmaRMSNorm(hidden, eps)
        # fc: [2*hidden -> hidden], replicated (small, unsharded in ckpt).
        self.fc = ReplicatedLinear(hidden * 2, hidden, bias=False)

        # One full-attention decoder layer (the MTP block, ``mtp.layers.0``).
        self.mtp_block = Qwen3_5DecoderLayer(
            config,
            layer_id=0,
            layer_type="full_attention",
            kv_layer_id=kv_layer_id,
            ssm_layer_id=None,
        )

        # Final norm (``mtp.norm``); the LM head is the base model's (tied).
        self.norm = GemmaRMSNorm(hidden, eps)

    @property
    def _embed(self):
        return self._parent_model[0].embed_tokens

    def forward(
        self,
        input_data: InputData,
        prev_hidden: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return the MTP block's post-norm hidden state ``[num_tokens, hidden]``.

        ``prev_hidden``: post-final-norm target hidden state of the position whose
        *next* token we are drafting. ``input_ids``: the already-known token id
        at each of those positions (the token the draft is conditioned on).
        """
        embed = self._embed
        if (
            getattr(embed, "tp_size", 1) == 1
            and input_ids.is_cuda
            and input_ids.is_contiguous()
            and prev_hidden.is_cuda
            and prev_hidden.stride(-1) == 1
            and embed.weight.stride(-1) == 1
        ):
            # TP=1 owns the complete table. Gather both MTP inputs, preserve
            # the established FP32 Gemma reduction, and write directly into
            # the FC input layout.
            eh = fused_mtp_embed_hidden_gemma_norm(
                input_ids,
                embed.weight,
                prev_hidden,
                self.pre_fc_norm_embedding.weight,
                self.pre_fc_norm_hidden.weight,
                self.pre_fc_norm_embedding.variance_epsilon,
            )
        else:
            e = self.pre_fc_norm_embedding(embed(input_ids))
            h = self.pre_fc_norm_hidden(prev_hidden)
            eh = torch.cat([e, h], dim=-1)
        x = self.fc(eh)
        # mtp_block is a standard decoder layer: (input_data, hidden, residual).
        x, residual = self.mtp_block(input_data, x, None)
        x, _ = self.norm(x, residual)
        return x

    def logits_from_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # The MTP head shares the base model's (tied) LM head.
        return self._lm_head[0](hidden_states)

    def _src_key(self, param_name: str) -> str:
        """Map an MTP-module parameter name to its checkpoint key.

        The checkpoint stores the head under ``mtp.``:

            fc / pre_fc_norm_embedding / pre_fc_norm_hidden / norm -> mtp.<same>
            mtp_block.self_attn.<x>   -> mtp.layers.0.self_attn.<x>
            mtp_block.mlp.<x>         -> mtp.layers.0.mlp.<x>
            mtp_block.input_layernorm / post_attention_layernorm
                                      -> mtp.layers.0.<same>

        The gated fused ``qkv_proj`` / fused ``gate_up_proj`` param names are
        preserved so the parent's rule handlers (which ``str.replace`` them back
        to ``q_proj`` / ``gate_proj`` / ``up_proj``) find the split checkpoint
        tensors.
        """
        if param_name.startswith("mtp_block."):
            rest = param_name[len("mtp_block."):]
            return f"mtp.layers.0.{rest}"
        # fc / pre_fc_norm_embedding / pre_fc_norm_hidden / norm: verbatim.
        return f"mtp.{param_name}"

    def load_weights(self, weights, parent_lm, mp_load_progress=None):
        """Load the ``mtp.*`` head weights, reusing the base rule table.

        ``parent_lm`` is the built :class:`Qwen3_5ForCausalLM`; we borrow its
        ``weight_rules()`` + ``LoadContext`` (gated-qkv geometry) and remap each
        of THIS module's parameter names to the checkpoint's ``mtp.*`` key via
        :meth:`_src_key`, so the same handlers that loaded the base full-attn
        layers load the MTP block unchanged.
        """
        rules = parent_lm.weight_rules()
        # The embed/lm_head rule never matches an MTP param (the head has no own
        # embed/lm_head — it shares the base's), so drop it for clarity.
        rules = [r for r in rules if r.name != "embed_lm_head"]
        ctx = parent_lm._make_load_context(weights)
        for name, p in dict(self.named_parameters()).items():
            src = self._src_key(name)
            for rule in rules:
                if rule.match(src):
                    rule.handler(ctx, src, p.data)
                    break
            else:
                p.data.copy_(get_tensor_from_dict(weights, src))


# ---------------------------------------------------------------------------
# Qwen3_5ForCausalLM
# ---------------------------------------------------------------------------


class Qwen3_5ForCausalLM(nn.Module):
    """Text-only Qwen3.5 causal LM."""

    def __init__(self, config, model_type=Qwen3_5Model):
        super().__init__()
        self.config = config
        self.model = model_type(config)
        self.max_model_len = config.max_position_embeddings
        self.num_layers = len(self.model.layers)
        self.start_layer = self.model.start_layer
        self.end_layer = self.model.end_layer
        # SSM bookkeeping that ``model_runner`` reads when sizing the
        # SSMSegment. ``num_kv_layers`` is a property (accounts for the MTP
        # head's extra full-attn slot) so it is not assigned here.
        self.num_ssm_layers = self.model.num_ssm_layers
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.ret_residual = True
        self.ssm_cache_config = self._build_ssm_cache_config(config)

        if is_last_pp_rank():
            self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
            if getattr(config, "tie_word_embeddings", False):
                self.lm_head.tie_weights(self.model.embed_tokens)

        # Optional MTP (Multi-Token Prediction) head for speculative decoding.
        # Built only when the checkpoint ships an ``mtp.*`` block
        # (``mtp_num_hidden_layers >= 1``) AND MTP is enabled by the runner
        # (``config.mtp_enabled``, resolved from the ``mtp_enabled`` CLI/arg with
        # auto-detect). Only on the last PP rank, where the base ``lm_head`` /
        # final hidden live. The head's full-attention block uses the KV slot
        # just past the base full-attention layers (``num_kv_layers``).
        self.mtp = None
        num_mtp = getattr(config, "mtp_num_hidden_layers", 0) or 0
        want_mtp = getattr(config, "mtp_enabled", False) and num_mtp >= 1
        if want_mtp and is_last_pp_rank():
            self.mtp = Qwen3_5MTP(
                config,
                kv_layer_id=self.model.num_kv_layers,
                parent_model=self.model,
            )
            self.mtp._lm_head = [self.lm_head]

    @property
    def num_kv_layers(self):
        # The MTP block runs its own full-attention layer and writes into the
        # KV slot just past the base full-attn layers, so the MemoryManager
        # must size one extra layer when the head is present. (PP=1: the head is
        # on this rank; PP>1: only the last rank, which is also the only rank
        # whose full-attn slots reach the top.)
        base = self.model.num_kv_layers
        return base + 1 if self.mtp is not None else base

    def _build_ssm_cache_config(self, config) -> SSMCacheConfig:
        """Compose the per-rank :class:`SSMCacheConfig`.

        ``num_layers`` is the count of GDN layers on this PP rank;
        ``conv_dim`` is the per-rank packed projection width (``2*K + V``)
        because :class:`Qwen3_5GatedDeltaNet` shards along the head dim;
        ``head_v_dim`` etc. are unchanged by TP.

        The recurrent-state dtype comes from the engine's
        ``mamba_ssm_cache_dtype`` (see ``ModelRunner``). At ``auto`` (the
        default), honour the checkpoint's ``mamba_ssm_dtype`` recommendation.
        This matters for speculative decoding: a multi-token verify keeps the
        recurrence in fp32 between
        tokens, so a bf16 state cache would no longer be bit-equivalent to
        ordinary one-token decode, which writes and reloads the cache after
        every token.
        """
        tp_size = get_tp_size()
        key_dim = config.linear_num_key_heads * config.linear_key_head_dim
        value_dim = config.linear_num_value_heads * config.linear_value_head_dim
        conv_dim_per_partition = (2 * key_dim + value_dim) // tp_size
        # Conv state needs to match the activation dtype so the conv1d
        # kernels can ``tl.load`` from it without an implicit cast. We use
        # the current default dtype (set by the engine to the checkpoint
        # dtype before instantiating the model).
        conv_state_dtype = torch.get_default_dtype()
        dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16,
                     "float16": torch.float16}
        req = str(getattr(config, "mamba_ssm_cache_dtype", "auto")).lower()
        if req == "auto":
            req = str(getattr(config, "mamba_ssm_dtype", "auto")).lower()
        ssm_dtype = dtype_map.get(req, conv_state_dtype)
        return SSMCacheConfig(
            num_layers=self.num_ssm_layers,
            conv_dim=conv_dim_per_partition,
            conv_kernel=config.linear_conv_kernel_dim,
            num_v_heads=config.linear_num_value_heads // tp_size,
            head_v_dim=config.linear_value_head_dim,
            head_k_dim=config.linear_key_head_dim,
            dtype=ssm_dtype,
            conv_state_dtype=conv_state_dtype,
            ssm_layer_ids=list(self.model.ssm_layer_global_ids),
        )

    def forward(self, input_data: InputData, hidden_states=None, residual=None):
        return self.model(input_data, hidden_states, residual)

    def compute_logits(self, input_data: InputData, hidden_states: torch.Tensor):
        idx = input_data.get_query_start_loc() - 1
        return self.logits_from_hidden(hidden_states[idx[1:]])

    def logits_from_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project the given hidden states to full-vocab logits.

        ``compute_logits`` gathers only each seq's last position (for
        sampling); this projects *every* supplied position and is used by the
        prompt-logprobs path.
        """
        return self.lm_head(hidden_states)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    # ----- weight loading --------------------------------------------------

    # Sub-keys of a GDN block filled en bloc by ``_load_gdn_layer_weights``
    # (and thus skipped in the per-parameter loop). Includes both bf16 and FP8
    # (``*_scale_inv``) variants; absent ones are ignored by the pre-pass.
    GDN_SUBS = (
        "conv1d.weight",
        "in_proj_qkvz.weight",
        "in_proj_qkvz.weight_scale_inv",
        "in_proj_ba.weight",
        "A_log",
        "dt_bias",
        "norm.weight",
        "out_proj.weight",
        "out_proj.weight_scale_inv",
    )

    def _attn_ref(self):
        for layer in self.model.layers:
            if layer.self_attn is not None:
                return layer.self_attn
        return None

    def _make_load_context(self, weights):
        attn = self._attn_ref()
        num_q_heads = attn.num_heads if attn is not None else 0
        num_kv_heads = attn.num_kv_heads if attn is not None else 0
        head_dim = attn.head_dim if attn is not None else self.head_dim
        num_q_rows = num_q_heads * (
            2 if (attn is not None and attn.attn_output_gate) else 1
        )
        return LoadContext(
            weights=weights,
            num_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            extra={"num_q_rows": num_q_rows, "_attn_ref": attn},
        )

    def weight_rules(self):
        return [
            WeightRule(
                contains("self_attn.qkv_proj"), h_qkv_proj_gated, "qkv_proj"
            ),
            WeightRule(contains("self_attn.o_proj"), h_proj_dim1, "o_proj"),
            WeightRule(contains("gate_up_proj"), h_gate_up, "gate_up_proj"),
            WeightRule(contains("down_proj"), h_proj_dim1, "down_proj"),
            WeightRule(
                contains("embed_tokens", "lm_head"), h_proj_dim0, "embed_lm_head"
            ),
        ]

    def load_weights(self, weights, mp_load_progress=None):
        # The base loader iterates ``self.named_parameters()`` and looks each up
        # by its parameter path under ``model.*`` / ``lm_head``. The MTP head's
        # params live under ``mtp.*`` (no ``model.layers.N`` path), so detach it
        # for the base pass and load it separately below.
        mtp = self.mtp
        self.mtp = None
        try:
            ctx = self._make_load_context(weights)
            # qkv rule only fires when a full-attention layer exists; if none,
            # ``num_q_rows`` is 0 and no qkv_proj parameters are present anyway.
            rules = self.weight_rules()
            if ctx.extra.get("_attn_ref") is None:
                rules = [r for r in rules if r.name != "qkv_proj"]
            run_weight_loader(
                self,
                weights,
                rules,
                mp_load_progress,
                pp_idx_offset=2,
                start_layer=self.start_layer,
                ctx=ctx,
                pre_passes=[make_gdn_pre_pass(self.GDN_SUBS, _load_gdn_layer_weights)],
            )
        finally:
            self.mtp = mtp
        # Then the MTP head's ``mtp.*`` weights, reusing this model's rule table
        # + load context (see Qwen3_5MTP.load_weights).
        if self.mtp is not None:
            self.mtp.load_weights(weights, self, mp_load_progress)


# ---------------------------------------------------------------------------
# Qwen3_5ForConditionalGeneration (VL wrapper)
# ---------------------------------------------------------------------------

# The VL wrapper is intentionally a thin subclass of
# :class:`Qwen3VLForConditionalGeneration` so we reuse the entire vision
# stack (patch embed, vision transformer blocks, deepstack mergers) and only
# override the language model. The wrapper's ``load_weights`` is reimplemented
# here because the parent's loader uses Qwen3-text projection names while our
# language model exposes the GDN/full-attn hybrid names.

from gllm.models.qwen3_vl import Qwen3VLForConditionalGeneration  # noqa: E402


class Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """Qwen3.5-VL: ``Qwen3-VL vision tower + Qwen3.5 hybrid text LM``."""

    def __init__(self, config):
        super().__init__(config, language_model_type=Qwen3_5ForCausalLM)
        # Encoder-disaggregation encoder process: no language model at all.
        if getattr(self, "skip_language", False) or self.language_model is None:
            self.ssm_cache_config = None
            self.num_kv_layers = 0
            self.num_ssm_layers = 0
            return
        # Expose ssm_cache_config and num_kv_layers at top-level so that
        # ``ModelRunner.init`` (which reads ``self.model``) finds them
        # without having to peek into ``self.language_model``.
        self.ssm_cache_config = self.language_model.ssm_cache_config
        self.num_kv_layers = self.language_model.num_kv_layers
        self.num_ssm_layers = self.language_model.num_ssm_layers

    @property
    def mtp(self):
        # ModelRunner reads ``self.model.mtp`` to detect / drive the MTP head.
        # It lives on the language model; surface it at the wrapper level.
        # (``compute_logits`` / ``logits_from_hidden`` are already delegated to
        # the language model by the Qwen3-VL parent.)
        lm = getattr(self, "language_model", None)
        return getattr(lm, "mtp", None) if lm is not None else None

    def load_weights(self, weights, mp_load_progress=None):
        # Language model load is delegated; it walks ``self.language_model``'s
        # named_parameters() and slices each tensor for the current TP rank.
        if not getattr(self, "skip_language", False) and self.language_model is not None:
            self.language_model.load_weights(weights, mp_load_progress)

        if not is_first_pp_rank():
            return

        # Encoder-disaggregation LM node skips the vision tower entirely.
        if getattr(self, "skip_visual", False) or self.visual is None:
            return

        # Visual tower load: same pattern as ``Qwen3VLForConditionalGeneration``.
        ctx = LoadContext(
            weights=weights,
            num_heads=self.visual.num_heads // get_tp_size(),
            head_dim=self.visual.hidden_size // self.visual.num_heads,
            extra={"prefix": "visual."},
        )
        rules = [
            WeightRule(contains("attn.qkv"), hv_qkv_fused_split, "v_qkv"),
            WeightRule(
                contains("attn.proj.weight", "linear_fc2.weight"),
                hv_proj_dim1,
                "v_proj_dim1",
            ),
            WeightRule(contains("linear_fc1"), hv_proj_dim0, "v_fc1"),
        ]
        run_vision_loader(self.visual, weights, rules, ctx)
