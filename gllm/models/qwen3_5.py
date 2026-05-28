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
  :class:`gllm.memory_manager.SSMSegment` working pool; the slot id is
  ``sequence.ssm_state_slot`` (filled by the scheduler and pushed to GPU
  by :meth:`InputData._cal_ssm_metadata`).
* Some checkpoints ship an ``mtp.*`` multi-token-prediction head for
  speculative decoding; gllm does not load or run it yet (only ``model.*``
  and ``lm_head`` are used).

KV-cache layer accounting:

* gllm's ``FlashAttention(layer_id, ...)`` indexes ``segment.k_cache[layer_id]``
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

from gllm.dist_utils import (
    get_local_rank,
    get_pp_layers,
    get_tp_rank,
    get_tp_size,
    is_first_pp_rank,
    is_last_pp_rank,
    resolve_pp_layer_idx,
)
from gllm.input_data import InputData
from gllm.layers.attention import FlashAttention
from gllm.layers.layernorm import GemmaRMSNorm, RMSNorm
from gllm.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from gllm.layers.ops.fla import (
    RMSNormGated,
    chunk_gated_delta_rule,
    fused_gdn_gating,
    fused_recurrent_gated_delta_rule,
    fused_recurrent_gated_delta_rule_packed_decode,
)
from gllm.layers.ops.mamba import causal_conv1d_fn, causal_conv1d_update
from gllm.layers.rotary_embedding import MRotaryEmbedding, RotaryEmbedding
from gllm.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from gllm.memory_manager import SSMCacheConfig
from gllm.models.qwen2 import Qwen2MLP
from gllm.models.qwen2_moe import Qwen2MoeSparseMoeBlock
from gllm.models.weight_utils import (
    copy_gate_up_proj,
    copy_qkv_proj,
    copy_single_proj_dim0,
    copy_single_proj_dim1,
    get_tensor_from_dict,
)
from gllm.utils import get_model_load_pbar


_GLOBAL_LAYER_TYPE_ATTRS = ("layer_types", "layers_block_type")


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
    :class:`gllm.memory_manager.SSMSegment` working pool, addressed via
    ``input_data.get_ssm_state_slot_per_seq()``.
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

        self.dt_bias = nn.Parameter(torch.ones(self.tp_num_v_heads))
        self.A_log = nn.Parameter(
            torch.empty(self.tp_num_v_heads, dtype=torch.float32)
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
        ``[L, working_pool_size, ...]`` tensor (slot 0 == CUDA-graph dummy);
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
    ) -> None:
        """Copy this layer's working state into the snapshot pool slots
        designated by ``input_data.get_ssm_snapshot_write_slot_per_seq``.

        Indexed copy is vectorized via ``index_select`` + ``index_copy_``
        so the cost is one small kernel per linear-attn layer per forward,
        not a Python loop over sequences. ``valid_mask`` filters out the
        ``-1`` rows (decode steps, non-cacheable boundaries, CUDA-graph
        padding) cheaply on-device, so the indices passed to the copy
        kernel only contain real snapshot targets.
        """
        snap_targets = input_data.get_ssm_snapshot_write_slot_per_seq()
        if snap_targets is None:
            return
        seg = input_data.memory_manager.ssm_segment
        if seg.conv_state_snap is None:
            return
        # Host-side early-exit: the original ``bool(valid_mask.any())`` check
        # forced one ``cudaStreamSynchronize`` per linear-attn layer per
        # prefill step (18 layers × 32 prefill steps in the 16k/c8 profile
        # contributed ~1.2s of host wait). The CPU mirror
        # ``ssm_snapshot_write_slot_per_seq_cpu`` is already populated in
        # ``InputData`` and lags the GPU tensor by 0 steps, so we can decide
        # whether any target is non-negative without touching the GPU stream.
        snap_cpu = getattr(
            input_data, "ssm_snapshot_write_slot_per_seq_cpu", None
        )
        if snap_cpu is not None:
            if int(snap_cpu.amax()) < 0:
                return
            valid_mask = snap_targets >= 0
        else:
            valid_mask = snap_targets >= 0
            if not bool(valid_mask.any()):
                return
        src_slots = input_data.get_ssm_state_slot_per_seq()
        valid_idx = valid_mask.nonzero(as_tuple=False).squeeze(-1)
        src_idx = src_slots.index_select(0, valid_idx).to(torch.long)
        dst_idx = snap_targets.index_select(0, valid_idx).to(torch.long)

        conv_snap = seg.conv_state_snap[self.ssm_layer_id]
        temporal_snap = seg.temporal_state_snap[self.ssm_layer_id]

        conv_snap.index_copy_(
            0, dst_idx, conv_state_working.index_select(0, src_idx)
        )
        temporal_snap.index_copy_(
            0, dst_idx, ssm_state_working.index_select(0, src_idx)
        )

    def _is_decode_batch(self, input_data: InputData) -> bool:
        """All-decode batches have exactly one query token per sequence and
        the seq is past prompt (``computed_prompt``). gllm partitions prefill
        and decode batches in the scheduler so a single forward is
        homogeneous; we still derive the answer from the batch's max query
        length to stay robust against future fused-batch scheduling.
        """
        return getattr(input_data, "max_query_len", 1) == 1

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
        # Flatten per-head dims into one channel dim for ``causal_conv1d_*``;
        # the kernels expect ``mixed_qkv`` as ``[T, C_in]`` (prefill) or
        # ``[B, C_in]`` (decode).
        mixed_qkv = torch.cat(
            (q.reshape(seq_len, -1), k.reshape(seq_len, -1), v.reshape(seq_len, -1)),
            dim=-1,
        )

        conv_state, ssm_state = self._ssm_state_tensors(input_data)
        cache_indices = input_data.get_ssm_state_slot_per_seq()
        has_initial_state = input_data.get_has_initial_state_per_seq()
        query_start_loc = input_data.get_query_start_loc()
        # ``conv1d.weight`` is stored as ``(C_in, 1, kernel)`` so the kernel
        # consumes it as a 2-D ``(C_in, kernel)`` view (no copy).
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), -1)
        conv_bias = self.conv1d.bias  # None for Qwen3.5

        if self._is_decode_batch(input_data):
            # Decode: one new token per seq updates conv_state in-place and
            # runs the recurrent kernel for a single step. ``conv_state`` and
            # ``ssm_state`` are mutated in-place; the slot id is the row
            # index.
            mixed_qkv = causal_conv1d_update(
                mixed_qkv,
                conv_state,
                conv_weights,
                conv_bias,
                self.activation,
                conv_state_indices=cache_indices,
            )

            # Fast path: a single fused Triton kernel that splits the packed
            # ``mixed_qkv`` and runs the recurrence (sgl's
            # ``packed_decode``). The kernel reads ``initial_state`` indexed
            # by ``ssm_state_indices`` and writes the new state in-place at
            # the same slots — matching sglang's contract.
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
        else:
            # Prefill: causal_conv1d over the packed varlen sequence, then
            # ``chunk_gated_delta_rule`` for the bulk and the recurrence for
            # the tail. We follow sglang's "extend" path verbatim.
            mixed_qkv = mixed_qkv.transpose(0, 1)  # [C_in, T]
            mixed_qkv = causal_conv1d_fn(
                mixed_qkv,
                conv_weights,
                conv_bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                seq_lens_cpu=getattr(input_data, "seq_lens_cpu", None),
            ).transpose(0, 1)[:seq_len]

            qd, kd, vd = torch.split(
                mixed_qkv, [self.key_dim // get_tp_size(),
                            self.key_dim // get_tp_size(),
                            self.value_dim // get_tp_size()], dim=-1)
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
                # Qwen3.5 trains with QK L2-normalization baked into the
                # kernel — matches HF (use_qk_l2norm_in_kernel=True) and
                # sglang (same flag). Without it, the recurrence diverges
                # from the trained behavior right away.
                use_qk_l2norm_in_kernel=True,
            )[:2]
            if last_recurrent_state is not None:
                ssm_state[cache_indices] = last_recurrent_state.to(
                    ssm_state.dtype, copy=False
                )

            # Phase G.3: persist the just-computed state into the snapshot
            # pool for seqs whose chunk ended on a page boundary that
            # ``PrefixSegment`` pre-reserved a snapshot slot for. This is
            # how cross-seq prefix-cache hits later restore the GDN
            # recurrent state into a fresh working slot. ``-1`` slots are a
            # no-op so non-PrefixSegment runs / non-cacheable boundaries
            # don't pay anything beyond a cheap mask check.
            self._maybe_snapshot_state(input_data, conv_state, ssm_state)

        z_shape = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        if core_attn_out.shape != z.shape:
            pad = torch.zeros_like(z)
            pad[: core_attn_out.shape[0], :] = core_attn_out
            core_attn_out = pad

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
        self.attn = FlashAttention(
            kv_layer_id,
            self.scaling,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
        )

    def forward(self, input_data: InputData, hidden_states: torch.Tensor):
        qkv = self.qkv_proj(hidden_states)
        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            orig_shape = q_gate.shape[:-1]
            q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*orig_shape, -1)
            gate = gate.reshape(*orig_shape, -1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            gate = None

        q_shape, k_shape = q.shape, k.shape
        q = self.q_norm(q.reshape(-1, self.head_dim)).view(q_shape)
        k = self.k_norm(k.reshape(-1, self.head_dim)).view(k_shape)
        q, k = self.rotary_emb(input_data.get_position(), q, k)

        attn_out = self.attn.forward(q, k, v, input_data)

        if gate is not None:
            attn_out = attn_out * torch.sigmoid(gate)
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
            hidden_states = self.self_attn(input_data, hidden_states)
        else:
            hidden_states = self.linear_attn(input_data, hidden_states)

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3_5Model(nn.Module):
    """Hybrid GDN + full-attention text decoder.

    The constructor builds two parallel layer indices in step with the
    HF ``layer_types`` schedule:

    * ``self._kv_layer_ids[i] = j``  when layer ``i`` is full-attention and
      this is the j-th full-attention layer (0-indexed). ``j`` is what
      ``FlashAttention`` uses to address ``segment.k_cache[j]`` /
      ``v_cache[j]``.
    * ``self._ssm_layer_ids[i] = j`` when layer ``i`` is linear-attention.
      ``j`` selects ``SSMSegment.conv_state_working[j]`` /
      ``temporal_state_working[j]``.

    ``self.num_kv_layers`` and ``self.ssm_layer_global_ids`` are then read by
    ``model_runner.init`` and surfaced to :class:`MemoryManager` so the KV
    page budget and the SSM pool match the model's real shape.
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
        # boundary — KV cache and SSM pool are also sized per-rank.
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
        # PrefixSegment uses a dense pool so we just need the count.
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
        # SSMSegment.
        self.num_kv_layers = self.model.num_kv_layers
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

    def _build_ssm_cache_config(self, config) -> SSMCacheConfig:
        """Compose the per-rank :class:`SSMCacheConfig`.

        ``num_layers`` is the count of GDN layers on this PP rank;
        ``conv_dim`` is the per-rank packed projection width (``2*K + V``)
        because :class:`Qwen3_5GatedDeltaNet` shards along the head dim;
        ``head_v_dim`` etc. are unchanged by TP. The state dtype is
        ``mamba_ssm_dtype`` (float32 on Qwen3.5-0.8B), keeping recurrent
        accumulation in fp32 even when the rest of the model runs in
        bfloat16.
        """
        tp_size = get_tp_size()
        key_dim = config.linear_num_key_heads * config.linear_key_head_dim
        value_dim = config.linear_num_value_heads * config.linear_value_head_dim
        conv_dim_per_partition = (2 * key_dim + value_dim) // tp_size
        dtype_str = getattr(config, "mamba_ssm_dtype", "float32")
        dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16,
                     "float16": torch.float16}
        ssm_dtype = dtype_map.get(str(dtype_str), torch.float32)
        # Conv state needs to match the activation dtype so the conv1d
        # kernels can ``tl.load`` from it without an implicit cast. We use
        # the current default dtype (set by the engine to the checkpoint
        # dtype before instantiating the model).
        conv_state_dtype = torch.get_default_dtype()
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
        return self.lm_head(hidden_states[idx[1:]])

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    # ----- weight loading --------------------------------------------------

    def load_weights(self, weights, mp_load_progress=None):
        """Walk ``self.named_parameters()`` and pull each from the
        checkpoint.

        We special-case the GDN block because the checkpoint stores it as
        four split projections (``in_proj_qkv``, ``in_proj_z``, ``in_proj_b``,
        ``in_proj_a``) but the runtime fuses them into ``in_proj_qkvz`` and
        ``in_proj_ba``. The dispatch happens at ``layer.linear_attn``
        granularity so we touch a layer's GDN weights once and then continue
        to the next parameter that isn't part of it.
        """
        parameters = dict(self.named_parameters())
        if mp_load_progress is not None:
            mp_load_progress[get_local_rank() * 2] = len(parameters)
            mp_load_progress[get_local_rank() * 2 + 1] = 0
            update = lambda: mp_load_progress.__setitem__(
                get_local_rank() * 2 + 1,
                mp_load_progress[get_local_rank() * 2 + 1] + 1,
            )
        else:
            pbar = get_model_load_pbar(len(parameters))
            update = lambda: pbar.update(1)

        # First pass: handle GDN layers en bloc, recording which destination
        # parameters were filled so we can skip them in the per-param loop.
        gdn_filled: set = set()
        for local_idx, layer in enumerate(self.model.layers):
            if layer.linear_attn is None:
                continue
            global_idx = local_idx + self.start_layer
            prefix = f"model.layers.{global_idx}.linear_attn"
            _load_gdn_layer_weights(layer.linear_attn, prefix, weights)
            for sub in (
                "conv1d.weight",
                "in_proj_qkvz.weight",
                "in_proj_ba.weight",
                "A_log",
                "dt_bias",
                "norm.weight",
                "out_proj.weight",
            ):
                local_key = f"model.layers.{local_idx}.linear_attn.{sub}"
                gdn_filled.add(local_key)
                update()

        # Pass 2: the rest of the parameters.
        attn_ref = None
        for layer in self.model.layers:
            if layer.self_attn is not None:
                attn_ref = layer.self_attn
                break

        num_q_heads = attn_ref.num_heads if attn_ref is not None else 0
        num_kv_heads_local = attn_ref.num_kv_heads if attn_ref is not None else 0
        head_dim_local = attn_ref.head_dim if attn_ref is not None else self.head_dim

        for k, v in parameters.items():
            if k in gdn_filled:
                continue
            k_remote = resolve_pp_layer_idx(k, 2, self.model.start_layer)

            if attn_ref is not None and k.find("self_attn.qkv_proj") != -1:
                head_dim_patch = (
                    head_dim_local
                    if k.find("scale") == -1 or k.find("weight") == -1
                    else 1
                )
                num_q_rows = num_q_heads * (
                    2 if attn_ref.attn_output_gate else 1
                )
                copy_qkv_proj(
                    v.data,
                    get_tensor_from_dict(
                        weights, k_remote.replace("qkv_proj", "q_proj")
                    ),
                    get_tensor_from_dict(
                        weights, k_remote.replace("qkv_proj", "k_proj")
                    ),
                    get_tensor_from_dict(
                        weights, k_remote.replace("qkv_proj", "v_proj")
                    ),
                    num_q_rows,
                    num_kv_heads_local,
                    head_dim_patch,
                )
            elif k.find("self_attn.o_proj") != -1:
                copy_single_proj_dim1(
                    v.data, get_tensor_from_dict(weights, k_remote)
                )
            elif k.find("gate_up_proj") != -1:
                copy_gate_up_proj(
                    v.data,
                    get_tensor_from_dict(
                        weights, k_remote.replace("gate_up_proj", "gate_proj")
                    ),
                    get_tensor_from_dict(
                        weights, k_remote.replace("gate_up_proj", "up_proj")
                    ),
                )
            elif k.find("down_proj") != -1:
                copy_single_proj_dim1(
                    v.data, get_tensor_from_dict(weights, k_remote)
                )
            elif k.find("embed_tokens") != -1 or k.find("lm_head") != -1:
                copy_single_proj_dim0(
                    v.data, get_tensor_from_dict(weights, k_remote)
                )
            elif k.find("self_attn.q_norm") != -1 or k.find("self_attn.k_norm") != -1:
                v.data.copy_(get_tensor_from_dict(weights, k_remote))
            else:
                v.data.copy_(get_tensor_from_dict(weights, k_remote))
            update()


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
        # Expose ssm_cache_config and num_kv_layers at top-level so that
        # ``ModelRunner.init`` (which reads ``self.model``) finds them
        # without having to peek into ``self.language_model``.
        self.ssm_cache_config = self.language_model.ssm_cache_config
        self.num_kv_layers = self.language_model.num_kv_layers
        self.num_ssm_layers = self.language_model.num_ssm_layers

    def load_weights(self, weights, mp_load_progress=None):
        # Language model load is delegated; it walks ``self.language_model``'s
        # named_parameters() and slices each tensor for the current TP rank.
        self.language_model.load_weights(weights, mp_load_progress)

        if not is_first_pp_rank():
            return

        # Visual tower load: same pattern as ``Qwen3VLForConditionalGeneration``.
        parameters = dict(self.visual.named_parameters())
        num_heads = self.visual.num_heads // get_tp_size()
        head_dim = self.visual.hidden_size // self.visual.num_heads
        for k, v in parameters.items():
            if k.find("attn.qkv") != -1:
                src_qkv = get_tensor_from_dict(weights, f"visual.{k}")
                size_partition = src_qkv.shape[0] // 3
                src_q, src_k, src_v = src_qkv.split(
                    [size_partition, size_partition, size_partition], dim=0
                )
                copy_qkv_proj(
                    v.data, src_q, src_k, src_v, num_heads, num_heads, head_dim
                )
            elif (
                k.find("attn.proj.weight") != -1
                or k.find("linear_fc2.weight") != -1
            ):
                # Row-parallel projections: shard along dim 1 (input dim).
                # Match the suffix ``.weight`` explicitly so biases (1-D)
                # take the default fall-through copy. Mirrors qwen3_vl.
                copy_single_proj_dim1(
                    v.data, get_tensor_from_dict(weights, f"visual.{k}")
                )
            elif k.find("linear_fc1") != -1:
                copy_single_proj_dim0(
                    v.data, get_tensor_from_dict(weights, f"visual.{k}")
                )
            else:
                v.data.copy_(get_tensor_from_dict(weights, f"visual.{k}"))
