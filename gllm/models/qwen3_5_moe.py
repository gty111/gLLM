"""Qwen3.5-MoE: hybrid (linear-attention + full-attention) decoder with
sparse-MoE MLPs and FP8 block-quantized linear projections.

This module reuses the GDN / full-attention layer classes from
``qwen3_5.py`` verbatim and only swaps the per-layer dense MLP for a
``Qwen2MoeSparseMoeBlock`` (with the optional shared expert + shared-
expert gate that the Qwen3.5-MoE checkpoint always ships).

The VL wrapper (``Qwen3_5MoeForConditionalGeneration``) keeps the same
shape as ``Qwen3_5ForConditionalGeneration``: a thin subclass of
``Qwen3VLForConditionalGeneration`` that plugs in this language model.
Vision tower load is intentionally a best-effort copy of the existing
Qwen3.5 dense VL path. Checkpoint ``mtp.*`` weights are not loaded.

FP8 block-quant scope:

* Full-attention ``qkv_proj``/``o_proj`` are FP8 (the per-rank component
  sizes 2048/512 and 1024 are all multiples of ``block_n=128``).
* GDN ``in_proj_qkvz`` (qkv+z fused) and ``out_proj`` are FP8; the merged
  parameter shape stays block-aligned because every component is
  ``key_dim/tp_size`` or ``value_dim/tp_size`` rows.
* GDN ``in_proj_ba`` and ``conv1d`` stay in bf16 (sub-block size — the
  Qwen3.5-MoE checkpoint flags them in ``modules_to_not_convert``).
* MoE expert ``w13``/``w2`` are FP8 via ``Fp8MoEMethod``; the shared expert
  and ``lm_head`` follow the per-config quant scheme (FP8 for the shared
  expert, bf16 for ``lm_head``/``embed_tokens`` per the checkpoint).
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch import nn

from gllm.dist_utils import (
    get_ep_rank,
    get_ep_size,
    get_local_rank,
    get_tp_size,
    is_first_pp_rank,
    is_last_pp_rank,
    resolve_pp_layer_idx,
)
from gllm.input_data import InputData
from gllm.layers.moe import determine_expert_map
from gllm.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5GatedDeltaNet,
    Qwen3_5Model,
    _load_gdn_layer_weights,
)
from gllm.models.qwen3_vl import Qwen3VLForConditionalGeneration
from gllm.models.weight_utils import (
    copy_gate_up_proj,
    copy_qkv_proj_gqa,
    copy_single_proj_dim0,
    copy_single_proj_dim1,
    get_tensor_from_dict,
    load_fused_w13_per_expert,
    load_w2_per_expert,
    moe_expert_load_pool,
)
from gllm.utils import get_model_load_pbar


class Qwen3_5MoeLLMModel(Qwen3_5Model):
    """Drop-in subclass that simply forwards through the base model.

    Exists so ``Qwen3_5MoeLLMForCausalLM`` can plug a custom ``Model`` into
    the existing ``Qwen3_5ForCausalLM`` infrastructure without leaking
    "is this MoE?" awareness into ``Qwen3_5Model.__init__``. The MoE/dense
    dispatch happens inside ``Qwen3_5DecoderLayer`` via
    ``_is_moe_text_config``.
    """

    def __init__(self, config):
        super().__init__(config)


class Qwen3_5MoeLLMForCausalLM(Qwen3_5ForCausalLM):
    """Text-only Qwen3.5-MoE causal LM.

    Inherits the GDN / full-attention plumbing from the dense parent and
    only overrides :meth:`load_weights` to additionally handle the per-
    expert MoE tensors, the shared expert + gate, the router, and the FP8
    block-quant ``weight_scale_inv`` for every quantized linear (including
    the GQA-replicated qkv_proj where ``TP > total_num_kv_heads``).
    """

    def __init__(self, config):
        super().__init__(config, model_type=Qwen3_5MoeLLMModel)

    def load_weights(self, weights, mp_load_progress=None):
        """Load weights into this rank's parameters.

        Three logical groups:

        1. **GDN layers** are fused at load time
           (``_load_gdn_layer_weights``); we record every parameter the
           helper fills so we can skip them in the per-parameter loop.
        2. **MoE experts** are loaded per-expert with a thread pool because
           every checkpoint key is a different mmap'd tensor and the
           per-tensor H2D latency dominates without parallelism. Both
           ``w*_weight`` and (when FP8) ``w*_weight_scale_inv`` follow the
           same per-expert key formatting.
        3. **Everything else** runs in the standard per-parameter loop;
           the ``qkv_proj`` path uses :func:`copy_qkv_proj_gqa` because the
           Qwen3.5-MoE TP=4 config replicates the 2 kv heads across pairs
           of TP ranks, and the FP8 scale needs ``head_dim // block_n``
           rather than 1 row per head when ``head_dim > block_n``.
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

        # ---- Pass 1: GDN fusion -------------------------------------------
        gdn_filled: set = set()
        gdn_subs = [
            "conv1d.weight",
            "in_proj_qkvz.weight",
            "in_proj_qkvz.weight_scale_inv",
            "in_proj_ba.weight",
            "A_log",
            "dt_bias",
            "norm.weight",
            "out_proj.weight",
            "out_proj.weight_scale_inv",
        ]
        for local_idx, layer in enumerate(self.model.layers):
            if layer.linear_attn is None:
                continue
            global_idx = local_idx + self.start_layer
            prefix = f"model.layers.{global_idx}.linear_attn"
            _load_gdn_layer_weights(layer.linear_attn, prefix, weights)
            for sub in gdn_subs:
                local_key = f"model.layers.{local_idx}.linear_attn.{sub}"
                if local_key in parameters:
                    gdn_filled.add(local_key)
                    update()

        # ---- attention bookkeeping ----------------------------------------
        attn_ref = None
        for layer in self.model.layers:
            if layer.self_attn is not None:
                attn_ref = layer.self_attn
                break
        assert attn_ref is not None, "Qwen3.5-MoE expects ≥1 full-attn layer"

        num_q_heads_per_rank = attn_ref.num_heads
        num_kv_heads_per_rank = attn_ref.num_kv_heads
        head_dim_local = attn_ref.head_dim
        # ``total_num_kv_heads / tp_size`` rounded toward zero; ``QKVParallel``
        # sets ``num_kv_head_replicas = tp / total_kv`` whenever total_kv < tp.
        tp_size = get_tp_size()
        if attn_ref.total_num_kv_heads >= tp_size:
            num_kv_head_replicas = 1
        else:
            num_kv_head_replicas = tp_size // attn_ref.total_num_kv_heads

        # FP8 block_n for the qkv_proj's weight_scale_inv slicing. Falls back
        # to ``head_dim_local`` when the linear isn't FP8 (so the scale path
        # is never taken).
        qkv_block_n = (
            attn_ref.qkv_proj.weight_block_size[0]
            if hasattr(attn_ref.qkv_proj, "weight_block_size")
            else None
        )

        num_q_rows_per_rank = num_q_heads_per_rank * (
            2 if attn_ref.attn_output_gate else 1
        )

        # ---- MoE expert metadata ------------------------------------------
        num_experts = getattr(self.config, "num_experts", 0)
        if num_experts <= 0:
            raise ValueError(
                "Qwen3_5MoeLLMForCausalLM expects ``num_experts > 0``; got "
                f"{num_experts}. Use Qwen3_5ForCausalLM for the dense variant."
            )
        _, expert_map = determine_expert_map(
            get_ep_size(), get_ep_rank(), num_experts
        )

        with moe_expert_load_pool(num_experts) as pool:
            for k, v in parameters.items():
                if k in gdn_filled:
                    continue
                k_remote = resolve_pp_layer_idx(k, 2, self.model.start_layer)

                if k.find("self_attn.qkv_proj") != -1:
                    is_scale = k.endswith("weight_scale_inv")
                    if is_scale:
                        # FP8 block-quant scale: one row per ``block_n``
                        # output rows of the weight. ``head_dim_or_blocks``
                        # = ``head_dim // block_n`` so the per-component
                        # offsets in copy_qkv_proj_gqa land on the right
                        # scale rows. With head_dim=256 and block_n=128
                        # this is 2, not 1 (which would be the head_dim=128
                        # case the original Qwen2 loader was written for).
                        head_dim_or_blocks = head_dim_local // qkv_block_n
                    else:
                        head_dim_or_blocks = head_dim_local
                    copy_qkv_proj_gqa(
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
                        num_q_rows_per_rank,
                        num_kv_heads_per_rank,
                        num_kv_head_replicas,
                        head_dim_or_blocks,
                    )
                elif k.find("self_attn.o_proj") != -1:
                    copy_single_proj_dim1(
                        v.data, get_tensor_from_dict(weights, k_remote)
                    )
                elif k.find("mlp.experts.w13_weight") != -1:
                    # FusedMoE param shape (local_E, 2*I_p, H) or
                    # (local_E, 2*I_p_blocks, H_blocks) for the FP8 scale.
                    # The per-expert checkpoint keys differ only in the
                    # final suffix (``.weight`` vs ``.weight_scale_inv``).
                    is_scale = k.endswith("weight_scale_inv")
                    suffix = "weight_scale_inv" if is_scale else "weight"
                    base = k_remote.replace(
                        "w13_weight_scale_inv" if is_scale else "w13_weight",
                        "",
                    )
                    load_fused_w13_per_expert(
                        v.data,
                        weights,
                        key_for_gate=lambda i, base=base, s=suffix: (
                            f"{base}{i}.gate_proj.{s}"
                        ),
                        key_for_up=lambda i, base=base, s=suffix: (
                            f"{base}{i}.up_proj.{s}"
                        ),
                        expert_map=expert_map,
                        num_experts=num_experts,
                        pool=pool,
                    )
                elif k.find("mlp.experts.w2_weight") != -1:
                    is_scale = k.endswith("weight_scale_inv")
                    suffix = "weight_scale_inv" if is_scale else "weight"
                    base = k_remote.replace(
                        "w2_weight_scale_inv" if is_scale else "w2_weight",
                        "",
                    )
                    load_w2_per_expert(
                        v.data,
                        weights,
                        key_for_down=lambda i, base=base, s=suffix: (
                            f"{base}{i}.down_proj.{s}"
                        ),
                        expert_map=expert_map,
                        num_experts=num_experts,
                        pool=pool,
                    )
                elif k.find("gate_up_proj") != -1:
                    # Shared expert (the only ``gate_up_proj`` path left
                    # outside the FusedMoE experts) or any other dense
                    # gate_up_proj if a layer falls into ``mlp_only_layers``.
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
                    # Shared expert down_proj or dense down_proj (RowParallel).
                    copy_single_proj_dim1(
                        v.data, get_tensor_from_dict(weights, k_remote)
                    )
                elif k.find("mlp.gate.weight") != -1:
                    # Router: ``nn.Linear(hidden, num_experts)`` with no TP.
                    v.data.copy_(get_tensor_from_dict(weights, k_remote))
                elif k.find("shared_expert_gate") != -1:
                    # 1-D scalar gate per token (``Linear(hidden, 1)``).
                    v.data.copy_(get_tensor_from_dict(weights, k_remote))
                elif k.find("embed_tokens") != -1 or k.find("lm_head") != -1:
                    copy_single_proj_dim0(
                        v.data, get_tensor_from_dict(weights, k_remote)
                    )
                elif (
                    k.find("self_attn.q_norm") != -1
                    or k.find("self_attn.k_norm") != -1
                ):
                    v.data.copy_(get_tensor_from_dict(weights, k_remote))
                else:
                    v.data.copy_(get_tensor_from_dict(weights, k_remote))
                update()


# ---------------------------------------------------------------------------
# Qwen3_5MoeForConditionalGeneration (VL wrapper)
# ---------------------------------------------------------------------------


class Qwen3_5MoeForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """Qwen3.5-MoE-VL: Qwen3-VL vision tower + Qwen3.5-MoE hybrid text LM.

    The class only needs to forward the right ``language_model_type`` into
    :class:`Qwen3VLForConditionalGeneration` and re-expose the SSM cache
    bookkeeping so that ``ModelRunner.init`` finds it on ``self.model``
    (it never inspects ``self.language_model``).
    """

    def __init__(self, config):
        # NB: ``quantization_config`` is already mirrored from the top-level
        # config onto ``config.text_config`` by
        # ``ModelLoader.load_config`` (see
        # :func:`gllm.model_loader.propagate_quantization_config`). That
        # mirroring is essential for FP8 checkpoints like Qwen3.5-MoE-FP8,
        # which only carry the quant config at the top level — without it
        # the language sub-model would silently fall back to bf16.
        super().__init__(config, language_model_type=Qwen3_5MoeLLMForCausalLM)
        self.ssm_cache_config = self.language_model.ssm_cache_config
        self.num_kv_layers = self.language_model.num_kv_layers
        self.num_ssm_layers = self.language_model.num_ssm_layers

    def load_weights(self, weights, mp_load_progress=None):
        """Load language model weights; vision tower load is a best-effort
        bf16 path that mirrors :class:`Qwen3_5ForConditionalGeneration`.

        ``mtp.*`` checkpoint keys are skipped (MTP speculative decoding is
        not implemented); only ``model.*``, ``lm_head``, and ``visual.*``
        tensors are loaded.
        """
        self.language_model.load_weights(weights, mp_load_progress)

        if not is_first_pp_rank():
            return

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
                # Vision tower stays in bf16 so the dense per-head loader
                # is the right tool (no FP8 scale tensors to copy here).
                from gllm.models.weight_utils import copy_qkv_proj
                copy_qkv_proj(
                    v.data, src_q, src_k, src_v, num_heads, num_heads, head_dim
                )
            elif (
                k.find("attn.proj.weight") != -1
                or k.find("linear_fc2.weight") != -1
            ):
                copy_single_proj_dim1(
                    v.data, get_tensor_from_dict(weights, f"visual.{k}")
                )
            elif k.find("linear_fc1") != -1:
                copy_single_proj_dim0(
                    v.data, get_tensor_from_dict(weights, f"visual.{k}")
                )
            else:
                v.data.copy_(get_tensor_from_dict(weights, f"visual.{k}"))
