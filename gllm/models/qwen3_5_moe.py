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
    get_tp_size,
    is_first_pp_rank,
    is_last_pp_rank,
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
from gllm.models.weight_loader import (
    LoadContext,
    WeightRule,
    contains,
    h_gate_up,
    h_proj_dim0,
    h_proj_dim1,
    h_qkv_proj_gqa,
    h_w13_hybrid,
    h_w2_hybrid,
    hv_proj_dim0,
    hv_proj_dim1,
    hv_qkv_fused_split,
    make_gdn_pre_pass,
    run_vision_loader,
    run_weight_loader,
)


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

    def _make_load_context(self, weights):
        ctx = super()._make_load_context(weights)
        attn = ctx.extra.get("_attn_ref")
        assert attn is not None, "Qwen3.5-MoE expects >=1 full-attn layer"
        tp_size = get_tp_size()
        if attn.total_num_kv_heads >= tp_size:
            num_kv_head_replicas = 1
        else:
            num_kv_head_replicas = tp_size // attn.total_num_kv_heads
        qkv_block_n = (
            attn.qkv_proj.weight_block_size[0]
            if hasattr(attn.qkv_proj, "weight_block_size")
            else None
        )
        ctx.extra["num_kv_head_replicas"] = num_kv_head_replicas
        ctx.extra["qkv_block_n"] = qkv_block_n

        num_experts = getattr(self.config, "num_experts", 0)
        if num_experts <= 0:
            raise ValueError(
                "Qwen3_5MoeLLMForCausalLM expects ``num_experts > 0``; got "
                f"{num_experts}. Use Qwen3_5ForCausalLM for the dense variant."
            )
        ctx.num_experts = num_experts
        _, ctx.expert_map = determine_expert_map(
            get_ep_size(), get_ep_rank(), num_experts
        )
        return ctx

    def weight_rules(self):
        # GQA/FP8-aware qkv + hybrid (stacked-or-per-expert, FP8-aware) experts,
        # then the dense rules for o_proj / shared-expert gate_up / down /
        # embed. Router (``mlp.gate.weight``), ``shared_expert_gate`` and
        # q_norm/k_norm fall through to the default verbatim copy.
        return [
            WeightRule(contains("self_attn.qkv_proj"), h_qkv_proj_gqa, "qkv_proj"),
            WeightRule(contains("mlp.experts.w13_weight"), h_w13_hybrid, "w13_expert"),
            WeightRule(contains("mlp.experts.w2_weight"), h_w2_hybrid, "w2_expert"),
            WeightRule(contains("self_attn.o_proj"), h_proj_dim1, "o_proj"),
            WeightRule(contains("gate_up_proj"), h_gate_up, "gate_up_proj"),
            WeightRule(contains("down_proj"), h_proj_dim1, "down_proj"),
            WeightRule(
                contains("embed_tokens", "lm_head"), h_proj_dim0, "embed_lm_head"
            ),
        ]

    def load_weights(self, weights, mp_load_progress=None):
        run_weight_loader(
            self,
            weights,
            self.weight_rules(),
            mp_load_progress,
            pp_idx_offset=2,
            start_layer=self.start_layer,
            ctx=self._make_load_context(weights),
            pre_passes=[make_gdn_pre_pass(self.GDN_SUBS, _load_gdn_layer_weights)],
        )



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
        # Encoder-disaggregation encoder process: no language model at all.
        if getattr(self, "skip_language", False) or self.language_model is None:
            self.ssm_cache_config = None
            self.num_kv_layers = 0
            self.num_ssm_layers = 0
            return
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
        if not getattr(self, "skip_language", False) and self.language_model is not None:
            self.language_model.load_weights(weights, mp_load_progress)

        if not is_first_pp_rank():
            return

        # Encoder-disaggregation LM node skips the vision tower entirely.
        if getattr(self, "skip_visual", False) or self.visual is None:
            return

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
