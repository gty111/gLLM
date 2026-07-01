from typing import Optional

import torch
from sgl_kernel.scalar_type import scalar_types

from gllm.dist_utils import (
    get_ep_rank,
    get_ep_size,
    get_tp_size,
    is_use_ep,
    tensor_model_parallel_all_reduce,
)
from gllm import _custom_ops as ops
from gllm.layers.moe.moe_align_block_size import moe_align_block_size
from gllm.layers.moe.fused_moe_triton.fused_moe import fused_experts
from gllm.layers.moe.topk import select_experts
from gllm.utils import set_weight_attrs


class FusedMoEMethod(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=top_k,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
        )

        return fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
        )


class Fp8MoEMethod(FusedMoEMethod):
    def __init__(self, quant_config):
        super().__init__()
        self.quant_config = quant_config
        self.weight_block_size = self.quant_config["weight_block_size"]

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        super().create_weights(
            layer=layer,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=torch.float8_e4m3fn,
            extra_weight_attrs=extra_weight_attrs,
        )

        block_n, block_k = (
            self.weight_block_size[0],
            self.weight_block_size[1],
        )

        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                2 * ((intermediate_size_per_partition + block_n - 1) // block_n),
                (hidden_size + block_k - 1) // block_k,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                (hidden_size + block_n - 1) // block_n,
                (intermediate_size_per_partition + block_k - 1) // block_k,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
        layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
        assert self.quant_config["activation_scheme"] == "dynamic"

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=top_k,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
        )

        return fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            use_fp8_w8a8=True,
            w1_scale=layer.w13_weight_scale_inv,
            w2_scale=layer.w2_weight_scale_inv,
            block_shape=self.weight_block_size,
        )


def _get_scale_perms():
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def _marlin_permute_scales(
    s: torch.Tensor,
    size_k: int,
    size_n: int,
    group_size: int,
) -> torch.Tensor:
    scale_perm, scale_perm_single = _get_scale_perms()
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    s = s.reshape((-1, size_n)).contiguous()
    return s


class Int4MarlinMoEMethod(FusedMoEMethod):
    """INT4 routed-expert MoE via sgl-kernel marlin fused gemm.

    Expected checkpoint format per expert:
    - gate/up/down ``weight_packed``: int32 packed (8x int4 per int32)
    - gate/up/down ``weight_scale``: bf16 group scales (group_size=32)
    """

    def __init__(self, quant_config):
        super().__init__()
        self.quant_config = quant_config
        self.num_bits = int(self.quant_config.get("num_bits", 4))
        if self.num_bits != 4:
            raise ValueError(f"int4_moe only supports 4 bits, got {self.num_bits}")
        self.pack_factor = 32 // self.num_bits
        self.group_size = int(self.quant_config.get("group_size", 32))
        self.quant_type_id = scalar_types.uint4b8.id

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Pre-repack layout (matches checkpoint names):
        # w13: [E, 2I, H/8], w2: [E, H, I/8]
        if hidden_size % self.pack_factor != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by pack_factor={self.pack_factor}"
            )
        if intermediate_size_per_partition % self.pack_factor != 0:
            raise ValueError(
                "intermediate_size_per_partition must be divisible by "
                f"pack_factor={self.pack_factor}, got {intermediate_size_per_partition}"
            )
        if hidden_size % self.group_size != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by group_size={self.group_size}"
            )
        if intermediate_size_per_partition % self.group_size != 0:
            raise ValueError(
                "intermediate_size_per_partition must be divisible by "
                f"group_size={self.group_size}, got {intermediate_size_per_partition}"
            )

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.group_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.group_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        layer.register_buffer(
            "workspace",
            torch.zeros(1, dtype=torch.int32),
            persistent=False,
        )
        layer.register_buffer(
            "w13_g_idx",
            torch.empty((num_experts, 0), dtype=torch.int32),
            persistent=False,
        )
        layer.register_buffer(
            "w2_g_idx",
            torch.empty((num_experts, 0), dtype=torch.int32),
            persistent=False,
        )
        layer._int4_marlin_ready = False

    @staticmethod
    def _pick_block_size_m(num_tokens: int, num_experts: int, topk: int) -> int:
        block_size_m = 64
        for cand in (8, 16, 32, 48, 64):
            if num_tokens * topk / num_experts / cand < 0.9:
                block_size_m = cand
                break
        return block_size_m

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_int4_marlin_ready", False):
            return

        num_experts = layer.w13_weight_packed.shape[0]
        device = layer.w13_weight_packed.device
        perm = torch.empty(0, dtype=torch.int, device=device)

        marlin_w13 = []
        marlin_w2 = []
        marlin_s13 = []
        marlin_s2 = []

        for e in range(num_experts):
            w13_e = layer.w13_weight_packed[e]
            two_i = w13_e.shape[0]
            i = two_i // 2
            h = w13_e.shape[1] * self.pack_factor

            gate_q = w13_e[:i].T.contiguous()
            up_q = w13_e[i:].T.contiguous()
            gate_rep = ops.gptq_marlin_repack(
                b_q_weight=gate_q,
                perm=perm,
                size_k=h,
                size_n=i,
                num_bits=self.num_bits,
            )
            up_rep = ops.gptq_marlin_repack(
                b_q_weight=up_q,
                perm=perm,
                size_k=h,
                size_n=i,
                num_bits=self.num_bits,
            )
            marlin_w13.append(torch.cat([gate_rep, up_rep], dim=1).contiguous())

            s13_e = layer.w13_weight_scale[e].T.contiguous()
            marlin_s13.append(
                _marlin_permute_scales(s13_e, size_k=h, size_n=two_i, group_size=self.group_size)
            )

            w2_e = layer.w2_weight_packed[e]
            h2 = w2_e.shape[0]
            i2 = w2_e.shape[1] * self.pack_factor
            w2_q = w2_e.T.contiguous()
            w2_rep = ops.gptq_marlin_repack(
                b_q_weight=w2_q,
                perm=perm,
                size_k=i2,
                size_n=h2,
                num_bits=self.num_bits,
            )
            marlin_w2.append(w2_rep)

            s2_e = layer.w2_weight_scale[e].T.contiguous()
            marlin_s2.append(
                _marlin_permute_scales(s2_e, size_k=i2, size_n=h2, group_size=self.group_size)
            )

        layer.w13_weight_packed.data = torch.stack(marlin_w13, dim=0).contiguous()
        layer.w2_weight_packed.data = torch.stack(marlin_w2, dim=0).contiguous()
        layer.w13_weight_scale.data = torch.stack(marlin_s13, dim=0).contiguous()
        layer.w2_weight_scale.data = torch.stack(marlin_s2, dim=0).contiguous()

        sms = torch.cuda.get_device_properties(device).multi_processor_count
        layer.workspace = torch.zeros(sms * 4, dtype=torch.int32, device=device)
        layer._int4_marlin_ready = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if activation != "silu":
            raise ValueError(f"int4_moe only supports silu activation, got {activation}")

        self.process_weights_after_loading(layer)

        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=top_k,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
        )

        m, hidden = x.shape
        if global_num_experts == -1:
            global_num_experts = layer.w13_weight_packed.shape[0]
        block_size_m = self._pick_block_size_m(
            m,
            max(global_num_experts, 1),
            top_k,
        )

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids,
            block_size_m,
            global_num_experts,
            expert_map,
        )

        inter_size = layer.w2_weight_packed.shape[1] * 16
        gate_up = torch.empty(
            (m * top_k, 2 * inter_size),
            dtype=x.dtype,
            device=x.device,
        )

        gate_up = ops.moe_wna16_marlin_gemm(
            a=x,
            c_or_none=gate_up,
            b_q_weight=layer.w13_weight_packed,
            b_bias_or_none=None,
            b_scales=layer.w13_weight_scale,
            global_scale_or_none=None,
            b_zeros_or_none=None,
            g_idx_or_none=layer.w13_g_idx,
            perm_or_none=None,
            workspace=layer.workspace,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            topk_weights=topk_weights,
            moe_block_size=block_size_m,
            top_k=top_k,
            mul_topk_weights=apply_router_weight_on_input,
            is_ep=expert_map is not None,
            b_q_type_id=self.quant_type_id,
            size_m=m,
            size_n=2 * inter_size,
            size_k=hidden,
            is_k_full=True,
            use_atomic_add=False,
            use_fp32_reduce=True,
            is_zp_float=False,
        )

        act = torch.empty((m * top_k, inter_size), dtype=x.dtype, device=x.device)
        ops.silu_and_mul(act, gate_up.view(-1, 2 * inter_size))

        # Under EP each rank only owns a slice of experts, so the Marlin GEMM
        # writes output rows only for token-blocks routed to local experts and
        # leaves the rest untouched. ``moe_sum`` below then sums ``top_k`` rows
        # per token, so any non-local row must be a hard zero rather than the
        # uninitialized contents of ``torch.empty``. Mirror SGLang's reference
        # (``intermediate_cache3.zero_()`` when ``expert_map is not None``).
        if expert_map is not None:
            down = torch.zeros((m * top_k, hidden), dtype=x.dtype, device=x.device)
        else:
            down = torch.empty((m * top_k, hidden), dtype=x.dtype, device=x.device)
        down = ops.moe_wna16_marlin_gemm(
            a=act,
            c_or_none=down,
            b_q_weight=layer.w2_weight_packed,
            b_bias_or_none=None,
            b_scales=layer.w2_weight_scale,
            global_scale_or_none=None,
            b_zeros_or_none=None,
            g_idx_or_none=layer.w2_g_idx,
            perm_or_none=None,
            workspace=layer.workspace,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            topk_weights=topk_weights,
            moe_block_size=block_size_m,
            top_k=1,
            mul_topk_weights=not apply_router_weight_on_input,
            is_ep=expert_map is not None,
            b_q_type_id=self.quant_type_id,
            size_m=m * top_k,
            size_n=hidden,
            size_k=inter_size,
            is_k_full=True,
            use_atomic_add=False,
            use_fp32_reduce=True,
            is_zp_float=False,
        )

        out = torch.empty_like(x)
        ops.moe_sum(down.view(m, top_k, hidden), out)
        return out


class FusedMoE(torch.nn.Module):
    """FusedMoE layer for MoE models.

    This layer contains both MergedColumnParallel weights (gate_up_proj /
    w13) and RowParallelLinear weights (down_proj/ w2).

    Note: Mixtral uses w1, w2, and w3 for gate, up, and down_proj. We
    copy that naming convention here and handle any remapping in the
    load_weights function in each model implementation.

    Args:
        num_experts: Number of experts in the model
        top_k: Number of experts selected for each token
        hidden_size: Input hidden state size of the transformer
        intermediate_size: Intermediate size of the experts
        params_dtype: Data type for the parameters.
        reduce_results: Whether to all all_reduce on the output of the layer
        renomalize: Whether to renormalize the logits in the fused_moe kernel
        quant_config: Quantization configure.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        renormalize: bool = True,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        quant_config=None,
    ):
        super().__init__()

        self.quant_config = quant_config

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        self.tp_size = get_tp_size()
        self.ep_size = get_ep_size()
        self.ep_rank = get_ep_rank()

        self.global_num_experts = num_experts

        if is_use_ep():
            self.local_num_experts, self.expert_map = determine_expert_map(
                ep_size=self.ep_size,
                ep_rank=self.ep_rank,
                global_num_experts=self.global_num_experts,
            )
            self.intermediate_size_per_partition = intermediate_size
        else:
            self.local_num_experts, self.expert_map = (self.global_num_experts, None)
            self._validate_tp_size(intermediate_size)
            self.intermediate_size_per_partition = intermediate_size // self.tp_size

        self.top_k = top_k

        self.reduce_results = reduce_results
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.scoring_func = scoring_func
        self.e_score_correction_bias = e_score_correction_bias
        self.activation = activation
        self.apply_router_weight_on_input = apply_router_weight_on_input

        if self.scoring_func != "softmax" and not self.use_grouped_topk:
            raise ValueError(
                "Only softmax scoring function is supported for " "non-grouped topk."
            )

        self.quant_method = self.dispatch_quant_method()

        self.quant_method.create_weights(
            layer=self,
            num_experts=self.local_num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=self.intermediate_size_per_partition,
            params_dtype=params_dtype,
        )

    def _validate_tp_size(self, intermediate_size: int) -> None:
        """Fail fast (non-EP) when the TP size can't shard the experts.

        Two constraints:
          * ``intermediate_size`` must be divisible by ``tp_size``.
          * For block-scale FP8, the per-rank intermediate size must also be a
            multiple of the quant block size ``weight_block_size[0]``; otherwise
            the fused gate/up weight rows and their block scales disagree
            (``cdiv(2*inter_pp, block_n) != 2*cdiv(inter_pp, block_n)``) and the
            Triton MoE kernel aborts with a bare ``AssertionError`` deep in the
            launch. Surfacing the valid TP sizes here is far more actionable.
        """
        block_n = None
        if (
            self.quant_config is not None
            and self.quant_config.get("quant_method") == "fp8"
            and self.quant_config.get("weight_block_size")
        ):
            block_n = self.quant_config["weight_block_size"][0]

        def _tp_ok(tp: int) -> bool:
            if intermediate_size % tp != 0:
                return False
            if block_n is not None and (intermediate_size // tp) % block_n != 0:
                return False
            return True

        if _tp_ok(self.tp_size):
            return

        valid = [tp for tp in range(1, intermediate_size + 1) if _tp_ok(tp)]
        if block_n is not None:
            detail = (
                f"block-scale FP8 (weight_block_size[0]={block_n}) requires "
                f"intermediate_size ({intermediate_size}) to be divisible by "
                f"tp_size and the per-rank shard "
                f"(intermediate_size // tp_size) to be a multiple of {block_n}"
            )
        else:
            detail = (
                f"intermediate_size ({intermediate_size}) must be divisible by "
                f"tp_size"
            )
        raise ValueError(
            f"--tp {self.tp_size} is not compatible with this MoE: {detail}. "
            f"Supported tp sizes for this model: {valid}."
        )

    def dispatch_quant_method(self):
        if self.quant_config is None:
            return FusedMoEMethod()
        elif self.quant_config["quant_method"] == "fp8":
            assert "weight_block_size" in self.quant_config
            return Fp8MoEMethod(self.quant_config)
        elif self.quant_config["quant_method"] == "int4_moe":
            return Int4MarlinMoEMethod(self.quant_config)
        else:
            raise Exception(
                f"gLLM do not support quant_method {self.quant_config['quant_method']}"
            )

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        # Matrix multiply.
        final_hidden_states = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            renormalize=self.renormalize,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
            global_num_experts=self.global_num_experts,
            expert_map=self.expert_map,
            use_grouped_topk=self.use_grouped_topk,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
        )

        if self.reduce_results and self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states


def determine_expert_map(
    ep_size: int, ep_rank: int, global_num_experts: int
) -> tuple[int, Optional[torch.Tensor]]:
    """
    Calculates how many experts should be assigned to each rank for EP and
    creates a mapping from global to local expert index. Experts are
    distributed evenly across ranks. Any remaining are assigned to the
    last rank.

    Args:
        ep_size (int): The size of the expert parallel group
        global_num_experts (int): The total number of experts in the model.

    Returns:
        tuple[int, Optional[torch.Tensor]]: A tuple containing:
            - local_num_experts (int): The number of experts assigned
                to the current rank.
            - expert_map (Optional[torch.Tensor]): A tensor of shape
                (global_num_experts,) mapping from global to local index.
                Contains -1 for experts not assigned to the current rank.
                Returns None if ep_size is 1.
    """
    assert ep_size > 0
    if ep_size == 1:
        return (global_num_experts, None)

    local_num_experts = global_num_experts // ep_size

    # Create a tensor of size num_experts filled with -1
    expert_map = torch.full((global_num_experts,), -1, dtype=torch.int32)
    # Create a expert map for the local experts
    if ep_rank < (ep_size - 1):
        # Each non-last rank gets local_num_experts experts.
        expert_map[ep_rank * local_num_experts : (ep_rank + 1) * local_num_experts] = (
            torch.arange(0, local_num_experts, dtype=torch.int32)
        )
    else:
        # All remaining experts are assigned to the last rank.
        local_num_experts = global_num_experts - ep_rank * local_num_experts

        expert_map[-local_num_experts:] = torch.arange(
            0, local_num_experts, dtype=torch.int32
        )
    return (local_num_experts, expert_map)
