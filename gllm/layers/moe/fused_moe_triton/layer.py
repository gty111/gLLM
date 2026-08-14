from typing import Optional

import torch

from gllm.distributed.parallel_state import (
    get_ep_rank,
    get_ep_size,
    get_tp_size,
    is_use_ep,
    tensor_model_parallel_all_reduce,
)
from gllm import _custom_ops as ops
from gllm.layers.moe.fused_moe_triton.fused_moe import (
    fused_experts,
    get_config_dtype_str,
    make_fused_moe_workspace,
    use_fused_moe_workspace,
)
from gllm.layers.moe.topk import select_experts
from gllm.utils import set_weight_attrs


class FusedMoEMethod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._piecewise_workspaces = {}

    def _piecewise_workspace(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        top_k: int,
        global_num_experts: int,
        *,
        use_fp8_w8a8: bool = False,
        use_int8_w8a16: bool = False,
        use_int4_w4a16: bool = False,
        block_shape=None,
    ):
        # Import lazily so the generic MoE layer does not depend on the model
        # runner unless piecewise capture is actually active.
        from gllm.runtime.piecewise_cuda_graph import PiecewiseRuntime

        runtime = PiecewiseRuntime.current()
        if runtime is None:
            return None
        workspace_tokens = runtime.workspace_tokens
        key = (
            workspace_tokens,
            x.shape[-1],
            x.dtype,
            x.device,
            use_fp8_w8a8,
            use_int8_w8a16,
            use_int4_w4a16,
            tuple(block_shape or ()),
            runtime.workspace_token_sizes,
        )
        workspace = self._piecewise_workspaces.get(key)
        if workspace is None:
            workspace_input = x
            if x.shape[0] != workspace_tokens:
                workspace_input = x.new_empty(
                    (workspace_tokens, x.shape[-1])
                )
            config_dtype = get_config_dtype_str(
                dtype=x.dtype,
                use_fp8_w8a8=use_fp8_w8a8,
                use_int8_w8a16=use_int8_w8a16,
                use_int4_w4a16=use_int4_w4a16,
            )
            workspace = make_fused_moe_workspace(
                workspace_input,
                layer.w13_weight,
                layer.w2_weight,
                top_k,
                global_num_experts=global_num_experts,
                config_dtype=config_dtype,
                block_shape=block_shape,
                token_counts=runtime.workspace_token_sizes,
            )
            self._piecewise_workspaces[key] = workspace
        return workspace

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
                device="cuda",
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
                device="cuda",
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

        workspace = self._piecewise_workspace(
            layer, x, top_k, global_num_experts
        )
        with use_fused_moe_workspace(workspace):
            return fused_experts(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=layer.inplace,
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
        # ``scale_fmt="ue8m0"`` (DeepSeek-V3.2) rounds FP8 activation group
        # scales to powers of two, matching the reference numerics.
        self.use_ue8m0 = self.quant_config.get("scale_fmt") == "ue8m0"

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
                device="cuda",
            ),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                (hidden_size + block_n - 1) // block_n,
                (intermediate_size_per_partition + block_k - 1) // block_k,
                dtype=torch.float32,
                device="cuda",
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

        workspace = self._piecewise_workspace(
            layer,
            x,
            top_k,
            global_num_experts,
            use_fp8_w8a8=True,
            block_shape=self.weight_block_size,
        )
        with use_fused_moe_workspace(workspace):
            return fused_experts(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=layer.inplace,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                use_fp8_w8a8=True,
                w1_scale=layer.w13_weight_scale_inv,
                w2_scale=layer.w2_weight_scale_inv,
                block_shape=self.weight_block_size,
                use_ue8m0=self.use_ue8m0,
            )


class Int4FlashInferMoEMethod(FusedMoEMethod):
    """INT4 routed-expert MoE via FlashInfer TensorRT-LLM MXINT4.

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
        if self.group_size != 32:
            raise ValueError(
                "FlashInfer MXINT4 MoE requires group_size=32, got "
                f"{self.group_size}"
            )

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
                device="cuda",
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
                device="cuda",
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
                device="cuda",
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
                device="cuda",
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        layer._int4_flashinfer_ready = False

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_int4_flashinfer_ready", False):
            return

        w13, s13 = ops.prepare_flashinfer_mxint4_moe_weight(
            layer.w13_weight_packed,
            layer.w13_weight_scale,
            gated=True,
        )
        layer.w13_weight_packed.data = w13
        layer.w13_weight_scale.data = s13

        w2, s2 = ops.prepare_flashinfer_mxint4_moe_weight(
            layer.w2_weight_packed,
            layer.w2_weight_scale,
            gated=False,
        )
        layer.w2_weight_packed.data = w2
        layer.w2_weight_scale.data = s2
        layer._int4_flashinfer_ready = True

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
        if apply_router_weight_on_input:
            raise ValueError(
                "FlashInfer MXINT4 MoE does not support applying router "
                "weights before the expert activation"
            )

        self.process_weights_after_loading(layer)
        if global_num_experts == -1:
            global_num_experts = layer.w13_weight_packed.shape[0]
        local_expert_offset = 0
        if expert_map is not None:
            local_expert_offset = layer.ep_rank * (
                global_num_experts // layer.ep_size
            )

        return ops.flashinfer_mxint4_moe(
            hidden_states=x,
            router_logits=router_logits,
            gemm1_weights=layer.w13_weight_packed,
            gemm1_scales=layer.w13_weight_scale,
            gemm2_weights=layer.w2_weight_packed,
            gemm2_scales=layer.w2_weight_scale,
            global_num_experts=global_num_experts,
            local_num_experts=layer.local_num_experts,
            local_expert_offset=local_expert_offset,
            top_k=top_k,
            intermediate_size=layer.intermediate_size_per_partition,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            scoring_func=scoring_func,
            correction_bias=e_score_correction_bias,
        )


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
        # In-place output is the fast default for routed-only MoE blocks. A
        # caller that also consumes the original input (most notably a shared
        # expert running concurrently) must opt out before the first forward;
        # otherwise the routed moe_sum overwrites that branch's GEMM input.
        self.inplace = True

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
            return Int4FlashInferMoEMethod(self.quant_config)
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
    expert_map = torch.full(
        (global_num_experts,), -1, dtype=torch.int32, device="cuda"
    )
    # Create a expert map for the local experts
    if ep_rank < (ep_size - 1):
        # Each non-last rank gets local_num_experts experts.
        expert_map[ep_rank * local_num_experts : (ep_rank + 1) * local_num_experts] = (
            torch.arange(
                0, local_num_experts, dtype=torch.int32, device=expert_map.device
            )
        )
    else:
        # All remaining experts are assigned to the last rank.
        local_num_experts = global_num_experts - ep_rank * local_num_experts

        expert_map[-local_num_experts:] = torch.arange(
            0,
            local_num_experts,
            dtype=torch.int32,
            device=expert_map.device,
        )
    return (local_num_experts, expert_map)
