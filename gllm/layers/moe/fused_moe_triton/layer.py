import torch

from enum import Enum
from typing import Optional, Callable
from torch.nn.parameter import UninitializedParameter

from gllm.utils import set_weight_attrs
from gllm.layers.moe.topk import select_experts
from gllm.layers.moe.fused_moe_triton.fused_moe import fused_experts

class FusedMoeWeightScaleSupported(Enum):
    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"
    BLOCK = "block"

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
                num_experts, 2 * intermediate_size_per_partition, hidden_size, dtype=params_dtype
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts, hidden_size, intermediate_size_per_partition, dtype=params_dtype
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
        inplace: bool = True,
        no_combine: bool = False,
    ) -> torch.Tensor:
        return self.forward_cuda(
            x=x,
            layer=layer,
            router_logits=router_logits,
            top_k=top_k,
            renormalize=renormalize,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            inplace=inplace,
            no_combine=no_combine,
        )

    def forward_cuda(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        inplace: bool = True,
        no_combine: bool = False,
    ) -> torch.Tensor:
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=top_k,
            renormalize=renormalize,
        )

        return fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=inplace and not no_combine,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            no_combine=no_combine,
        )

class FusedMoE(torch.nn.Module):
    """FusedMoE layer for MoE models.
    Args:
        num_experts: Number of experts in the model
        top_k: Number of experts selected for each token
        hidden_size: Input hidden state size of the transformer
        intermediate_size: Intermediate size of the experts
        params_dtype: Data type for the parameters.
        reduce_results: Whether to all all_reduce on the output of the layer
        renomalize: Whether to renormalize the logits in the fused_moe kernel
        quant_config: Quantization configure.
        inplace: suggestion to compute inplace (modify input activation).
    """
    
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        use_presharded_weights: bool = False,
        inplace: bool = True,
        no_combine: bool = False,
        routed_scaling_factor: Optional[float] = None,
    ):
        super().__init__()
        
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
            
        self.routed_scaling_factor = routed_scaling_factor
        self.top_k = top_k
        self.num_experts = num_experts
        self.intermediate_size_per_partition = intermediate_size
        self.reduce_results = reduce_results
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.custom_routing_function = custom_routing_function
        self.correction_bias = correction_bias
        self.activation = activation
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.use_presharded_weights = use_presharded_weights
        self.inplace = inplace
        self.no_combine = no_combine
        self.local_num_experts = num_experts
        
        self.fusedMoEMethod = FusedMoEMethod()
        
        self.fusedMoEMethod.create_weights(
            layer=self,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=self.intermediate_size_per_partition,
            params_dtype=params_dtype,
            weight_loader=self.weight_loader,
        )
        
    def weight_loader(self, param: torch.nn.Parameter,
                      loaded_weight: torch.Tensor, weight_name: str,
                      shard_id: str, expert_id: int) -> None:

        expert_id = self._map_global_expert_id_to_local_expert_id(expert_id)
        if expert_id == -1:
            return

        # compressed-tensors checkpoints with packed weights are stored flipped
        # TODO (mgoin): check self.quant_method.quant_config.quant_format
        # against known CompressionFormat enum values that have this quality
        loaded_weight = loaded_weight.t().contiguous() if (
            self.quant_method.__class__.__name__
            == "CompressedTensorsWNA16MoEMethod") else loaded_weight

        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"shard_id must be ['w1','w2','w3'] but "
                             f"got {shard_id}.")

        WEIGHT_SCALE_SUPPORTED = [
            e.value for e in FusedMoeWeightScaleSupported
        ]
        # Fetch the dim to shard the parameter/loaded weight
        # based on the shard id. This will be whatever
        # dimension intermediate_size_per_partition is used.
        SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0}

        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            param.weight_type = loaded_weight.item()
            param.data.copy_(loaded_weight)
            return

        # is_transposed: if the dim to shard the weight
        # should be flipped. Required by GPTQ, compressed-tensors
        # should be whatever dimension intermediate_size_per_partition is
        is_transposed = getattr(param, "is_transposed", False)
        shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]
        if is_transposed:
            shard_dim = int(not shard_dim)

        full_load = len(loaded_weight.shape) == 3
        if full_load:
            shard_dim += 1

        # Materialize GGUF UninitializedParameter
        if is_gguf_weight and isinstance(param, UninitializedParameter):
            final_shape = list(loaded_weight.shape)
            if shard_id in ["w1", "w3"]:
                final_shape[1] *= 2
            final_shape[shard_dim] = final_shape[shard_dim] // self.tp_size
            param.materialize(final_shape, dtype=loaded_weight.dtype)

        expert_data = param.data if full_load else param.data[expert_id]
        # Case input scale: input_scale loading is only supported for fp8
        if "input_scale" in weight_name:
            # this is needed for compressed-tensors only
            loaded_weight = loaded_weight.to(param.data.device)

            if param.data[expert_id] != 1 and (param.data[expert_id] -
                                               loaded_weight).abs() > 1e-5:
                raise ValueError(
                    "input_scales of w1 and w3 of a layer "
                    f"must be equal. But got {param.data[expert_id]} "
                    f"vs. {loaded_weight}")

            self._load_single_value(param=param,
                                    loaded_weight=loaded_weight,
                                    expert_id=expert_id)
            return

        # Case g_idx
        if "g_idx" in weight_name:
            self._load_g_idx(shard_dim=0,
                             shard_id=shard_id,
                             loaded_weight=loaded_weight,
                             expert_data=expert_data,
                             tp_rank=self.tp_rank)
            return

        # Case weight scales, zero_points and offset
        if ("scale" in weight_name or "zero" in weight_name
                or "offset" in weight_name):
            # load the weight scales and zp based on the quantization scheme
            # supported weight scales/zp can be found in
            # FusedMoeWeightScaleSupported
            # TODO @dsikka: once hardened, refactor to use vLLM Parameters
            # specific to each case
            quant_method = getattr(param, "quant_method", None)
            if quant_method == FusedMoeWeightScaleSupported.CHANNEL.value:
                self._load_per_channel_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank)
            elif quant_method in [
                    FusedMoeWeightScaleSupported.GROUP.value,
                    FusedMoeWeightScaleSupported.BLOCK.value,
            ]:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank,
                    load_full_w2=getattr(param, "load_full_w2", False))
            elif quant_method == FusedMoeWeightScaleSupported.TENSOR.value:
                self._load_per_tensor_weight_scale(shard_id=shard_id,
                                                   param=param,
                                                   loaded_weight=loaded_weight,
                                                   expert_id=expert_id)
            else:
                raise ValueError(
                    f"quant method must be one of {WEIGHT_SCALE_SUPPORTED}")
            return

        # Case weight_shape
        if "weight_shape" in weight_name:
            # only required by compressed-tensors
            self._load_single_value(param=param,
                                    loaded_weight=loaded_weight,
                                    expert_id=expert_id)
            return

        # Case model weights
        if "weight" in weight_name:
            self._load_model_weight_or_group_weight_scale(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=self.tp_rank)
            return
        
    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        # Matrix multiply.
        final_hidden_states = self.fusedMoEMethod.apply(
            layer=self,
            x=hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            renormalize=self.renormalize,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
        )

        return final_hidden_states