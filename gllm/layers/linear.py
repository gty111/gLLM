from functools import partial
from typing import Optional, Union

import torch
from torch.nn.parameter import Parameter

from gllm.dist_utils import (
    divide,
    get_tp_rank,
    get_tp_size,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_reduce,
)
from gllm.layers.quantization.fp8 import fp8LinearMethod, validate_fp8_block_shape
from gllm.utils import get_device_capability


class LinearBase(torch.nn.Module):
    """Base linear layer.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        bias: If true, add bias.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        return_bias: If true, return bias together with outputs in forward pass.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        return_bias: bool = False,
        quant_config=None,
    ):
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        self.return_bias = return_bias
        self.quant_config = quant_config

    def create_weights(
        self,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        params_dtype: torch.dtype,
    ):
        if self.quant_config is None:
            weight = Parameter(
                torch.rand(
                    sum(output_partition_sizes),
                    input_size_per_partition,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )
            self.register_parameter("weight", weight)
        elif self.quant_config["quant_method"] == "fp8":
            if get_device_capability() < 89:
                raise Exception(
                    f"FP8 quantizaiton method is not supported on device capability less than 89 (current is {get_device_capability()})"
                )
            self.activation_scheme = self.quant_config["activation_scheme"]
            self.block_quant = "weight_block_size" in self.quant_config
            if self.block_quant:
                self.weight_block_size = self.quant_config["weight_block_size"]
                block_n, block_k = self.weight_block_size
                validate_fp8_block_shape(
                    layer=self,
                    input_size=self.input_size,
                    output_size=self.output_size,
                    input_size_per_partition=input_size_per_partition,
                    output_partition_sizes=output_partition_sizes,
                    block_size=self.weight_block_size,
                )
            weight_dtype = torch.float8_e4m3fn
            weight = Parameter(
                torch.rand(sum(output_partition_sizes), input_size_per_partition).to(
                    weight_dtype
                ),
                requires_grad=False,
            )
            self.register_parameter("weight", weight)
            if not self.block_quant:
                scale = Parameter(
                    torch.rand(len(output_partition_sizes), dtype=torch.float32),
                    requires_grad=False,
                )
                scale[:] = torch.finfo(torch.float32).min
                self.register_parameter("weight_scale", scale)
            else:
                assert self.activation_scheme == "dynamic"
                scale = Parameter(
                    torch.rand(
                        (sum(output_partition_sizes) + block_n - 1) // block_n,
                        (input_size_per_partition + block_k - 1) // block_k,
                        dtype=torch.float32,
                    ),
                    requires_grad=False,
                )
                scale[:] = torch.finfo(torch.float32).min
                self.register_parameter("weight_scale_inv", scale)

            if self.activation_scheme == "static":
                scale = Parameter(
                    torch.rand(len(output_partition_sizes), dtype=torch.float32),
                    requires_grad=False,
                )
                scale[:] = torch.finfo(torch.float32).min
                self.register_parameter("input_scale", scale)
            else:
                self.register_parameter("input_scale", None)
        else:
            raise Exception(
                f"gLLM do not support quant_method {self.quant_config['quant_method']}"
            )

    def dispatch_quant_method(self):
        if self.quant_config is None:
            return torch.nn.functional.linear
        elif self.quant_config["quant_method"] == "fp8":
            assert self.block_quant
            return partial(
                fp8LinearMethod,
                block_size=self.weight_block_size,
                weight_scale=self.weight_scale_inv,
                input_scale=self.input_scale,
            )
        else:
            raise Exception(
                f"gLLM do not support quant_method {self.quant_config['quant_method']}"
            )

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        raise NotImplementedError


class RowParallelLinear(LinearBase):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        skip_bias_add: This was added to enable performance optimization where
                       bias can be fused with other element-wise operations.
                       We skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        reduce_results: If true, call all-reduce on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y = X_iA_i
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.down_proj)
        return_bias: If true, return bias together with outputs in forward pass.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = True,
        return_bias: bool = False,
        quant_config=None,
    ):
        # Divide the weight matrix along the first dimension.
        self.tp_rank = get_tp_rank()
        self.tp_size = get_tp_size()
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size
        self.output_partition_sizes = [output_size]

        super().__init__(
            input_size,
            output_size,
            skip_bias_add,
            params_dtype,
            return_bias=return_bias,
            quant_config=quant_config,
        )

        self.create_weights(
            self.input_size_per_partition, self.output_partition_sizes, params_dtype
        )

        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results

        if not reduce_results and (bias and not skip_bias_add):
            raise ValueError(
                "When not reduce the results, adding bias to the "
                "results can lead to incorrect results"
            )

        if bias:
            self.bias = Parameter(torch.rand(self.output_size, dtype=params_dtype))
        else:
            self.register_parameter("bias", None)

        self.quant_method = self.dispatch_quant_method()

    def forward(
        self, input_
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        if self.input_is_parallel:
            input_parallel = input_
        else:
            tp_rank = self.tp_rank
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size
            )
            input_parallel = splitted_input[tp_rank].contiguous()

        # Matrix multiply.
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)

        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output_parallel = self.quant_method(input_parallel, self.weight, bias=bias_)

        if self.reduce_results and self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel

        output_bias = self.bias if self.skip_bias_add else None

        if not self.return_bias:
            return output
        return output, output_bias


class ColumnParallelLinear(LinearBase):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Args:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias.
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        output_sizes: list of output sizes packed into one output, like for QKV
                       the list would be size 3.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        output_sizes: Optional[list[int]] = None,
        return_bias: bool = False,
        quant_config=None,
        disable_tp: bool = False,
    ):
        # Divide the weight matrix along the last dimension.
        self.tp_size = get_tp_size() if not disable_tp else 1
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)
        self.output_partition_sizes = [self.output_size_per_partition]
        # If QKV or MergedColumn, use output size of each partition.
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = [
                divide(output_size, self.tp_size) for output_size in self.output_sizes
            ]

        super().__init__(
            input_size,
            output_size,
            skip_bias_add,
            params_dtype,
            return_bias=return_bias,
            quant_config=quant_config,
        )

        self.create_weights(
            self.input_size_per_partition, self.output_partition_sizes, params_dtype
        )

        if output_sizes is None:
            output_sizes = [output_size]

        if bias:
            self.bias = Parameter(
                torch.rand(self.output_size_per_partition, dtype=params_dtype)
            )
        else:
            self.register_parameter("bias", None)

        self.quant_method = self.dispatch_quant_method()

    def forward(
        self, input_
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        bias = self.bias if not self.skip_bias_add else None

        # Matrix multiply.
        output_parallel = self.quant_method(input_, self.weight, bias=bias)
        output_bias = self.bias if self.skip_bias_add else None
        if not self.return_bias:
            return output_parallel
        return output_parallel, output_bias


class MergedColumnParallelLinear(ColumnParallelLinear):
    """Packed linear layers with column parallelism.

    Similar to ColumnParallelLinear, but the weight matrix is concatenated
    along the output dimension. When the weight matrix is loaded, the
    different partitions are sharded separately.

    Args:
        input_size: input dimension of the linear layer.
        output_sizes: list of output dimensions of the linear layer.
        bias: If true, add bias.
        gather_output: If true, call all-gather on output and make the output
                       available to all GPUs, otherwise, every GPU will have
                       its own output.
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
        return_bias: If true, return bias together with outputs in forward pass.
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        return_bias: bool = False,
        quant_config=None,
        disable_tp: bool = False,
    ):
        self.output_sizes = output_sizes
        tp_size = get_tp_size() if not disable_tp else 1
        assert all(output_size % tp_size == 0 for output_size in output_sizes)
        super().__init__(
            input_size=input_size,
            output_size=sum(output_sizes),
            bias=bias,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            return_bias=return_bias,
            quant_config=quant_config,
            disable_tp=disable_tp,
        )


class QKVParallelLinear(ColumnParallelLinear):
    """Linear layers for the attention's QKV transformation.

    Linear layers for the linear transformation of the query, key, and value
    vectors in the attention layer. The weight matrix is concatenated along
    the output dimension. The layer is parallelized along the head dimension.
    When the number of key/value heads is smaller than the number of query
    heads (e.g., multi-query/grouped-query attention), the key/value head may
    be replicated while the query heads are partitioned.

    Args:
        hidden_size: input hidden state size of the transformer.
        head_size: size of each attention head.
        total_num_heads: total number of attention query heads.
        total_num_kv_heads: total number of attention key/value heads. If
                            None, assume total_num_kv_heads = total_num_heads.
        bias: If true, add bias.
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
        return_bias: If true, return bias together with outputs in forward pass.
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: Optional[int] = None,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        return_bias: bool = False,
        quant_config=None,
    ):
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        # Divide the weight matrix along the last dimension.
        tp_size = get_tp_size()
        self.num_heads = divide(self.total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        input_size = self.hidden_size
        output_size = (
            (self.num_heads + 2 * self.num_kv_heads) * tp_size * self.head_size
        )
        self.output_sizes = [
            self.num_heads * self.head_size * tp_size,  # q_proj
            self.num_kv_heads * self.head_size * tp_size,  # k_proj
            self.num_kv_heads * self.head_size * tp_size,  # v_proj
        ]

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            return_bias=return_bias,
            quant_config=quant_config,
        )


class ReplicatedLinear(LinearBase):
    """Replicated linear layer.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        bias: If true, add bias.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
        return_bias: If true, return bias together with outputs in forward pass.
        disable_tp: Take no effect for replicated linear layers.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config=None,
        *,
        return_bias: bool = False,
    ):
        # If MergedReplicatedLinear, use output size of each partition.
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = self.output_sizes
        else:
            self.output_partition_sizes = [output_size]

        super().__init__(
            input_size,
            output_size,
            skip_bias_add,
            params_dtype,
            return_bias=return_bias,
            quant_config=quant_config,
        )

        self.create_weights(
            self.input_size,
            self.output_partition_sizes,
            self.params_dtype,
        )

        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size, dtype=self.params_dtype)
            )
        else:
            self.register_parameter("bias", None)

        self.quant_method = self.dispatch_quant_method()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        bias = self.bias if not self.skip_bias_add else None
        assert self.quant_method is not None

        output = self.quant_method(x, self.weight, bias=bias)
        output_bias = self.bias if self.skip_bias_add else None

        if not self.return_bias:
            return output
        return output, output_bias
