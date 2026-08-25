#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

namespace {

template <typename weight_t>
__device__ __forceinline__ float load_weight(const weight_t* weight, int col);

template <>
__device__ __forceinline__ float load_weight<__nv_bfloat16>(
    const __nv_bfloat16* weight, int col) {
  return __bfloat162float(weight[col]);
}

template <>
__device__ __forceinline__ float load_weight<float>(
    const float* weight, int col) {
  return weight[col];
}

template <typename weight_t, bool has_residual>
__global__ void gemma_rmsnorm_bf16_kernel(
    __nv_bfloat16* input,
    __nv_bfloat16* residual,
    const weight_t* weight,
    __nv_bfloat16* output,
    float epsilon,
    float mean_factor,
    int n_cols,
    int64_t input_stride,
    int64_t residual_stride,
    int64_t output_stride) {
  extern __shared__ float squares[];
  const int row = blockIdx.x;
  __nv_bfloat16* input_row = input + row * input_stride;
  __nv_bfloat16* residual_row = residual + row * residual_stride;
  __nv_bfloat16* output_row = output + row * output_stride;

  for (int col = threadIdx.x; col < n_cols; col += blockDim.x) {
    float value = __bfloat162float(input_row[col]);
    if constexpr (has_residual) {
      value = __fadd_rn(value, __bfloat162float(residual_row[col]));
      const __nv_bfloat16 folded = __float2bfloat16_rn(value);
      residual_row[col] = folded;
      value = __bfloat162float(folded);
    }
    squares[col] = __fmul_rn(value, value);
  }
  __syncthreads();

  if (threadIdx.x < 32) {
    const int lane = threadIdx.x;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    for (int col = lane * 4; col < n_cols; col += 128) {
      acc0 = __fadd_rn(acc0, squares[col]);
      acc1 = __fadd_rn(acc1, squares[col + 1]);
      acc2 = __fadd_rn(acc2, squares[col + 2]);
      acc3 = __fadd_rn(acc3, squares[col + 3]);
    }
    float sum = __fadd_rn(acc0, acc1);
    sum = __fadd_rn(sum, acc2);
    sum = __fadd_rn(sum, acc3);
    for (int offset = 16; offset > 0; offset >>= 1) {
      sum = __fadd_rn(
          sum, __shfl_down_sync(0xffffffff, sum, offset));
    }
    if (lane == 0) {
      const float variance = __fmul_rn(sum, mean_factor);
      squares[0] = rsqrtf(__fadd_rn(variance, epsilon));
    }
  }
  __syncthreads();

  const float inv_rms = squares[0];
  const __nv_bfloat16* source = has_residual ? residual_row : input_row;
  for (int col = threadIdx.x; col < n_cols; col += blockDim.x) {
    const float value = __bfloat162float(source[col]);
    const float normalized = __fmul_rn(value, inv_rms);
    const float scale = __fadd_rn(load_weight(weight, col), 1.0f);
    output_row[col] = __float2bfloat16_rn(
        __fmul_rn(normalized, scale));
  }
}

template <typename weight_t>
void launch_gemma_rmsnorm(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight,
    torch::Tensor output,
    double epsilon,
    bool has_residual) {
  const int n_cols = input.size(-1);
  const int64_t n_rows = input.numel() / n_cols;
  const float mean_factor = 1.0f / static_cast<float>(n_cols);
  // Per-head Q/K norms are only 256 elements wide. A 1024-thread CTA leaves
  // three quarters of its threads idle and limits occupancy to one CTA/SM;
  // 256 threads cover the row in one pass while preserving the exact first-
  // warp reduction tree below. Hidden-state norms keep 1024 threads so their
  // 2K-5K element rows retain coalesced parallel loads/stores.
  const dim3 block(1024);
  const dim3 grid(n_rows);
  const size_t shared_bytes = n_cols * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream(input.device().index());
  auto* input_ptr = reinterpret_cast<__nv_bfloat16*>(input.data_ptr());
  auto* residual_ptr = reinterpret_cast<__nv_bfloat16*>(residual.data_ptr());
  auto* output_ptr = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());
  auto* weight_ptr = reinterpret_cast<const weight_t*>(weight.data_ptr());
  if (has_residual) {
    gemma_rmsnorm_bf16_kernel<weight_t, true><<<
        grid, block, shared_bytes, stream>>>(
        input_ptr,
        residual_ptr,
        weight_ptr,
        output_ptr,
        static_cast<float>(epsilon),
        mean_factor,
        n_cols,
        input.stride(-2),
        residual.stride(-2),
        output.stride(-2));
  } else {
    gemma_rmsnorm_bf16_kernel<weight_t, false><<<
        grid, block, shared_bytes, stream>>>(
        input_ptr,
        residual_ptr,
        weight_ptr,
        output_ptr,
        static_cast<float>(epsilon),
        mean_factor,
        n_cols,
        input.stride(-2),
        residual.stride(-2),
        output.stride(-2));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void gemma_rmsnorm_bf16(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight,
    torch::Tensor output,
    double epsilon,
    bool has_residual) {
  TORCH_CHECK(input.is_cuda(), "input must be CUDA");
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input must be BF16");
  TORCH_CHECK(residual.scalar_type() == torch::kBFloat16, "residual must be BF16");
  TORCH_CHECK(output.scalar_type() == torch::kBFloat16, "output must be BF16");
  TORCH_CHECK(input.stride(-1) == 1, "input feature dimension must be contiguous");
  TORCH_CHECK(residual.stride(-1) == 1, "residual feature dimension must be contiguous");
  TORCH_CHECK(output.stride(-1) == 1, "output feature dimension must be contiguous");
  TORCH_CHECK(input.size(-1) % 128 == 0, "hidden size must be divisible by 128");
  const c10::cuda::CUDAGuard device_guard(input.device());
  if (weight.scalar_type() == torch::kBFloat16) {
    launch_gemma_rmsnorm<__nv_bfloat16>(
        input, residual, weight, output, epsilon, has_residual);
  } else if (weight.scalar_type() == torch::kFloat32) {
    launch_gemma_rmsnorm<float>(
        input, residual, weight, output, epsilon, has_residual);
  } else {
    TORCH_CHECK(false, "weight must be BF16 or FP32");
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("gemma_rmsnorm_bf16", &gemma_rmsnorm_bf16);
}
