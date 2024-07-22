#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "dispatch_utils.h"

// namespace gllm{
    
// template <typename scalar_t>
// __global__ void store_cache_kernel(
//     const scalar_t* __restrict__ key,   // [num_tokens, num_heads, head_dim]
//     const scalar_t* __restrict__ value, // [num_tokens, num_heads, head_dim]
//     scalar_t* __restrict__ key_cache,   // [num_pages, num_token_page, num_heads, head_dim]
//     scalar_t* __restrict__ value_cache, // [num_pages, num_token_page, num_heads, head_dim]
//     const int64_t* __restrict__ slot_mapping, // [num_tokens]
//     const int num_heads, 
//     const int head_dim
// ){
//     const int64_t token_idx = blockIdx.x;
//     const int64_t slot_idx = slot_mapping[token_idx];
//     const int n = num_heads * head_dim;
    
//     for (int i = threadIdx.x ; i < n; i += blockDim.x){
//         const int64_t dst_idx = slot_idx * n + i;
//         const int64_t src_idx = token_idx * n + i;
//         key_cache[dst_idx] = key[src_idx];
//         value_cache[dst_idx] = value[src_idx];
//         printf("%ld %ld %ld %ld %f %f\n",token_idx,slot_idx,dst_idx,src_idx,(float)key[src_idx],(float)value[src_idx]);
//     }
// }

// #define CALL_STORE_CACHE(DTYPE)                                       \
//   gllm::store_cache_kernel<DTYPE>                                     \
//       <<<grid, block, 0, stream>>>(                                   \
//           reinterpret_cast<DTYPE*>(key.data_ptr()),                   \
//           reinterpret_cast<DTYPE*>(value.data_ptr()),                 \
//           reinterpret_cast<DTYPE*>(key_cache.data_ptr()),             \
//           reinterpret_cast<DTYPE*>(value_cache.data_ptr()),           \
//           slot_mapping.data_ptr<int64_t>(), num_heads, head_dim);

// }

// void store_cache(
//     torch::Tensor& key,         // [num_tokens, num_heads, head_dim]
//     torch::Tensor& value,       // [num_tokens, num_heads, head_dim]
//     torch::Tensor& key_cache,   // [num_pages, num_token_page, num_heads, head_dim]
//     torch::Tensor& value_cache, // [num_pages, num_token_page, num_heads, head_dim]
//     torch::Tensor& slot_mapping // [num_tokens]
// ){
//     int num_tokens = key.size(0);
//     int num_heads = key.size(1);
//     int head_dim = key.size(2);

//     dim3 grid(num_tokens);
//     dim3 block(std::min(num_heads*head_dim,512));

//     const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
//     if(key.dtype() == at::ScalarType::BFloat16){
//         CALL_STORE_CACHE(__nv_bfloat16)
//     }else{
//         assert(0);
//     }
// }
namespace vllm{

template <typename scalar_t>
__global__ void reshape_and_cache_flash_kernel(
    const scalar_t* __restrict__ key,    // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,  // [num_tokens, num_heads, head_size]
    scalar_t* __restrict__ k_cache,      // [num_blocks, block_size, num_heads,
                                         // head_size]
    scalar_t* __restrict__ v_cache,      // [num_blocks, block_size, num_heads,
                                         // head_size]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int block_stride, const int key_stride, const int value_stride,
    const int num_heads, const int head_size, const int block_size) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  // NOTE: slot_idx can be -1 if the token is padded
  if (slot_idx < 0) {
    return;
  }
  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int n = num_heads * head_size;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t src_key_idx = token_idx * key_stride + i;
    const int64_t src_value_idx = token_idx * value_stride + i;
    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int64_t tgt_value_idx = block_idx * block_stride +
                                  block_offset * num_heads * head_size +
                                  head_idx * head_size + head_offset;
    k_cache[tgt_value_idx] = key[src_key_idx];
    v_cache[tgt_value_idx] = value[src_value_idx];
  }
}
}  // namespace vllm


void reshape_and_cache_flash(
    torch::Tensor& key,      // [num_tokens, num_heads, head_size]
    torch::Tensor& value,    // [num_tokens, num_heads, head_size]
    torch::Tensor& k_cache,  // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor& v_cache,  // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor& slot_mapping  // [num_tokens]
) {
  int num_tokens = key.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = k_cache.size(1);

  int key_stride = key.stride(0);
  int value_stride = value.stride(0);
  int block_stride = k_cache.stride(0);
  TORCH_CHECK(k_cache.stride(0) == v_cache.stride(0));

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      key.scalar_type(), "reshape_and_cache_flash", [&] {
        vllm::reshape_and_cache_flash_kernel<scalar_t>
            <<<grid, block, 0, stream>>>(
                key.data_ptr<scalar_t>(), value.data_ptr<scalar_t>(),
                k_cache.data_ptr<scalar_t>(), v_cache.data_ptr<scalar_t>(),
                slot_mapping.data_ptr<int64_t>(), block_stride, key_stride,
                value_stride, num_heads, head_size, block_size);
      });
}