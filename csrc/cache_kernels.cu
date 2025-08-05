#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "dispatch_utils.h"
#include "cuda_utils.h"

#include "attention/attention_dtypes.h"


// The following macro is used to dispatch the conversion function based on
// the data type of the key and value cache. The FN is a macro that calls a
// function with template<typename scalar_t, typename cache_t,
// Fp8KVCacheDataType kv_dt>.
#define DISPATCH_BY_KV_CACHE_DTYPE(SRC_DTYPE, KV_DTYPE, FN)                  \
  if (KV_DTYPE == "auto") {                                                  \
    if (SRC_DTYPE == at::ScalarType::Float) {                                \
      FN(float, float, vllm::Fp8KVCacheDataType::kAuto);                     \
    } else if (SRC_DTYPE == at::ScalarType::Half) {                          \
      FN(uint16_t, uint16_t, vllm::Fp8KVCacheDataType::kAuto);               \
    } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                      \
      FN(__nv_bfloat16, __nv_bfloat16, vllm::Fp8KVCacheDataType::kAuto);     \
    } else {                                                                 \
      TORCH_CHECK(false, "Unsupported input type of kv cache: ", SRC_DTYPE); \
    }                                                                        \
  } else {                                                                   \
    if (KV_DTYPE == "fp8" || KV_DTYPE == "fp8_e4m3") {                       \
      if (SRC_DTYPE == at::ScalarType::Float) {                              \
        FN(float, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);              \
      } else if (SRC_DTYPE == at::ScalarType::Half) {                        \
        FN(uint16_t, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);           \
      } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                    \
        FN(__nv_bfloat16, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);      \
      } else {                                                               \
        TORCH_CHECK(false,                                                   \
                    "Unsupported input type of kv cache: ", SRC_DTYPE);      \
      }                                                                      \
    } else if (KV_DTYPE == "fp8_e5m2") {                                     \
      if (SRC_DTYPE == at::ScalarType::Float) {                              \
        FN(float, uint8_t, vllm::Fp8KVCacheDataType::kFp8E5M2);              \
      } else if (SRC_DTYPE == at::ScalarType::Half) {                        \
        FN(uint16_t, uint8_t, vllm::Fp8KVCacheDataType::kFp8E5M2);           \
      } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                    \
        FN(__nv_bfloat16, uint8_t, vllm::Fp8KVCacheDataType::kFp8E5M2);      \
      } else {                                                               \
        TORCH_CHECK(false,                                                   \
                    "Unsupported input type of kv cache: ", SRC_DTYPE);      \
      }                                                                      \
    } else {                                                                 \
      TORCH_CHECK(false, "Unsupported data type of kv cache: ", KV_DTYPE);   \
    }                                                                        \
  }

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

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void concat_and_cache_mla_kernel(
    const scalar_t* __restrict__ kv_c,  // [num_tokens, kv_lora_rank]
    const scalar_t* __restrict__ k_pe,  // [num_tokens, pe_dim]
    cache_t* __restrict__ kv_cache,  // [num_blocks, block_size, (kv_lora_rank
                                     // + pe_dim)]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int block_stride,                    //
    const int entry_stride,                    //
    const int kv_c_stride,                     //
    const int k_pe_stride,                     //
    const int kv_lora_rank,                    //
    const int pe_dim,                          //
    const int block_size,                      //
    const float* scale                         //
) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  // NOTE: slot_idx can be -1 if the token is padded
  if (slot_idx < 0) {
    return;
  }
  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  auto copy = [&](const scalar_t* __restrict__ src, cache_t* __restrict__ dst,
                  int src_stride, int dst_stride, int size, int offset) {
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
      const int64_t src_idx = token_idx * src_stride + i;
      const int64_t dst_idx =
          block_idx * block_stride + block_offset * entry_stride + i + offset;
      dst[dst_idx] = src[src_idx];
    }
  };

  copy(kv_c, kv_cache, kv_c_stride, block_stride, kv_lora_rank, 0);
  copy(k_pe, kv_cache, k_pe_stride, block_stride, pe_dim, kv_lora_rank);
}

// grid is launched with dimensions (batch, num_splits)
template <typename scalar_t>
__global__ void gather_cache(
    const scalar_t* __restrict__ src_cache,   // [NUM_BLOCKS, BLOCK_SIZE,
                                              // ENTRIES...]
    scalar_t* __restrict__ dst,               // [TOT_TOKENS, ENTRIES...]
    const int32_t* __restrict__ block_table,  // [BATCH, BLOCK_INDICES]
    const int32_t* __restrict__ cu_seq_lens,  // [BATCH+1]
    const int32_t block_size, const int32_t entry_size,
    const int64_t block_table_stride, const int64_t cache_block_stride,
    const int64_t cache_entry_stride, const int64_t dst_entry_stride,
    const int32_t* __restrict__ seq_starts) {  // Optional: starting offsets per
                                               // batch

  const int64_t bid = blockIdx.x;  // Batch ID
  const int32_t num_splits = gridDim.y;
  const int32_t split = blockIdx.y;
  const int32_t seq_start = cu_seq_lens[bid];
  const int32_t seq_end = cu_seq_lens[bid + 1];
  const int32_t seq_len = seq_end - seq_start;
  const int32_t tot_blocks = cuda_utils::ceil_div(seq_len, block_size);
  const int32_t split_blocks = cuda_utils::ceil_div(tot_blocks, num_splits);

  const int32_t split_start = split * split_blocks;
  const int32_t split_end = min((split + 1) * split_blocks, tot_blocks);

  const bool is_active_split = (split_start < tot_blocks);
  const bool is_last_split = (split_end == tot_blocks);

  if (!is_active_split) return;

  int32_t full_blocks_end = split_end;
  int32_t partial_block_size = 0;

  // Adjust the pointer for the block_table for this batch.
  // If seq_starts is provided, compute an offset based on (seq_starts[bid] /
  // page_size)
  const int32_t batch_offset = bid * block_table_stride;
  int32_t offset = 0;
  if (seq_starts != nullptr) {
    offset = seq_starts[bid] / block_size;
  }
  const int32_t* batch_block_table = block_table + batch_offset + offset;

  // Adjust dst pointer based on the cumulative sequence lengths.
  dst += seq_start * dst_entry_stride;

  if (is_last_split) {
    partial_block_size = seq_len % block_size;
    if (partial_block_size) full_blocks_end -= 1;
  }

  auto copy_entry = [&](const scalar_t* __restrict__ _src,
                        scalar_t* __restrict__ _dst) {
    for (int i = threadIdx.x; i < entry_size; i += blockDim.x)
      _dst[i] = _src[i];
  };

  for (int pid = split_start; pid < full_blocks_end; ++pid) {
    auto block_id = batch_block_table[pid];
    auto block_start_ptr = src_cache + block_id * cache_block_stride;
    auto block_dst_ptr = dst + pid * block_size * dst_entry_stride;
    for (int eid = 0; eid < block_size; ++eid) {
      copy_entry(block_start_ptr + eid * cache_entry_stride,
                 block_dst_ptr + eid * dst_entry_stride);
    }
  }

  if (partial_block_size) {
    auto block_id = batch_block_table[full_blocks_end];
    auto block_start_ptr = src_cache + block_id * cache_block_stride;
    auto block_dst_ptr = dst + full_blocks_end * block_size * dst_entry_stride;
    for (int eid = 0; eid < partial_block_size; ++eid) {
      copy_entry(block_start_ptr + eid * cache_entry_stride,
                 block_dst_ptr + eid * dst_entry_stride);
    }
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

// KV_T is the data type of key and value tensors.
// CACHE_T is the stored data type of kv-cache.
// KV_DTYPE is the real data type of kv-cache.
#define CALL_CONCAT_AND_CACHE_MLA(KV_T, CACHE_T, KV_DTYPE)              \
  vllm::concat_and_cache_mla_kernel<KV_T, CACHE_T, KV_DTYPE>            \
      <<<grid, block, 0, stream>>>(                                     \
          reinterpret_cast<KV_T*>(kv_c.data_ptr()),                     \
          reinterpret_cast<KV_T*>(k_pe.data_ptr()),                     \
          reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),              \
          slot_mapping.data_ptr<int64_t>(), block_stride, entry_stride, \
          kv_c_stride, k_pe_stride, kv_lora_rank, pe_dim, block_size,   \
          reinterpret_cast<const float*>(scale.data_ptr()));

void concat_and_cache_mla(
    torch::Tensor& kv_c,          // [num_tokens, kv_lora_rank]
    torch::Tensor& k_pe,          // [num_tokens, pe_dim]
    torch::Tensor& kv_cache,      // [num_blocks, block_size, (kv_lora_rank +
                                  // pe_dim)]
    torch::Tensor& slot_mapping,  // [num_tokens] or [num_actual_tokens]
    const std::string& kv_cache_dtype, torch::Tensor& scale) {
  // NOTE(woosuk): In vLLM V1, key.size(0) can be different from
  // slot_mapping.size(0) because of padding for CUDA graphs.
  // In vLLM V0, key.size(0) is always equal to slot_mapping.size(0) because
  // both include padding.
  // In vLLM V1, however, key.size(0) can be larger than slot_mapping.size(0)
  // since key includes padding for CUDA graphs, while slot_mapping does not.
  // In this case, slot_mapping.size(0) represents the actual number of tokens
  // before padding.
  // For compatibility with both cases, we use slot_mapping.size(0) as the
  // number of tokens.
  int num_tokens = slot_mapping.size(0);
  int kv_lora_rank = kv_c.size(1);
  int pe_dim = k_pe.size(1);
  int block_size = kv_cache.size(1);

  TORCH_CHECK(kv_cache.size(2) == kv_lora_rank + pe_dim);

  int kv_c_stride = kv_c.stride(0);
  int k_pe_stride = k_pe.stride(0);
  int block_stride = kv_cache.stride(0);
  int entry_stride = kv_cache.stride(1);

  dim3 grid(num_tokens);
  dim3 block(std::min(kv_lora_rank, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(kv_c));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_BY_KV_CACHE_DTYPE(kv_c.dtype(), kv_cache_dtype,
                             CALL_CONCAT_AND_CACHE_MLA);
}

// Macro to dispatch the kernel based on the data type.
#define CALL_GATHER_CACHE(CPY_DTYPE)                                    \
  vllm::gather_cache<CPY_DTYPE><<<grid, block, 0, stream>>>(            \
      reinterpret_cast<CPY_DTYPE*>(src_cache.data_ptr()),               \
      reinterpret_cast<CPY_DTYPE*>(dst.data_ptr()),                     \
      block_table.data_ptr<int32_t>(), cu_seq_lens.data_ptr<int32_t>(), \
      block_size, entry_size, block_table_stride, cache_block_stride,   \
      cache_entry_stride, dst_entry_stride, seq_starts_ptr);

// Gather sequences from the cache into the destination tensor.
//  - cu_seq_lens contains the cumulative sequence lengths for each batch
//  - block_table contains the cache block indices for each sequence
//  - Optionally, seq_starts (if provided) offsets the starting block index by
//  (seq_starts[bid] / page_size)
void gather_cache(
    torch::Tensor const& src_cache,    // [NUM_BLOCKS, BLOCK_SIZE, ENTRIES...]
    torch::Tensor const& dst,          // [TOT_TOKENS, ENTRIES...]
    torch::Tensor const& block_table,  // [BATCH, BLOCK_INDICES]
    torch::Tensor const& cu_seq_lens,  // [BATCH+1]
    int64_t batch_size,
    std::optional<torch::Tensor> seq_starts = std::nullopt) {
  at::cuda::OptionalCUDAGuard device_guard(src_cache.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int32_t block_size = src_cache.size(1);
  int32_t entry_size = src_cache.flatten(2, -1).size(2);

  TORCH_CHECK(block_table.dtype() == torch::kInt32,
              "block_table must be int32");
  TORCH_CHECK(cu_seq_lens.dtype() == torch::kInt32,
              "cu_seq_lens must be int32");
  if (seq_starts.has_value()) {
    TORCH_CHECK(seq_starts.value().dtype() == torch::kInt32,
                "seq_starts must be int32");
  }

  TORCH_CHECK(src_cache.device() == dst.device(),
              "src_cache and dst must be on the same device");
  TORCH_CHECK(src_cache.device() == block_table.device(),
              "src_cache and block_table must be on the same device");
  TORCH_CHECK(src_cache.device() == cu_seq_lens.device(),
              "src_cache and cu_seq_lens must be on the same device");
  if (seq_starts.has_value()) {
    TORCH_CHECK(src_cache.device() == seq_starts.value().device(),
                "src_cache and seq_starts must be on the same device");
  }

  int64_t block_table_stride = block_table.stride(0);
  int64_t cache_block_stride = src_cache.stride(0);
  int64_t cache_entry_stride = src_cache.stride(1);
  int64_t dst_entry_stride = dst.stride(0);

  // Decide on the number of splits based on the batch size.
  int num_splits = batch_size > 128 ? 2 : batch_size > 64 ? 4 : 16;
  dim3 grid(batch_size, num_splits);
  dim3 block(1024);

  TORCH_CHECK(src_cache.dtype() == dst.dtype(),
              "src_cache and dst must have the same dtype");

  const int dtype_bits = src_cache.element_size() * 8;
  const int32_t* seq_starts_ptr =
      seq_starts.has_value() ? seq_starts.value().data_ptr<int32_t>() : nullptr;

  if (dtype_bits == 32) {
    CALL_GATHER_CACHE(uint32_t);
  } else if (dtype_bits == 16) {
    CALL_GATHER_CACHE(uint16_t);
  } else if (dtype_bits == 8) {
    CALL_GATHER_CACHE(uint8_t);
  } else {
    TORCH_CHECK(false, "Unsupported data type width: ", dtype_bits);
  }
}
