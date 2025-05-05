#include "core/registration.h"
#include "moe_ops.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  // Apply topk softmax to the gating outputs.
  m.def(
      "topk_softmax(Tensor! topk_weights, Tensor! topk_indices, Tensor! "
      "token_expert_indices, Tensor gating_output) -> ()");
  m.impl("topk_softmax", torch::kCUDA, &topk_softmax);

  // Calculate the result of moe by summing up the partial results
  // from all selected experts.
  m.def("moe_sum(Tensor! input, Tensor output) -> ()");
  m.impl("moe_sum", torch::kCUDA, &moe_sum);

  // Aligning the number of tokens to be processed by each expert such
  // that it is divisible by the block size.
  m.def(
      "moe_align_block_size(Tensor topk_ids, int num_experts,"
      "                     int block_size, Tensor! sorted_token_ids,"
      "                     Tensor! experts_ids,"
      "                     Tensor! num_tokens_post_pad) -> ()");
  m.impl("moe_align_block_size", torch::kCUDA, &moe_align_block_size);

  // temporarily adapted from
  // https://github.com/sgl-project/sglang/commit/ded9fcd09a43d5e7d5bb31a2bc3e9fc21bf65d2a
  m.def(
      "sgl_moe_align_block_size(Tensor topk_ids, int num_experts,"
      "                         int block_size, Tensor! sorted_token_ids,"
      "                         Tensor! experts_ids,"
      "                         Tensor! num_tokens_post_pad) -> ()");
  m.impl("sgl_moe_align_block_size", torch::kCUDA, &sgl_moe_align_block_size);


}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
