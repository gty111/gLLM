
#include <torch/all.h>
#include "ops.h"
#include "core/registration.h"

// #include <torch/library.h>

// Note on op signatures:
// The X_meta signatures are for the meta functions corresponding to op X.
// They must be kept in sync with the signature for X. Generally, only
// functions that return Tensors require a meta function.
//
// See the following links for detailed docs on op registration and function
// schemas.
// https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ptttacy8y1u9
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#annotations

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops)
{
    // vLLM custom ops

    // Activation ops
    // Activation function used in SwiGLU.
    ops.def("silu_and_mul(Tensor! out, Tensor input) -> ()");
    ops.impl("silu_and_mul", torch::kCUDA, &silu_and_mul);

    // Activation function used in GeGLU with `none` approximation.
    ops.def("gelu_and_mul(Tensor! out, Tensor input) -> ()");
    ops.impl("gelu_and_mul", torch::kCUDA, &gelu_and_mul);

    // Activation function used in GeGLU with `tanh` approximation.
    ops.def("gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()");
    ops.impl("gelu_tanh_and_mul", torch::kCUDA, &gelu_tanh_and_mul);

    // GELU implementation used in GPT-2.
    ops.def("gelu_new(Tensor! out, Tensor input) -> ()");
    ops.impl("gelu_new", torch::kCUDA, &gelu_new);

    // Approximate GELU implementation.
    ops.def("gelu_fast(Tensor! out, Tensor input) -> ()");
    ops.impl("gelu_fast", torch::kCUDA, &gelu_fast);

    // Quick GELU implementation.
    ops.def("gelu_quick(Tensor! out, Tensor input) -> ()");
    ops.impl("gelu_quick", torch::kCUDA, &gelu_quick);

    // Layernorm
    // Apply Root Mean Square (RMS) Normalization to the input tensor.
    ops.def(
        "rms_norm(Tensor! out, Tensor input, Tensor weight, float epsilon) -> "
        "()");
    ops.impl("rms_norm", torch::kCUDA, &rms_norm);

    // In-place fused Add and RMS Normalization.
    ops.def(
        "fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor weight, "
        "float epsilon) -> ()");
    ops.impl("fused_add_rms_norm", torch::kCUDA, &fused_add_rms_norm);

    // Rotary embedding
    // Apply GPT-NeoX or GPT-J style rotary embedding to query and key.
    ops.def(
        "rotary_embedding(Tensor positions, Tensor! query,"
        "                 Tensor! key, int head_size,"
        "                 Tensor cos_sin_cache, bool is_neox) -> ()");
    ops.impl("rotary_embedding", torch::kCUDA, &rotary_embedding);

    // Apply GPT-NeoX or GPT-J style rotary embedding to query and key
    // (supports multiple loras).
    ops.def(
        "batched_rotary_embedding(Tensor positions, Tensor! query,"
        "                         Tensor! key, int head_size,"
        "                         Tensor cos_sin_cache, bool is_neox,"
        "                         int rot_dim,"
        "                         Tensor cos_sin_cache_offsets) -> ()");
    ops.impl("batched_rotary_embedding", torch::kCUDA, &batched_rotary_embedding);

    // Cache ops
    ops.def(
        "reshape_and_cache_flash(Tensor key, Tensor value,"
        "            Tensor! key_cache, Tensor! value_cache,"
        "            Tensor slot_mapping) -> ()");
    ops.impl("reshape_and_cache_flash", torch::kCUDA, &reshape_and_cache_flash);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
