#pragma once

#include <optional>
#include <torch/library.h>


void rms_norm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight,
              double epsilon);

void fused_add_rms_norm(torch::Tensor& input, torch::Tensor& residual,
                        torch::Tensor& weight, double epsilon);

void rotary_embedding(torch::Tensor& positions, torch::Tensor& query,
                      torch::Tensor& key, int64_t head_size,
                      torch::Tensor& cos_sin_cache, bool is_neox);

void batched_rotary_embedding(torch::Tensor& positions, torch::Tensor& query,
                              torch::Tensor& key, int64_t head_size,
                              torch::Tensor& cos_sin_cache, bool is_neox,
                              int64_t rot_dim,
                              torch::Tensor& cos_sin_cache_offsets);

void silu_and_mul(torch::Tensor& out, torch::Tensor& input);

void gelu_and_mul(torch::Tensor& out, torch::Tensor& input);

void gelu_tanh_and_mul(torch::Tensor& out, torch::Tensor& input);

void gelu_new(torch::Tensor& out, torch::Tensor& input);

void gelu_fast(torch::Tensor& out, torch::Tensor& input);

void gelu_quick(torch::Tensor& out, torch::Tensor& input);

void reshape_and_cache_flash(
    torch::Tensor& key, torch::Tensor& value, 
    torch::Tensor& key_cache, torch::Tensor& value_cache, 
    torch::Tensor& slot_mapping);