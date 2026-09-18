"""Exercise the Qwen vision head dimension with inference-mode metadata."""

import pytest
import torch

from gllm.layers.ops.flash_attn_compat import flash_attn_varlen_func


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("return_lse", [False, True])
@torch.inference_mode()
def test_vision_attention_uses_updated_inference_lengths(return_lse):
    torch.manual_seed(123)
    q = torch.randn(7, 16, 72, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(9, 16, 72, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    starts_q = torch.tensor([0, 2, 7], device="cuda", dtype=torch.int32)
    starts_k = torch.tensor([0, 3, 9], device="cuda", dtype=torch.int32)
    for q_bounds, k_bounds in [([0, 2, 7], [0, 3, 9]), ([0, 4, 7], [0, 5, 9])]:
        starts_q.copy_(torch.tensor(q_bounds, device="cuda", dtype=torch.int32))
        starts_k.copy_(torch.tensor(k_bounds, device="cuda", dtype=torch.int32))
        output = flash_attn_varlen_func(
            q, k, v, starts_q, starts_k, backend="flashinfer", causal=False,
            max_seqlen_q=7, max_seqlen_k=9,
            return_softmax_lse=return_lse,
        )
        if return_lse:
            output, lse = output
        expected = []
        expected_lse = []
        for i in range(2):
            qi = q[q_bounds[i]:q_bounds[i + 1]].float().transpose(0, 1)
            ki = k[k_bounds[i]:k_bounds[i + 1]].float().transpose(0, 1)
            vi = v[k_bounds[i]:k_bounds[i + 1]].float().transpose(0, 1)
            scores = (qi @ ki.transpose(-1, -2)) * 72 ** -0.5
            expected.append((scores.softmax(-1) @ vi).transpose(0, 1))
            expected_lse.append(scores.logsumexp(-1))
        torch.testing.assert_close(output.float(), torch.cat(expected), atol=0.02, rtol=0.02)
        if return_lse:
            torch.testing.assert_close(lse, torch.cat(expected_lse, dim=1), atol=0.02, rtol=0.02)
