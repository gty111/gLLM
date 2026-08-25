import pytest
import torch

from gllm.layers.ops.gemma_rmsnorm import gemma_rms_norm_reference_reduction
from gllm.layers.ops.mtp_embed_norm import fused_mtp_embed_hidden_gemma_norm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("rows", [1, 32, 128])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_fused_mtp_embed_norm_is_bitwise_exact(rows, weight_dtype):
    torch.manual_seed(29)
    vocab_size = 257
    hidden_size = 5_120
    table = torch.randn(
        vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16
    )
    token_ids = torch.randint(
        0, vocab_size, (rows,), device="cuda", dtype=torch.int64
    )
    hidden = torch.randn(
        rows, hidden_size, device="cuda", dtype=torch.bfloat16
    )
    embedding_weight = (
        torch.randn(hidden_size, device="cuda", dtype=weight_dtype) * 0.05
    )
    hidden_weight = (
        torch.randn(hidden_size, device="cuda", dtype=weight_dtype) * 0.05
    )

    embedding = torch.nn.functional.embedding(token_ids, table)
    expected_embedding = torch.empty_like(embedding)
    expected_hidden = torch.empty_like(hidden)
    gemma_rms_norm_reference_reduction(
        expected_embedding, embedding, embedding_weight, 1e-6
    )
    gemma_rms_norm_reference_reduction(
        expected_hidden, hidden, hidden_weight, 1e-6
    )
    expected = torch.cat((expected_embedding, expected_hidden), dim=-1)

    actual = fused_mtp_embed_hidden_gemma_norm(
        token_ids,
        table,
        hidden,
        embedding_weight,
        hidden_weight,
        1e-6,
    )
    assert torch.equal(actual, expected)
