import pytest
import torch

from gllm.layers.ops.triton_decode_attention import decode_attention_fwd

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
@pytest.mark.parametrize("num_heads", [2, 8])  # MHA kernel, then the grouped kernel
def test_pages_past_int32_element_offsets(num_heads):
    """A page whose element offset exceeds int32 must still be addressed.

    A layer view of a page-major cache arena strides over every layer of a
    page, so ``page * stride`` passes 2**31 elements long before the cache is
    full, even though the stride itself is a small int32.
    """
    page_size, kv_heads, dim = 16, 2, 64
    page_elems = page_size * kv_heads * dim
    stride = 2**20
    page = 2**31 // stride + 1  # its first element lies past the int32 range
    total = page * stride + 2 * page_elems
    if torch.cuda.mem_get_info()[0] < total * 2 + (1 << 30):
        pytest.skip("not enough free GPU memory for a > 2**31-element view")
    arena = torch.zeros(total, dtype=torch.bfloat16, device="cuda")
    shape = (page + 1, page_size, kv_heads, dim)
    strides = (stride, kv_heads * dim, dim, 1)
    k_cache = arena.as_strided(shape, strides)
    v_cache = arena.as_strided(shape, strides, storage_offset=page_elems)
    torch.manual_seed(0)
    keys = torch.randn(page_size, kv_heads, dim, device="cuda").bfloat16()
    values = torch.randn(page_size, kv_heads, dim, device="cuda").bfloat16()
    k_cache[page].copy_(keys)
    v_cache[page].copy_(values)

    seq_len = 11
    q = torch.randn(1, num_heads, dim, device="cuda").bfloat16()
    out = torch.empty_like(q)
    lse = torch.empty(1, num_heads, device="cuda", dtype=torch.float32)
    splits = 2
    scratch = torch.empty(1, num_heads, splits, dim + 1, device="cuda", dtype=torch.float32)
    scale = dim ** -0.5
    decode_attention_fwd(
        q, k_cache, v_cache, out, lse,
        torch.tensor([[page]], dtype=torch.int32, device="cuda"),
        torch.tensor([seq_len], dtype=torch.int32, device="cuda"),
        scratch, splits, scale, page_size,
    )

    group = num_heads // kv_heads
    k = keys[:seq_len].float().repeat_interleave(group, dim=1)  # [T, H, D]
    v = values[:seq_len].float().repeat_interleave(group, dim=1)
    scores = torch.einsum("hd,thd->ht", q[0].float(), k) * scale
    expected = torch.einsum("ht,thd->hd", scores.softmax(-1), v)
    torch.testing.assert_close(out[0].float(), expected, atol=2e-2, rtol=2e-2)
