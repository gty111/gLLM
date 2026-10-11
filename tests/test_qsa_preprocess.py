"""Compare fused publication against the original norm/RoPE/cache operators."""

import pytest
import torch
from torch.nn import functional as F

from gllm.layers.attention.qsa import apply_indexer_rope, indexer_rotary_cache
from gllm.layers.layernorm import GemmaRMSNorm
from gllm.layers.ops.qsa import prepare_qsa
from gllm.layers.rotary_embedding import MRotaryEmbedding, RotaryEmbedding

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@torch.inference_mode()
def test_preprocess_widens_large_cache_page_offsets():
    pages, stride = 32770, 65536
    elements = (pages - 1) * stride + 16 * 768
    if torch.cuda.mem_get_info()[0] < elements * 2 + (1 << 30):
        pytest.skip("Requires approximately 4 GB of free GPU memory")
    storage = torch.empty(elements, device="cuda", dtype=torch.bfloat16)
    keys = storage.as_strided((pages, 16, 1, 64), (stride, 768, 64, 1))
    values = storage.as_strided((pages, 16, 1, 64), (stride, 768, 64, 1), 128)
    cache = storage.as_strided((pages, 16, 384), (stride, 768, 1), 256)
    rope = RotaryEmbedding(64, 32, 128, 10000, True)
    qkv = torch.randn(1, 384, device="cuda", dtype=torch.bfloat16)
    index = torch.randn(1, 256, device="cuda", dtype=qkv.dtype)
    norm = torch.zeros(64, device="cuda", dtype=qkv.dtype)
    inorm = torch.zeros(128, device="cuda", dtype=qkv.dtype)
    slot = torch.tensor([(pages - 1) * 16], device="cuda", dtype=torch.int32)
    position = torch.tensor([37], device="cuda", dtype=torch.int64)
    bounds = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    prepare_qsa(
        qkv,
        index,
        norm,
        norm,
        inorm,
        position,
        norm,
        slot,
        bounds,
        keys,
        values,
        cache,
        rope.cos_sin_cache,
        heads=2,
        kv_heads=1,
        index_heads=1,
        head_dim=64,
        index_dim=128,
        rotary_dim=32,
        eps=1e-6,
        page_size=16,
    )
    torch.testing.assert_close(values[-1, 0, 0], qkv[0, -64:], atol=0, rtol=0)
    torch.testing.assert_close(cache[-1, 0, :128], index[0, 128:], atol=0, rtol=0)
    expected = torch.nn.functional.pad(rope.cos_sin_cache[37].to(qkv.dtype), (0, 96))
    torch.testing.assert_close(cache[-1, 0, 256:], expected, atol=0, rtol=0)


@pytest.mark.parametrize("rows", [1, 32, 128])
@pytest.mark.parametrize("mode", ["text", "section", "interleaved"])
@torch.inference_mode()
def test_fused_qsa_norm_rope_paged_publication_and_replay(rows, mode):
    torch.manual_seed(71)
    heads, kv_heads, dim, index_heads, idim, rot = 6, 1, 256, 4, 128, 64
    dtype = torch.bfloat16
    rope = (
        RotaryEmbedding(dim, rot, 2048, 10000, True)
        if mode == "text"
        else MRotaryEmbedding(
            dim, rot, 512, 10000, True, [16, 8, 8], mode == "interleaved"
        )
    )
    if mode == "interleaved":
        axes = rope.interleaved_axis[: rot // 2]
    else:
        axes = torch.tensor(
            [0] * 16 + [1] * 8 + [2] * 8, device="cuda", dtype=torch.int32
        )
    norms = [GemmaRMSNorm(d, 1e-6).to(dtype) for d in (dim, dim, idim)]
    for norm in norms:
        norm.weight.normal_(std=0.2)
    qkv = torch.randn(
        rows, (heads * 2 + kv_heads * 2) * dim, device="cuda", dtype=dtype
    )
    index = torch.randn(rows, (index_heads + 1) * idim, device="cuda", dtype=dtype)
    positions = torch.arange(rows, device="cuda", dtype=torch.int64) + 37
    if mode != "text":
        positions = torch.stack((positions, positions + 19, positions + 43))
    slots = torch.randperm(256, device="cuda")[:rows]
    bounds = torch.tensor(
        [0, rows] if rows == 1 else [0, 3, rows], device="cuda", dtype=torch.int32
    )
    keys = torch.zeros(16, 16, kv_heads, dim, device="cuda", dtype=dtype)
    values = torch.zeros_like(keys)
    cache = torch.zeros(16, 16, 384, device="cuda", dtype=dtype)

    def fused():
        return prepare_qsa(
            qkv,
            index,
            *(n.weight for n in norms),
            positions,
            axes,
            slots,
            bounds,
            keys,
            values,
            cache,
            rope.cos_sin_cache,
            heads=heads,
            kv_heads=kv_heads,
            index_heads=index_heads,
            head_dim=dim,
            index_dim=idim,
            rotary_dim=rot,
            eps=1e-6,
            page_size=16,
        )

    def check(actual):
        qg, k, v = qkv.split([heads * 2 * dim, kv_heads * dim, kv_heads * dim], -1)
        q = qg.reshape(rows, heads, dim * 2)[..., :dim]
        q = norms[0](q).reshape(rows, -1)
        k = norms[1](k.reshape(rows, kv_heads, dim)).reshape(rows, -1)
        q, k = rope(positions, q, k)
        iq, raw = index.split([index_heads * idim, idim], -1)
        iq = apply_indexer_rope(
            norms[2](iq.reshape(rows, index_heads, idim)), positions, rope
        )
        entry = F.pad(
            torch.cat(
                (
                    raw,
                    torch.zeros_like(raw),
                    indexer_rotary_cache(positions, rope).to(dtype),
                ),
                -1,
            ),
            (0, 64),
        )
        pages, offsets = slots // 16, slots % 16
        torch.testing.assert_close(
            actual[0], q.reshape(rows, heads, dim), atol=0.016, rtol=0.01
        )
        torch.testing.assert_close(actual[1], iq, atol=0.016, rtol=0.01)
        torch.testing.assert_close(
            keys[pages, offsets], k.reshape(rows, kv_heads, dim), atol=0.016, rtol=0.01
        )
        torch.testing.assert_close(
            values[pages, offsets], v.reshape(rows, kv_heads, dim), atol=0, rtol=0
        )
        torch.testing.assert_close(cache[pages, offsets], entry, atol=0, rtol=0)
        torch.testing.assert_close(
            actual[2].long(),
            torch.bucketize(torch.arange(rows, device="cuda"), bounds[1:], right=True),
        )
        untouched = torch.ones(256, device="cuda", dtype=torch.bool)
        untouched[slots] = False
        assert not keys.flatten(0, 1)[untouched].count_nonzero()
        assert not cache.flatten(0, 1)[untouched].count_nonzero()

    for _ in range(3):
        actual = fused()
    check(actual)
    if rows == 32:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = fused()
        qkv.normal_()
        index.normal_()
        positions.add_(11)
        slots.copy_(slots.roll(1))
        bounds[1] = 9
        graph.replay()
        check(actual)
