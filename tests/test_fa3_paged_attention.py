"""Run on an SM80/SM89/SM90 GPU with the installed SGL kernel wheel."""

from types import SimpleNamespace

import pytest
import torch

from gllm.layers.attention.qkv_backends import FA3AttentionBackend
from gllm.layers.ops.flash_attn_compat import flash_attn_varlen_func

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (8, 9),
    reason="SGL kernel FlashAttention-3 needs an SM8x or SM90 GPU",
)


def _case(query_lens, dtype, head_dim, kv_heads, page_size):
    torch.manual_seed(123)
    batch = len(query_lens)
    pages = max(4, (max(query_lens) + page_size - 1) // page_size)
    q = torch.randn(sum(query_lens), 8, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch * pages, page_size, kv_heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    table = torch.randperm(batch * pages, device="cuda", dtype=torch.int32).view(batch, pages)
    # Two requests share a cached prefix page, as in prefix-cache reuse.
    if batch > 1:
        table[1, 0] = table[0, 0]
    lengths = torch.tensor([max(n, page_size + 3) for n in query_lens], device="cuda", dtype=torch.int32)
    starts = torch.tensor([0, *query_lens], device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
    data = SimpleNamespace(get_block_table=lambda: table, get_seq_lens=lambda: lengths,
                           get_query_start_loc=lambda: starts)
    plan = SimpleNamespace(max_query_len=max(query_lens), batch_size=batch)
    backend = FA3AttentionBackend(pages * page_size, batch)
    return backend, q, k, v, backend.prepare_metadata(data, plan)


def _reference(q, k, v, metadata):
    outputs = []
    for row in range(metadata.batch_size):
        start, end = metadata.query_start_loc[row:row + 2].tolist()
        length = int(metadata.seq_lens[row])
        keys = k[metadata.block_table[row].long()].flatten(0, 1)[:length]
        values = v[metadata.block_table[row].long()].flatten(0, 1)[:length]
        keys = keys.repeat_interleave(q.shape[1] // k.shape[2], dim=1).float()
        values = values.repeat_interleave(q.shape[1] // v.shape[2], dim=1).float()
        scores = torch.einsum("qhd,khd->hqk", q[start:end].float(), keys) * q.shape[-1] ** -0.5
        mask = torch.arange(length, device=q.device)[None, :] <= (
            length - (end - start) + torch.arange(end - start, device=q.device)[:, None]
        )
        scores.masked_fill_(~mask, float("-inf"))
        outputs.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), values))
    return torch.cat(outputs).to(q.dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim,kv_heads,page_size", [(64, 8, 16), (128, 2, 16), (256, 1, 32)])
@pytest.mark.parametrize("query_lens", [(1, 1), (4, 4), (1, 7, 19), (1, 7, 129)])
def test_fa3_paged_matches_reference(dtype, head_dim, kv_heads, page_size, query_lens):
    backend, q, k, v, metadata = _case(query_lens, dtype, head_dim, kv_heads, page_size)
    out = backend.forward(q, k, v, metadata, head_dim ** -0.5)
    torch.testing.assert_close(out, _reference(q, k, v, metadata), atol=1e-2, rtol=1e-2)


def test_fa3_graph_replay_refreshes_lengths_pages_and_query_boundaries():
    backend, q, k, v, metadata = _case((1, 3), torch.bfloat16, 128, 2, 16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        backend.forward(q, k, v, metadata, 128 ** -0.5)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = backend.forward(q, k, v, metadata, 128 ** -0.5)
    for lengths in ([33, 27], [47, 49]):
        metadata.seq_lens.copy_(torch.tensor(lengths, device=q.device, dtype=torch.int32))
        metadata.query_start_loc.copy_(torch.tensor([0, 2, 4], device=q.device, dtype=torch.int32))
        metadata.block_table.copy_(metadata.block_table.roll(1, dims=1))
        q.normal_()
        graph.replay()
        torch.testing.assert_close(out, _reference(q, k, v, metadata), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("qk_dim,v_dim,causal", [(128, 128, False), (192, 128, True)])
@pytest.mark.parametrize("return_lse", [True, False])
def test_fa3_varlen_adapter_and_lse(qk_dim, v_dim, causal, return_lse):
    q = torch.randn(5, 4, qk_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(7, 4, qk_dim, device="cuda", dtype=q.dtype)
    v = torch.randn(7, 4, v_dim, device="cuda", dtype=q.dtype)
    starts_q = torch.tensor([0, 2, 5], device=q.device, dtype=torch.int32)
    starts_k = torch.tensor([0, 3, 7], device=q.device, dtype=torch.int32)
    result = flash_attn_varlen_func(q, k, v, starts_q, starts_k, backend="fa3",
                                    max_seqlen_q=3, max_seqlen_k=4, causal=causal,
                                    return_softmax_lse=return_lse)
    out = result[0] if return_lse else result
    refs, lses = [], []
    for qs, qe, ks, ke in [(0, 2, 0, 3), (2, 5, 3, 7)]:
        scores = torch.einsum("qhd,khd->hqk", q[qs:qe].float(), k[ks:ke].float()) * qk_dim ** -0.5
        if causal:
            mask = torch.ones(qe - qs, ke - ks, device=q.device, dtype=torch.bool).tril((ke - ks) - (qe - qs))
            scores.masked_fill_(~mask, float("-inf"))
        lses.append(scores.logsumexp(-1))
        refs.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), v[ks:ke].float()))
    torch.testing.assert_close(out, torch.cat(refs).to(q.dtype), atol=1e-2, rtol=1e-2)
    if return_lse:
        torch.testing.assert_close(result[1], torch.cat(lses, dim=1), atol=1e-2, rtol=1e-2)
