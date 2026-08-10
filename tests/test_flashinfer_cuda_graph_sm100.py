"""SM100 FlashInfer paged-attention CUDA Graph correctness probe.

Run directly with the gLLM environment.  It checks graph replay after changing
Q/K/V, sequence lengths, query offsets, and page tables, and compares against a
plain PyTorch causal GQA reference.
"""

import math

import torch

from gllm.runtime.forward_metadata import ForwardMetadataPlan
from gllm.layers.attention.qkv_backends import FlashInferAttentionBackend
from gllm.layers.ops.cache_kernels import reshape_and_cache_flash


class _Input:
    def __init__(self, batch, max_pages, qlen=4):
        self.max_query_len = qlen
        self.num_decodes = batch if qlen == 1 else 0
        self.is_mtp_verify = qlen > 1
        self.seq_lens = torch.empty(batch, dtype=torch.int32, device="cuda")
        self.qsl = torch.arange(
            0, (batch + 1) * self.max_query_len, self.max_query_len,
            dtype=torch.int32, device="cuda",
        )
        self.block_table = torch.empty(
            batch, max_pages, dtype=torch.int32, device="cuda"
        )
        self.forward_metadata_plan = ForwardMetadataPlan.uniform_gpu(
            num_rows=batch,
            qlen=qlen,
            is_mtp_verify=qlen > 1,
        )

    def get_seq_lens(self):
        return self.seq_lens

    def get_query_start_loc(self):
        return self.qsl

    def get_block_table(self):
        return self.block_table

    def prepare_metadata(self, backend):
        return backend.prepare_metadata(self, self.forward_metadata_plan)


def _reference(q, kc, vc, pages, seq_lens, qlen, scale):
    rows = []
    page_size = kc.shape[1]
    repeat = q.shape[1] // kc.shape[2]
    for i, length in enumerate(seq_lens):
        ids = torch.as_tensor(
            pages[i][: math.ceil(length / page_size)],
            dtype=torch.long,
            device=q.device,
        )
        k = kc[ids].reshape(-1, kc.shape[2], kc.shape[3])[:length]
        v = vc[ids].reshape(-1, vc.shape[2], vc.shape[3])[:length]
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
        qi = q[i * qlen : (i + 1) * qlen]
        score = torch.einsum("qhd,khd->qhk", qi.float(), k.float()) * scale
        context = length - qlen
        key_pos = torch.arange(length, device=q.device)
        query_pos = context + torch.arange(qlen, device=q.device)
        score.masked_fill_(key_pos[None, None, :] > query_pos[:, None, None], -torch.inf)
        prob = torch.softmax(score, dim=-1)
        rows.append(torch.einsum("qhk,khd->qhd", prob, v.float()).to(q.dtype))
    return torch.cat(rows)


def main():
    torch.manual_seed(17)
    batch, qlen, page_size = 4, 4, 16
    q_heads, kv_heads, head_dim = 16, 4, 128
    num_pages, max_pages = 40, 4
    scale = head_dim**-0.5
    inp = _Input(batch, max_pages)
    backend = FlashInferAttentionBackend(64, batch)
    q = torch.empty(batch * qlen, q_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    kc = torch.empty(num_pages, page_size, kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    vc = torch.empty_like(kc)

    # Warm up every lazy FlashInfer allocation before capture.
    inp.seq_lens.copy_(torch.tensor([20, 31, 36, 48], dtype=torch.int32, device="cuda"))
    inp.block_table.copy_(torch.arange(batch * max_pages, dtype=torch.int32, device="cuda").view(batch, max_pages))
    q.normal_(); kc.normal_(); vc.normal_()
    for _ in range(11):
        backend.forward(q, kc, vc, inp.prepare_metadata(backend), scale)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        # Qwen3.5-35B has 11 ordinary attention layers sharing one backend and
        # workspace. Repeated calls catch workspace/PDL ordering problems that
        # a single isolated invocation cannot expose.
        for _ in range(11):
            meta = inp.prepare_metadata(backend)
            graph_out = backend.forward(q, kc, vc, meta, scale)

    cases = [
        ([19, 22, 37, 47], [[20, 3, 11, 8], [5, 17, 1, 29], [6, 31, 2, 15], [7, 18, 32, 9]]),
        ([32, 21, 46, 18], [[33, 4, 21, 0], [19, 8, 35, 2], [12, 30, 22, 1], [39, 10, 5, 6]]),
    ]
    for case_id, (lens, table) in enumerate(cases):
        q.normal_(); kc.normal_(); vc.normal_()
        inp.seq_lens.copy_(torch.tensor(lens, dtype=torch.int32, device="cuda"))
        inp.block_table.copy_(torch.tensor(table, dtype=torch.int32, device="cuda"))
        graph.replay()
        torch.cuda.synchronize()
        ref = _reference(q, kc, vc, table, lens, qlen, scale)
        diff = (graph_out - ref).abs().float()
        print(
            f"case={case_id} max_abs={diff.max().item():.6g} "
            f"mean_abs={diff.mean().item():.6g} allclose={torch.allclose(graph_out, ref, atol=2e-2, rtol=2e-2)}"
        )
        assert torch.allclose(graph_out, ref, atol=2e-2, rtol=2e-2)

    # Capture an ordinary qlen=1 decode graph after the qlen=4 MTP decode graph
    # using the same backend,
    # workspace, and cumulative-length buffer. Alternate replays to catch stale
    # shared state between the ordinary-decode and MTP-verify graph buckets.
    inp1 = _Input(batch, max_pages, qlen=1)
    q1 = torch.randn(batch, q_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    lens1 = [17, 25, 33, 41]
    table1 = [[1, 7, 8, 9], [2, 10, 11, 12], [3, 13, 14, 15], [4, 16, 17, 18]]
    inp1.seq_lens.copy_(torch.tensor(lens1, dtype=torch.int32, device="cuda"))
    inp1.block_table.copy_(torch.tensor(table1, dtype=torch.int32, device="cuda"))
    backend.forward(q1, kc, vc, inp1.prepare_metadata(backend), scale)
    torch.cuda.synchronize()
    decode_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(decode_graph):
        decode_out = backend.forward(
            q1, kc, vc, inp1.prepare_metadata(backend), scale
        )
    for order in range(2):
        decode_graph.replay()
        graph.replay()
        decode_graph.replay()
        torch.cuda.synchronize()
        ref1 = _reference(q1, kc, vc, table1, lens1, 1, scale)
        diff = (decode_out - ref1).abs().float()
        print(
            f"alternate={order} qlen=1 max_abs={diff.max().item():.6g} "
            f"allclose={torch.allclose(decode_out, ref1, atol=2e-2, rtol=2e-2)}"
        )
        assert torch.allclose(decode_out, ref1, atol=2e-2, rtol=2e-2)

    # Real model order: a Triton producer writes this forward's K/V rows, then
    # FlashInfer (PDL enabled) consumes those slots in the same captured graph.
    # This catches an otherwise invisible producer/consumer ordering failure.
    lensw = [20, 31, 36, 48]
    tablew = [[20, 3, 11, 8], [5, 17, 1, 29], [6, 31, 2, 15], [7, 18, 32, 9]]
    inp.seq_lens.copy_(torch.tensor(lensw, dtype=torch.int32, device="cuda"))
    inp.block_table.copy_(torch.tensor(tablew, dtype=torch.int32, device="cuda"))
    knew = torch.empty(batch * qlen, kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    vnew = torch.empty_like(knew)
    slots = []
    for i, length in enumerate(lensw):
        for pos in range(length - qlen, length):
            slots.append(tablew[i][pos // page_size] * page_size + pos % page_size)
    slots = torch.tensor(slots, dtype=torch.int64, device="cuda")
    one = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    q.normal_(); knew.normal_(); vnew.normal_()
    reshape_and_cache_flash(knew, vnew, kc, vc, slots, "auto", one, one)
    backend.forward(q, kc, vc, inp.prepare_metadata(backend), scale)
    torch.cuda.synchronize()
    rw_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(rw_graph):
        reshape_and_cache_flash(knew, vnew, kc, vc, slots, "auto", one, one)
        rw_out = backend.forward(q, kc, vc, inp.prepare_metadata(backend), scale)
    for replay in range(3):
        q.normal_(); knew.normal_(); vnew.normal_()
        # Poison the destination slots first, ensuring success cannot come from
        # values left by capture or the previous replay.
        kc.view(-1, kv_heads, head_dim).index_fill_(0, slots, 0)
        vc.view(-1, kv_heads, head_dim).index_fill_(0, slots, 0)
        rw_graph.replay()
        torch.cuda.synchronize()
        rw_ref = _reference(q, kc, vc, tablew, lensw, qlen, scale)
        diff = (rw_out - rw_ref).abs().float()
        print(
            f"write-read={replay} max_abs={diff.max().item():.6g} "
            f"allclose={torch.allclose(rw_out, rw_ref, atol=2e-2, rtol=2e-2)}"
        )
        assert torch.allclose(rw_out, rw_ref, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    main()
