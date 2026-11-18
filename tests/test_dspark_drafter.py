"""DSpark drafter over the target KV segment vs the dense reference path."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _tensor_config():
    from gllm.models.qwen38_dspark import Qwen38DSparkConfig

    return Qwen38DSparkConfig(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        intermediate_size=128, num_hidden_layers=2, rms_norm_eps=1e-6, vocab_size=128,
        block_size=5, target_layer_ids=(0, 1), mask_token_id=127, markov_rank=8,
        rope_theta=10000.0, max_position_embeddings=64, rope_scaling=None,
    )


def _build_model(config):
    """Build on CUDA in BF16, with random stand-ins for the target vocab."""
    from gllm.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
    from gllm.models.qwen38_dspark import Qwen38DSpark

    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        embed = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        with torch.no_grad():
            embed.weight.normal_(0, 0.02)
            lm_head.weight.normal_(0, 0.02)
        model = Qwen38DSpark(config, embed=embed, lm_head=lm_head)
        with torch.no_grad():
            for param in model.parameters():
                param.normal_(0, 0.05)
        return model
    finally:
        torch.set_default_dtype(previous)


def _setup(context_lens, page_size=16, base_layer=3):
    torch.manual_seed(0)
    config = _tensor_config()
    draft = _build_model(config)
    width = draft.noise_width
    # Target geometry: 1 head x (2 * 16) dims = the draft's 2 heads x 16 dims.
    layers = base_layer + config.num_hidden_layers
    pages_per_request = [(n + width + page_size - 1) // page_size for n in context_lens]
    num_pages = sum(pages_per_request) + 4
    shape = (num_pages, layers, page_size, 1, config.num_key_value_heads * config.head_dim)
    keys = torch.zeros(shape, device="cuda", dtype=torch.bfloat16)
    values = torch.zeros_like(keys)
    segment = SimpleNamespace(
        k_cache=[keys[:, i] for i in range(layers)], v_cache=[values[:, i] for i in range(layers)]
    )
    pages = torch.randperm(num_pages - 1)[: sum(pages_per_request)] + 1  # shuffled, avoid page 0
    tables, begin = [], 0
    for count in pages_per_request:
        tables.append(pages[begin:begin + count].tolist())
        begin += count
    return config, draft, segment, tables


def test_paged_drafter_matches_dense_reference():
    context_lens = [37, 20]
    page_size = 16
    config, draft, segment, tables = _setup(context_lens, page_size)
    views = draft.kv_layer_views(segment, base_layer=3)
    taps = [torch.randn(n, config.target_hidden_size, device="cuda", dtype=torch.bfloat16)
            for n in context_lens]

    # Pack both histories (plus a padding row that must be skipped) like a
    # target batch, then write them through the drafter API.
    rows, positions, slots = [], [], []
    for request, (n, table) in enumerate(zip(context_lens, tables)):
        rows.append(taps[request])
        pos = torch.arange(n)
        positions.append(pos)
        slots.append(torch.tensor(table)[pos // page_size] * page_size + pos % page_size)
    rows.append(torch.randn(1, config.target_hidden_size, device="cuda", dtype=torch.bfloat16))
    positions.append(torch.tensor([0]))
    slots.append(torch.tensor([-1]))
    draft.write_context_kv(
        torch.cat(rows), torch.cat(positions).cuda(), torch.cat(slots).cuda(), views
    )

    anchors = torch.tensor([5, 9], device="cuda")
    lens = torch.tensor(context_lens, device="cuda", dtype=torch.int32)
    width = max(len(t) for t in tables)
    block_table = torch.zeros((2, width), dtype=torch.int32)
    for request, table in enumerate(tables):
        block_table[request, : len(table)] = torch.tensor(table)
    hidden = draft.noise_hidden(anchors, lens, block_table.cuda(), views)
    proposals = draft.greedy_proposals(hidden, anchors)

    for request, n in enumerate(context_lens):
        cache = draft.prefill(taps[request][None, : n - 1], anchors[request : request + 1])
        expected, _ = draft.forward_draft(
            taps[request][None, n - 1 :], anchors[request : request + 1],
            start_pos=n, caches=cache,
        )
        torch.testing.assert_close(hidden[request].float(), expected[0].float(), atol=3e-2, rtol=3e-2)
        reference, _, _ = draft.forward_head(hidden[request : request + 1].clone(), anchors[request : request + 1])
        assert proposals[request].tolist() == reference[0, 1:].tolist()


def test_padding_slot_writes_nothing():
    config, draft, segment, tables = _setup([4])
    views = draft.kv_layer_views(segment, base_layer=3)
    before = [k.clone() for k, _ in views]
    draft.write_context_kv(
        torch.randn(2, config.target_hidden_size, device="cuda", dtype=torch.bfloat16),
        torch.tensor([0, 1], device="cuda"), torch.tensor([-1, -1], device="cuda"), views,
    )
    assert all(torch.equal(a, k) for a, (k, _) in zip(before, views))


def _sampler_runner():
    from gllm.speculative.mtp import MtpMixin

    runner = SimpleNamespace(_mtp_tie_overflow=torch.zeros((), device="cuda", dtype=torch.int64))
    for name in ("_mtp_sparse_probs", "_mtp_probs_static", "_gumbel_argmax",
                 "_dspark_sparse_pick", "_dspark_dense_pick"):
        setattr(runner, name, getattr(MtpMixin, name).__get__(runner))
    return runner


def test_propose_with_argmax_matches_greedy_proposals():
    from gllm.speculative.dspark_drafter import DSparkDrafter

    config, draft, segment, tables = _setup([9, 30])
    drafter = DSparkDrafter(draft, kv_base_layer=3, taps=None)
    drafter.bind_kv(segment)
    anchors = torch.tensor([3, 7], device="cuda")
    lens = torch.tensor([9, 30], device="cuda", dtype=torch.int32)
    width = max(len(t) for t in tables)
    block_table = torch.zeros((2, width), dtype=torch.int32)
    for request, table in enumerate(tables):
        block_table[request, : len(table)] = torch.tensor(table)
    block_table = block_table.cuda()
    greedy = drafter.propose(anchors, lens, block_table)
    picked = drafter.propose_with(anchors, lens, block_table, lambda logits: logits.argmax(-1))
    assert torch.equal(greedy, picked)


@pytest.mark.parametrize("sparse", [True, False])
def test_sampled_pick_records_the_distribution_it_draws_from(sparse):
    """Every draw lies in q's support and ``drawn`` is q at the drawn token;
    the empirical draw frequencies match q."""
    torch.manual_seed(0)
    runner = _sampler_runner()
    vocab, rows = 128, 4096
    logits = torch.randn(1, vocab, device="cuda").repeat(rows, 1) * 3
    temps = torch.full((rows, 1), 0.8, device="cuda")
    top_ks = torch.full((rows,), 8, dtype=torch.int32, device="cuda")
    top_ps = torch.full((rows,), 0.9, device="cuda")
    record = []
    if sparse:
        pick = runner._dspark_sparse_pick(temps, top_ks, top_ps, 72, record)
    else:
        dense = torch.empty((rows, 1, vocab), device="cuda")
        pick = runner._dspark_dense_pick(temps, top_ks, top_ps, dense)
    tokens = pick(logits)
    if sparse:
        vals, idx, drawn = record[0]
        column = (idx == tokens[:, None]).float().argmax(1)
        assert torch.equal(idx.gather(1, column[:, None]).squeeze(1), tokens)
        torch.testing.assert_close(drawn, vals.gather(1, column[:, None]).squeeze(1))
        q = torch.zeros(vocab, device="cuda").scatter_(0, idx[0], vals[0])
    else:
        q = dense[0, 0]
    assert (q[tokens] > 0).all()
    empirical = torch.bincount(tokens, minlength=vocab).float() / rows
    assert (empirical - q).abs().max() < 0.03


def _tap_target(hidden_size, num_layers=3):
    """Tiny target backbone (real ``Qwen3_5Model._forward``) with dense layers."""
    from gllm.models.qwen3_5 import Qwen3_5Model

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)

        def forward(self, data, hidden, residual):
            residual = hidden.clone() if residual is None else residual + hidden
            return torch.tanh(self.proj(residual)), residual

    class Norm(torch.nn.Module):
        def forward(self, hidden, residual):
            return hidden + residual, residual

    model = Qwen3_5Model.__new__(Qwen3_5Model)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=hidden_size)
    model.start_layer, model.end_layer = 0, num_layers
    model.embed_tokens = torch.nn.Embedding(128, hidden_size)
    model.layers = torch.nn.ModuleList(Layer() for _ in range(num_layers))
    model.norm = Norm()
    model._fuse_tail = False
    return model.cuda().bfloat16()


@torch.no_grad()
def test_graph_replays_refresh_tap_sink_and_draft_kv():
    """A captured target forward + ``kv_sync`` must follow each replay's inputs.

    The sink and the draft K/V are written at fixed addresses inside the
    graph; every replay with new tokens, positions and slots must leave
    exactly what an eager run on those inputs leaves.
    """
    from gllm.speculative.dspark_drafter import DSparkDrafter

    rows, page_size = 6, 16
    config, draft, segment, tables = _setup([3 * page_size], page_size)
    target = _tap_target(config.hidden_size)
    taps = torch.zeros(16, config.target_hidden_size, device="cuda", dtype=torch.bfloat16)
    target.set_hidden_tap_sink(config.target_layer_ids, taps)
    drafter = DSparkDrafter(draft, kv_base_layer=3, taps=taps)
    drafter.bind_kv(segment)

    tokens = torch.zeros(rows, dtype=torch.long, device="cuda")
    positions = torch.zeros(rows, dtype=torch.long, device="cuda")
    slots = torch.zeros(rows, dtype=torch.long, device="cuda")
    data = SimpleNamespace(
        seqs=[SimpleNamespace(mm_contents=None)], get_tokens=lambda: tokens,
        get_position=lambda: positions, get_slot_mapping=lambda: slots,
    )

    def step():
        target(data)
        drafter.kv_sync(data, rows)

    table = torch.tensor(tables[0])
    gen = torch.Generator().manual_seed(1)

    def load(start):
        """New tokens for rows at positions ``start..start+rows``."""
        tokens.copy_(torch.randint(0, 128, (rows,), generator=gen))
        pos = torch.arange(start, start + rows)
        positions.copy_(pos)
        slots.copy_(table[pos // page_size] * page_size + pos % page_size)

    def snapshot():
        return taps[:rows].clone(), [(k.clone(), v.clone()) for k, v in drafter.kv_views]

    load(0)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step()  # warm up outside capture
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()

    previous = None
    for start in (0, 10, 20):  # the second window crosses a page boundary
        load(start)
        state = (tokens.clone(), positions.clone(), slots.clone())
        graph.replay()
        torch.cuda.synchronize()
        replayed = snapshot()

        # Eager rerun of the same inputs from a clean slate.
        taps.zero_()
        for k, v in drafter.kv_views:
            k.zero_()
            v.zero_()
        if previous is not None:  # restore earlier windows the eager run does not rewrite
            for (k, v), (pk, pv) in zip(drafter.kv_views, previous[1]):
                k.copy_(pk)
                v.copy_(pv)
        for dst, src in zip((tokens, positions, slots), state):
            dst.copy_(src)
        step()
        eager = snapshot()

        torch.testing.assert_close(replayed[0], eager[0], rtol=0, atol=0)
        for (rk, rv), (ek, ev) in zip(replayed[1], eager[1]):
            torch.testing.assert_close(rk, ek, rtol=0, atol=0)
            torch.testing.assert_close(rv, ev, rtol=0, atol=0)
        if previous is not None:
            assert not torch.equal(replayed[0], previous[0])  # the sink did move
        written = slots // page_size, slots % page_size
        assert all(k[written].abs().sum() > 0 for k, _ in drafter.kv_views)
        previous = replayed
