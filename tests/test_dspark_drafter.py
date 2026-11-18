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
