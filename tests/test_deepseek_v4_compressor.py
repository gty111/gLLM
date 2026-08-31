import pytest
import torch

from gllm.layers.attention.deepseek_v4.compressor import (
    CompressorState,
    DeepseekV4Compressor,
    compress_decode,
    compress_decode_batch,
    compress_prefill,
    compress_prefill_batch,
    compress_prefill_continue_batch,
    make_compressor_state,
)
from gllm.layers.attention.deepseek_v4.ops import precompute_rope_frequencies


@pytest.mark.parametrize("ratio,head_dim,length", [(4, 8, 12), (128, 4, 256)])
def test_sequential_compressor_matches_full_prefill(ratio, head_dim, length):
    torch.manual_seed(41 + ratio)
    overlap = ratio == 4
    channels = (1 + overlap) * head_dim
    kv = torch.randn(2, length, channels, dtype=torch.float32)
    score = torch.randn_like(kv)
    ape = torch.randn(ratio, channels, dtype=torch.float32)

    prefill, _ = compress_prefill(kv, score, ape, ratio)
    assert prefill is not None
    state = make_compressor_state(2, ratio, head_dim, device="cpu")
    decoded = []
    for position in range(length):
        row = compress_decode(
            kv[:, position : position + 1],
            score[:, position : position + 1],
            ape,
            ratio,
            position,
            state,
        )
        if row is not None:
            decoded.append(row)
    decoded = torch.cat(decoded, dim=1)
    torch.testing.assert_close(decoded, prefill, rtol=1e-6, atol=1e-6)


def test_prefill_remainder_becomes_decode_state():
    torch.manual_seed(47)
    ratio, head_dim = 4, 8
    kv = torch.randn(1, 11, 2 * head_dim)
    score = torch.randn_like(kv)
    ape = torch.randn(ratio, 2 * head_dim)
    prefix, state = compress_prefill(kv[:, :7], score[:, :7], ape, ratio)
    assert prefix is not None and prefix.shape[1] == 1
    continuation = []
    for position in range(7, 11):
        row = compress_decode(
            kv[:, position : position + 1],
            score[:, position : position + 1],
            ape,
            ratio,
            position,
            state,
        )
        if row is not None:
            continuation.append(row)
    combined = torch.cat([prefix, *continuation], dim=1)
    full, _ = compress_prefill(kv, score, ape, ratio)
    torch.testing.assert_close(combined, full, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("ratio,head_dim", [(4, 8), (128, 4)])
def test_heterogeneous_batch_decode_matches_independent_rows(ratio, head_dim):
    torch.manual_seed(79 + ratio)
    positions = torch.tensor([ratio - 2, ratio - 1, ratio + 1], device="cuda")
    channels = (2 if ratio == 4 else 1) * head_dim
    ape = torch.randn(ratio, channels, device="cuda")
    batch_state = make_compressor_state(3, ratio, head_dim, device="cuda")
    row_states = [make_compressor_state(1, ratio, head_dim, device="cuda") for _ in range(3)]

    # Seed every state by independently consuming its prefix.
    for row, position in enumerate(positions.tolist()):
        for prefix_position in range(position):
            kv = torch.randn(1, 1, channels, device="cuda")
            score = torch.randn_like(kv)
            compress_decode(
                kv, score, ape, ratio, prefix_position, row_states[row]
            )
        batch_state.kv[row].copy_(row_states[row].kv[0])
        batch_state.score[row].copy_(row_states[row].score[0])

    kv = torch.randn(3, 1, channels, device="cuda")
    score = torch.randn_like(kv)
    actual, boundaries = compress_decode_batch(
        kv, score, ape, ratio, positions, batch_state
    )
    for row, position in enumerate(positions.tolist()):
        expected = compress_decode(
            kv[row : row + 1],
            score[row : row + 1],
            ape,
            ratio,
            position,
            row_states[row],
        )
        assert boundaries[row].item() == (expected is not None)
        if expected is not None:
            torch.testing.assert_close(actual[row : row + 1], expected)
        torch.testing.assert_close(batch_state.kv[row], row_states[row].kv[0])
        torch.testing.assert_close(batch_state.score[row], row_states[row].score[0])


@pytest.mark.parametrize("ratio,head_dim,lengths", [(4, 8, [6, 12, 9]), (128, 4, [127, 256, 131])])
def test_variable_length_batch_prefill_matches_independent_rows(
    ratio, head_dim, lengths
):
    torch.manual_seed(137 + ratio)
    channels = (2 if ratio == 4 else 1) * head_dim
    max_length = max(lengths)
    kv = torch.randn(len(lengths), max_length, channels)
    score = torch.randn_like(kv)
    ape = torch.randn(ratio, channels)
    actual, state, counts = compress_prefill_batch(
        kv, score, ape, ratio, torch.tensor(lengths)
    )

    for row, length in enumerate(lengths):
        expected, expected_state = compress_prefill(
            kv[row : row + 1, :length],
            score[row : row + 1, :length],
            ape,
            ratio,
        )
        count = length // ratio
        assert counts[row].item() == count
        if count:
            torch.testing.assert_close(
                actual[row : row + 1, :count], expected
            )
        torch.testing.assert_close(state.kv[row], expected_state.kv[0])
        torch.testing.assert_close(state.score[row], expected_state.score[0])


@pytest.mark.parametrize(
    "ratio,head_dim,starts,lengths",
    [
        (4, 8, [3, 8, 6], [7, 5, 2]),
        (128, 4, [127, 128, 131], [4, 130, 5]),
    ],
)
def test_variable_length_continuation_prefill_matches_token_updates(
    ratio, head_dim, starts, lengths
):
    torch.manual_seed(181 + ratio)
    batch = len(starts)
    channels = (2 if ratio == 4 else 1) * head_dim
    max_start = max(starts)
    max_length = max(lengths)
    prefix_kv = torch.randn(batch, max_start, channels)
    prefix_score = torch.randn_like(prefix_kv)
    suffix_kv = torch.randn(batch, max_length, channels)
    suffix_score = torch.randn_like(suffix_kv)
    ape = torch.randn(ratio, channels)

    batch_state = make_compressor_state(batch, ratio, head_dim, device="cpu")
    row_states = []
    for row, start in enumerate(starts):
        state = make_compressor_state(1, ratio, head_dim, device="cpu")
        for position in range(start):
            compress_decode(
                prefix_kv[row : row + 1, position : position + 1],
                prefix_score[row : row + 1, position : position + 1],
                ape,
                ratio,
                position,
                state,
            )
        batch_state.kv[row].copy_(state.kv[0])
        batch_state.score[row].copy_(state.score[0])
        row_states.append(state)

    actual, batch_state, counts = compress_prefill_continue_batch(
        suffix_kv,
        suffix_score,
        ape,
        ratio,
        torch.tensor(starts),
        torch.tensor(lengths),
        batch_state,
    )
    for row, (start, length) in enumerate(zip(starts, lengths, strict=True)):
        expected = []
        for offset in range(length):
            emitted = compress_decode(
                suffix_kv[row : row + 1, offset : offset + 1],
                suffix_score[row : row + 1, offset : offset + 1],
                ape,
                ratio,
                start + offset,
                row_states[row],
            )
            if emitted is not None:
                expected.append(emitted)
        assert counts[row].item() == len(expected)
        if expected:
            torch.testing.assert_close(
                actual[row : row + 1, : len(expected)],
                torch.cat(expected, dim=1),
                rtol=1e-6,
                atol=1e-6,
            )
        torch.testing.assert_close(batch_state.kv[row], row_states[row].kv[0])
        torch.testing.assert_close(batch_state.score[row], row_states[row].score[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compressor_module_prefill_decode_matches_single_prefill():
    torch.manual_seed(43)
    hidden_size, head_dim, rope_dim, ratio = 128, 128, 64, 4
    module = DeepseekV4Compressor(
        hidden_size,
        head_dim,
        rope_dim,
        ratio,
        norm_eps=1e-6,
    )
    module.wkv.weight.data.normal_(std=0.03)
    module.wgate.weight.data.normal_(std=0.03)
    module.ape.data.normal_(std=0.03)
    module.norm_weight.data.uniform_(0.9, 1.1)
    hidden = torch.randn(
        1, 12, hidden_size, device="cuda", dtype=torch.bfloat16
    ) * 0.2
    full_freq = precompute_rope_frequencies(
        rope_dim,
        12,
        original_sequence_length=0,
        base=40000.0,
        factor=1.0,
        beta_fast=32,
        beta_slow=1,
        device="cuda",
    )

    expected, _ = module.prefill(hidden, full_freq[0:12:ratio])
    first, state = module.prefill(hidden[:, :8], full_freq[0:8:ratio])
    emitted = [first]
    for position in range(8, 12):
        out = module.decode(
            hidden[:, position : position + 1],
            full_freq[position + 1 - ratio],
            position=position,
            state=state,
        )
        if out is not None:
            emitted.append(out)
    actual = torch.cat(emitted, dim=1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


# --- fused decode kernel -------------------------------------------------
#
# ``compress_decode_batch`` dispatches to a Triton kernel on CUDA. It must be
# numerically identical to the reference below it, including how it advances
# the rolling state, since a drift there corrupts every later token silently.

_FUSED_CASES = [
    (ratio, head_dim, batch)
    for ratio in (4, 128)
    for head_dim in (64, 576)
    for batch in (1, 5, 16)
]


def _reference_decode_batch(kv, score, ape, ratio, positions, state):
    """Independent plain-PyTorch oracle, kept in the test on purpose.

    ``compress_decode_batch`` is a single Triton kernel in production. Keeping
    the spec spelled out here means the oracle cannot drift along with the
    implementation it checks.
    """
    batch, _, channels = kv.shape
    overlap = ratio == 4
    head_dim = channels // (1 + overlap)

    positions = positions.to(device=kv.device, dtype=torch.long)
    cursor = positions.remainder(ratio)
    rows = torch.arange(batch, device=kv.device)
    score = score[:, 0] + ape.index_select(0, cursor)
    boundary = positions.add(1).remainder(ratio).eq(0)

    if overlap:
        dst = ratio + cursor
        state.kv[rows, dst] = kv[:, 0]
        state.score[rows, dst] = score
        pooled_kv = torch.cat(
            [state.kv[:, :ratio, :head_dim], state.kv[:, ratio:, head_dim:]], dim=1
        )
        pooled_score = torch.cat(
            [state.score[:, :ratio, :head_dim], state.score[:, ratio:, head_dim:]],
            dim=1,
        )
        output = (pooled_kv * pooled_score.softmax(dim=1)).sum(dim=1, keepdim=True)
        update = boundary.view(batch, 1, 1)
        state.kv[:, :ratio].copy_(
            torch.where(update, state.kv[:, ratio:], state.kv[:, :ratio])
        )
        state.score[:, :ratio].copy_(
            torch.where(update, state.score[:, ratio:], state.score[:, :ratio])
        )
        return output, boundary

    state.kv[rows, cursor] = kv[:, 0]
    state.score[rows, cursor] = score
    output = (state.kv * state.score.softmax(dim=1)).sum(dim=1, keepdim=True)
    return output, boundary


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("ratio,head_dim,batch", _FUSED_CASES)
def test_fused_decode_matches_reference(ratio, head_dim, batch):
    torch.manual_seed(ratio * 1000 + head_dim + batch)
    coff = 2 if ratio == 4 else 1
    channels = coff * head_dim

    ref = make_compressor_state(batch, ratio, head_dim, device="cuda")
    ref.kv.normal_()
    ref.score.normal_()
    fused = CompressorState(kv=ref.kv.clone(), score=ref.score.clone())
    ape = torch.randn(ratio, channels, device="cuda", dtype=torch.float32)

    # Walk past a group boundary so the state shift is exercised.
    for _ in range(2 * ratio + 3):
        kv = torch.randn(batch, 1, channels, device="cuda", dtype=torch.float32)
        score = torch.randn(batch, 1, channels, device="cuda", dtype=torch.float32)
        positions = torch.randint(0, 1000, (batch,), device="cuda", dtype=torch.long)

        got, got_boundary = compress_decode_batch(
            kv, score, ape, ratio, positions, fused
        )
        want, want_boundary = _reference_decode_batch(
            kv, score, ape, ratio, positions, ref
        )

        assert torch.equal(got_boundary, want_boundary)
        torch.testing.assert_close(got, want, rtol=0, atol=2e-5)
        # The state is what carries error forward, so check it every step.
        torch.testing.assert_close(fused.kv, ref.kv, rtol=0, atol=2e-5)
        torch.testing.assert_close(
            fused.score.nan_to_num(neginf=-1e30),
            ref.score.nan_to_num(neginf=-1e30),
            rtol=0,
            atol=2e-5,
        )


