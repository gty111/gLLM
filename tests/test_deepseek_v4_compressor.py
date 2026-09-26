import pytest
import torch

from gllm.layers.attention.deepseek_v4.compressor import (
    CompressorState,
    compress_decode_batch,
    make_compressor_state,
)


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
