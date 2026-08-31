import pytest
import torch

from gllm.layers.deepseek_v4_mhc import hc_split_sinkhorn, mhc_head, mhc_post, mhc_pre


def _official_split_reference(mixes, scale, base, hc=4, iterations=20, eps=1e-6):
    pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(
        mixes[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc]
    )
    comb = (
        mixes[..., 2 * hc :] * scale[2] + base[2 * hc :]
    ).reshape(*mixes.shape[:-1], hc, hc)
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iterations - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def test_hc_split_sinkhorn_matches_official_formula():
    torch.manual_seed(23)
    mixes = torch.randn(2, 3, 24, dtype=torch.float32)
    scale = torch.randn(3, dtype=torch.float32)
    base = torch.randn(24, dtype=torch.float32)
    actual = hc_split_sinkhorn(mixes, scale, base)
    expected = _official_split_reference(mixes, scale, base)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_mhc_pre_and_post_match_official_formula():
    torch.manual_seed(29)
    x = torch.randn(2, 3, 4, 16, dtype=torch.bfloat16)
    fn = torch.randn(24, 64, dtype=torch.float32)
    scale = torch.randn(3, dtype=torch.float32)
    base = torch.randn(24, dtype=torch.float32)

    layer_input, post, comb = mhc_pre(
        x, fn, scale, base, norm_eps=1e-6
    )
    x_fp32 = x.flatten(2).float()
    mixes = torch.nn.functional.linear(x_fp32, fn) * torch.rsqrt(
        x_fp32.square().mean(-1, keepdim=True) + 1e-6
    )
    pre_ref, post_ref, comb_ref = _official_split_reference(
        mixes, scale, base
    )
    input_ref = torch.sum(pre_ref.unsqueeze(-1) * x.float(), dim=2).to(x.dtype)
    torch.testing.assert_close(layer_input, input_ref, rtol=0, atol=0)
    torch.testing.assert_close(post, post_ref, rtol=0, atol=0)
    torch.testing.assert_close(comb, comb_ref, rtol=0, atol=0)

    transformed = torch.randn(2, 3, 16, dtype=torch.bfloat16)
    actual = mhc_post(transformed, x, post, comb)
    expected = (
        post.unsqueeze(-1) * transformed.unsqueeze(-2)
        + torch.sum(comb.unsqueeze(-1) * x.unsqueeze(-2), dim=2)
    ).to(transformed.dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_mhc_head_matches_official_formula():
    torch.manual_seed(13)
    x = torch.randn(2, 3, 4, 16, dtype=torch.bfloat16)
    fn = torch.randn(4, 64, dtype=torch.float32)
    scale = torch.randn(1, dtype=torch.float32)
    base = torch.randn(4, dtype=torch.float32)

    flat = x.flatten(-2).float()
    inv_rms = torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)
    mixes = torch.nn.functional.linear(flat, fn) * inv_rms
    weights = torch.sigmoid(mixes * scale + base) + 1e-6
    expected = (weights.unsqueeze(-1) * flat.view(x.shape)).sum(-2).to(x.dtype)
    actual = mhc_head(
        x,
        fn,
        scale,
        base,
        norm_eps=1e-6,
        hc_mult=4,
        hc_eps=1e-6,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("tokens", [1, 7, 512])
def test_fused_sinkhorn_matches_the_reference_within_fp32_rounding(tokens):
    """The fused gate is one launch instead of ~170, at a different summation
    order. It is not bitwise identical to the elementwise reference and is not
    meant to be; it must stay within fp32 rounding of it."""
    from gllm.layers.deepseek_v4_mhc import hc_split_sinkhorn_reference
    from gllm.layers.ops.deepseek_v4.mhc import hc_split_sinkhorn_fused

    hc_mult, iters, eps = 4, 20, 1e-6
    torch.manual_seed(17 + tokens)
    mixes = torch.randn(
        tokens, (2 + hc_mult) * hc_mult, device="cuda", dtype=torch.float32
    ) * 2.0
    scale = torch.randn(3, device="cuda") * 0.5
    base = torch.randn((2 + hc_mult) * hc_mult, device="cuda") * 0.5

    kwargs = dict(hc_mult=hc_mult, sinkhorn_iters=iters, eps=eps)
    expected = hc_split_sinkhorn_reference(mixes, scale, base, **kwargs)
    actual = hc_split_sinkhorn_fused(mixes, scale, base, **kwargs)

    for want, got in zip(expected, actual):
        assert got.shape == want.shape
        assert got.dtype is torch.float32
        torch.testing.assert_close(got, want, rtol=2e-5, atol=2e-6)

    # Sinkhorn's fixed point: both marginals of ``comb`` are normalized, and the
    # fused form must land on it just as squarely as the reference.
    comb = actual[2]
    torch.testing.assert_close(
        comb.sum(dim=-1), torch.ones_like(comb.sum(dim=-1)), rtol=2e-3, atol=2e-3
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sinkhorn_dispatch_prefers_the_fused_kernel_on_cuda():
    from gllm.layers.deepseek_v4_mhc import (
        hc_split_sinkhorn,
        hc_split_sinkhorn_reference,
    )
    from gllm.layers.ops.deepseek_v4.mhc import hc_split_sinkhorn_fused

    torch.manual_seed(3)
    mixes = torch.randn(4, 24, device="cuda", dtype=torch.float32)
    scale = torch.randn(3, device="cuda")
    base = torch.randn(24, device="cuda")

    dispatched = hc_split_sinkhorn(mixes, scale, base)
    fused = hc_split_sinkhorn_fused(
        mixes, scale, base, hc_mult=4, sinkhorn_iters=20, eps=1e-6
    )
    for a, b in zip(dispatched, fused):
        assert torch.equal(a, b)

    # A CPU tensor has no fused path and must take the reference unchanged.
    cpu = [t.cpu() for t in (mixes, scale, base)]
    for a, b in zip(
        hc_split_sinkhorn(*cpu), hc_split_sinkhorn_reference(*cpu)
    ):
        assert torch.equal(a, b)


# --- fused mhc_pre / mhc_post -------------------------------------------
#
# Both dispatch to Triton on CUDA. The reference stays reachable by disabling
# the guard, and is the oracle here.


def _reference_pre(x, hc_fn, hc_scale, hc_base, *, norm_eps, hc_mult):
    from unittest import mock

    import gllm.layers.deepseek_v4_mhc as mod

    with mock.patch.object(mod, "_fused_mhc_usable", lambda *_: False):
        return mhc_pre(
            x, hc_fn, hc_scale, hc_base, norm_eps=norm_eps, hc_mult=hc_mult
        )


def _reference_post(x, residual, post, comb):
    """Spelled out so the contraction's orientation is pinned explicitly."""
    return (
        post.unsqueeze(-1) * x.unsqueeze(-2)
        + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=-3)
    ).to(x.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("hidden", [256, 4096])
@pytest.mark.parametrize("tokens", [1, 3, 16])
def test_fused_mhc_pre_matches_reference(hidden, tokens):
    torch.manual_seed(hidden + tokens)
    hc = 4
    width = hc * hidden
    x = torch.randn(tokens, 1, hc, hidden, device="cuda", dtype=torch.bfloat16)
    hc_fn = torch.randn(
        (2 + hc) * hc, width, device="cuda", dtype=torch.float32
    ) / width**0.5
    hc_scale = torch.randn(3, device="cuda", dtype=torch.float32)
    hc_base = torch.randn((2 + hc) * hc, device="cuda", dtype=torch.float32)

    got = mhc_pre(x, hc_fn, hc_scale, hc_base, norm_eps=1e-6, hc_mult=hc)
    want = _reference_pre(
        x, hc_fn, hc_scale, hc_base, norm_eps=1e-6, hc_mult=hc
    )

    # layer_input is bf16, so one ULP; the fp32 gates track much tighter.
    torch.testing.assert_close(got[0], want[0], rtol=8e-3, atol=8e-3)
    torch.testing.assert_close(got[1], want[1], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(got[2], want[2], rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("hidden", [256, 4096])
@pytest.mark.parametrize("tokens", [1, 16])
def test_fused_mhc_post_matches_reference(hidden, tokens):
    torch.manual_seed(hidden * 2 + tokens)
    hc = 4
    x = torch.randn(tokens, 1, hidden, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(
        tokens, 1, hc, hidden, device="cuda", dtype=torch.bfloat16
    )
    post = torch.rand(tokens, 1, hc, device="cuda", dtype=torch.float32)
    comb = torch.rand(tokens, 1, hc, hc, device="cuda", dtype=torch.float32)

    got = mhc_post(x, residual, post, comb)
    want = _reference_post(x, residual, post, comb)
    torch.testing.assert_close(got, want, rtol=8e-3, atol=8e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_mhc_post_contracts_combs_first_axis():
    """A symmetric ``comb`` hides a transposed contraction; use a rank-1 one.

    The reference sums over ``dim=-3`` of the broadcast product, which is
    comb's *first* index. Getting this backwards is silent -- the shapes match
    either way.
    """
    hc, hidden = 4, 64
    x = torch.zeros(1, 1, hidden, device="cuda", dtype=torch.bfloat16)
    residual = torch.zeros(
        1, 1, hc, hidden, device="cuda", dtype=torch.bfloat16
    )
    # Only stream 0 carries signal, so the output isolates comb's row 0.
    residual[0, 0, 0, :] = 1.0
    post = torch.zeros(1, 1, hc, device="cuda", dtype=torch.float32)
    comb = torch.zeros(1, 1, hc, hc, device="cuda", dtype=torch.float32)
    comb[0, 0, 0, :] = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")

    got = mhc_post(x, residual, post, comb)
    # out[n] = sum_m comb[m, n] * residual[m] = comb[0, n] * 1
    expected = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
    torch.testing.assert_close(
        got[0, 0, :, 0].float(), expected, rtol=0, atol=0
    )
    torch.testing.assert_close(got, _reference_post(x, residual, post, comb))
