"""The fused FutureMap resolve must match the expression it replaced.

``resolve_future`` used to be a ``torch.where`` over a ``clamp``/gather; it is
now a single Triton kernel, so that expression lives here as the reference.
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a GPU"
)


def _reference(ids: torch.Tensor, buf: torch.Tensor) -> torch.Tensor:
    return torch.where(ids < 0, buf[torch.clamp(-ids, min=0)], ids)


@pytest.mark.parametrize("n", [1, 7, 16, 33, 256, 257, 1024, 4096])
def test_fused_resolve_matches_reference(n):
    from gllm.runtime.async_runtime import resolve_future_inplace

    torch.manual_seed(n)
    buf = torch.randint(0, 150_000, (4096,), dtype=torch.int64, device="cuda")
    ids = torch.randint(0, 150_000, (n,), dtype=torch.int64, device="cuda")
    slots = torch.randint(0, buf.numel(), (n,), device="cuda")
    ids = torch.where(torch.rand(n, device="cuda") < 0.6, -slots, ids)
    if n >= 2:
        # Slot 0 encodes as 0, which is *not* a placeholder, and the last slot
        # is the only index that can run off the end of the buffer.
        ids[0] = 0
        ids[1] = -(buf.numel() - 1)
    want = _reference(ids, buf)
    got = ids.clone()
    resolve_future_inplace(got, buf)
    assert torch.equal(got, want)


def test_empty_batch_is_a_noop():
    from gllm.runtime.async_runtime import resolve_future_inplace

    buf = torch.zeros(8, dtype=torch.int64, device="cuda")
    ids = torch.zeros(0, dtype=torch.int64, device="cuda")
    resolve_future_inplace(ids, buf)
    assert ids.numel() == 0
