import hashlib

import pytest
import torch

from gllm.multimodal.mixin import _hash_tensor_bytes


def _digest_before(t):
    """The digest as computed before bfloat16 support, for dtypes numpy knows."""
    h = hashlib.sha256()
    h.update(str(t.dtype).encode())
    h.update(repr(tuple(t.shape)).encode())
    h.update(memoryview(t.numpy().tobytes()))
    return h.digest()


@pytest.mark.parametrize("t", [torch.randn(4, 6), torch.tensor([1, 2, 4]),
                               torch.randn(3, 8).half()])
def test_digest_is_unchanged_for_numpy_dtypes(t):
    assert _hash_tensor_bytes(t) == _digest_before(t)


def test_bfloat16_pixels_can_be_hashed():
    pixels = torch.randn(4, 6)
    bf16 = pixels.bfloat16()
    assert _hash_tensor_bytes(bf16) == _hash_tensor_bytes(bf16.clone())
    assert _hash_tensor_bytes(bf16) != _hash_tensor_bytes(pixels)
    assert _hash_tensor_bytes(bf16) != _hash_tensor_bytes((pixels + 1).bfloat16())


def test_non_contiguous_input_hashes_like_its_contiguous_copy():
    t = torch.randn(6, 4).bfloat16().t()
    assert _hash_tensor_bytes(t) == _hash_tensor_bytes(t.contiguous())
