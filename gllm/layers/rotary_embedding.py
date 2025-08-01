import math
import torch

from typing import Dict, List, Union, Optional
from torch import nn

from gllm.utils import yarn_get_mscale

def _rotate_neox(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def _rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)

class RotaryEmbedding(nn.Module):
    """Original rotary positional embedding."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style

        cache = self._compute_cos_sin_cache()

        cache = cache.to(dtype=torch.get_default_dtype())
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_inv_freq(self, base):
        """Compute the inverse frequency."""
        # NOTE(woosuk): The HF implementation uses `torch.arange(...).float()`.
        # However, we use `torch.arange(..., dtype=torch.float)` instead to
        # avoid numerical issues with large base values (e.g., 10000000).
        # This may cause a slight numerical difference between the HF
        # implementation and ours.
        # NOTE(woosuk): To exactly match the HF implementation, we need to
        # use CPU to compute the cache and then move it to GPU. However, we
        # create the cache on GPU for faster initialization. This may cause
        # a slight numerical difference between the HF implementation and ours.
        inv_freq = 1.0 / (base**(torch.arange(
            0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))
        return inv_freq

    def _compute_cos_sin_cache(self):
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward(
        self,
        positions,
        query,
        key,
        offsets=None,
    ):
        from gllm import _custom_ops as ops

        # ops.rotary_embedding()/batched_rotary_embedding()
        # are in-place operations that update the query and key tensors.
        if offsets is not None:
            ops.batched_rotary_embedding(positions, query, key, self.head_size,
                                         self.cos_sin_cache,
                                         self.is_neox_style, self.rotary_dim,
                                         offsets)
        else:
            ops.rotary_embedding(positions, query, key, self.head_size,
                                 self.cos_sin_cache, self.is_neox_style)
        return query, key

class LinearScalingRotaryEmbedding(RotaryEmbedding):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factors: Union[List[float], float]
    ) -> None:
        if isinstance(scaling_factors, float):
            scaling_factors = [scaling_factors]
        self.scaling_factors: List[float] = scaling_factors  # noqa
        super().__init__(head_size, rotary_dim, max_position_embeddings, base,
                         is_neox_style)
        # Lazy initialized.
        self._scaling_factor_to_offset: Dict[float, int]

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.base)
        cache_list: List[torch.Tensor] = []
        # offsets to the next cache in a tensor.
        # Each offset corresponds to the same index in scaling_factors.
        offsets: List[int] = []
        for scaling_factor in self.scaling_factors:
            # NOTE(woosuk): self.max_position_embeddings is the original
            # maximum length before applying the rope scaling.
            # Thus, the maximum length after applying the rope scaling is
            # self.max_position_embeddings * self.scaling_factor.
            max_len = self.max_position_embeddings * scaling_factor
            t = torch.arange(max_len, dtype=torch.float)
            t = t / scaling_factor

            freqs = torch.einsum("i,j -> ij", t, inv_freq)
            cos = freqs.cos()
            sin = freqs.sin()
            cache = torch.cat((cos, sin), dim=-1)
            if not cache_list:
                offset = 0
            else:
                last_offset = offsets[-1]
                next_max_len = cache_list[-1].shape[0]
                offset = last_offset + next_max_len
            offsets.append(offset)
            cache_list.append(cache)
        self._scaling_factor_to_offset = {
            float(scaling_factor): offsets[i]
            for i, scaling_factor in enumerate(self.scaling_factors)
        }
        assert len(self.scaling_factors) == len(offsets)
        return torch.cat(cache_list, dim=0)

    @property
    def scaling_factor_to_offset(self) -> Dict[float, int]:
        return self._scaling_factor_to_offset
    
class Llama3RotaryEmbedding(RotaryEmbedding):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        low_freq_factor: float,
        high_freq_factor: float,
        orig_max_position: int,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.orig_max_position = orig_max_position
        super().__init__(head_size, rotary_dim, max_position_embeddings, base,
                         is_neox_style)

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        inv_freqs = super()._compute_inv_freq(base)
        low_freq_wavelen = self.orig_max_position / self.low_freq_factor
        high_freq_wavelen = self.orig_max_position / self.high_freq_factor

        wave_len = 2 * math.pi / inv_freqs
        if self.low_freq_factor != self.high_freq_factor:
            smooth = (self.orig_max_position / wave_len - self.low_freq_factor
                      ) / (self.high_freq_factor - self.low_freq_factor)
        else:
            smooth = 0
        new_freqs = torch.where(
            wave_len < high_freq_wavelen,
            inv_freqs,
            torch.where(
                wave_len > low_freq_wavelen,
                inv_freqs / self.scaling_factor,
                (1 - smooth) * inv_freqs / self.scaling_factor +
                smooth * inv_freqs,
            ),
        )
        return new_freqs

# Inverse dim formula to find dim based on number of rotations
def _yarn_find_correction_dim(num_rotations: int,
                              dim: int,
                              base: float = 10000,
                              max_position_embeddings: int = 2048) -> float:
    return (dim * math.log(max_position_embeddings /
                           (num_rotations * 2 * math.pi))) / (2 *
                                                              math.log(base))

# Find dim range bounds based on rotations
def _yarn_find_correction_range(
        low_rot: int,
        high_rot: int,
        dim: int,
        base: float = 10000,
        max_position_embeddings: int = 2048) -> tuple[int, int]:
    low = math.floor(
        _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(
        _yarn_find_correction_dim(high_rot, dim, base,
                                  max_position_embeddings))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def _yarn_linear_ramp_mask(low: float, high: float, dim: int,
                           dtype: torch.dtype) -> torch.Tensor:
    if low == high:
        high += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=dtype) - low) / (high - low)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func

class DeepseekScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with YaRN method.

    Credits to Peng et al. github.com/jquesnelle/yarn
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        scaling_factor: float,
        *,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1,
        mscale_all_dim: float = 0,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        # Get n-d magnitude scaling corrected for interpolation.
        self.mscale = float(
            yarn_get_mscale(self.scaling_factor, float(mscale)) /
            yarn_get_mscale(self.scaling_factor, float(mscale_all_dim)) *
            attn_factor)
        super().__init__(head_size, rotary_dim, max_position_embeddings, base,
                         is_neox_style)

    def _compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:
        pos_freqs = self.base**(
            torch.arange(0,
                         self.rotary_dim,
                         2,
                         dtype=torch.float) /
            self.rotary_dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low, high = _yarn_find_correction_range(self.beta_fast, self.beta_slow,
                                                self.rotary_dim, self.base,
                                                self.max_position_embeddings)
        # Get n-d rotational scaling corrected for extrapolation
        inv_freq_mask = (1 - _yarn_linear_ramp_mask(
            low, high, self.rotary_dim // 2,
            dtype=torch.float)) * self.extrapolation_factor
        inv_freq = inv_freq_interpolation * (
            1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.scaling_factor)
        t = torch.arange(self.max_position_embeddings * self.scaling_factor,
                         dtype=torch.float32)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = (freqs.cos() * self.mscale)
        sin = (freqs.sin() * self.mscale)
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """PyTorch-native implementation equivalent to forward()."""
        assert key is not None
        query_rot = query[..., :self.rotary_dim]
        key_rot = key[..., :self.rotary_dim]
        if self.rotary_dim < self.head_size:
            query_pass = query[..., self.rotary_dim:]
            key_pass = key[..., self.rotary_dim:]

        if self.cos_sin_cache.device != positions.device:
            self.cos_sin_cache: torch.Tensor = self.cos_sin_cache.to(
                positions.device)
        cos_sin = self.cos_sin_cache[torch.add(positions, offsets)
                                     if offsets is not None else positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.is_neox_style:
            # NOTE(woosuk): Here we assume that the positions tensor has the
            # shape [batch_size, seq_len].
            cos = cos.repeat(1, 1, 2).unsqueeze(-2)
            sin = sin.repeat(1, 1, 2).unsqueeze(-2)
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)

        rotate_fn = _rotate_neox if self.is_neox_style else _rotate_gptj
        query_rot = query_rot * cos + rotate_fn(query_rot) * sin
        key_rot = key_rot * cos + rotate_fn(key_rot) * sin

        if self.rotary_dim < self.head_size:
            query = torch.cat((query_rot, query_pass), dim=-1)
            key = torch.cat((key_rot, key_pass), dim=-1)
        else:
            query = query_rot
            key = key_rot
        return query, key