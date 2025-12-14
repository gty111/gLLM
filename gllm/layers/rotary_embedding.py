import math
import torch
import triton
import triton.language as tl
import numpy as np

from typing import Dict, List, Union, Optional
from torch import nn

from gllm.utils import yarn_get_mscale
from gllm.vllm_flash_attn.layers.rotary import apply_rotary_emb

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

# yarn functions
# Inverse dim formula to find dim based on number of rotations
def yarn_find_correction_dim(num_rotations: int,
                             dim: int,
                             base: float = 10000,
                             max_position_embeddings: int = 2048) -> float:
    return (dim * math.log(max_position_embeddings /
                           (num_rotations * 2 * math.pi))) / (2 *
                                                              math.log(base))

# Find dim range bounds based on rotations
def yarn_find_correction_range(
        low_rot: int,
        high_rot: int,
        dim: int,
        base: float = 10000,
        max_position_embeddings: int = 2048) -> tuple[int, int]:
    low = math.floor(
        yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(
        yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def yarn_linear_ramp_mask(low: float, high: float, dim: int,
                          dtype: torch.dtype) -> torch.Tensor:
    if low == high:
        high += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=dtype) - low) / (high - low)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func

class YaRNScalingRotaryEmbedding(RotaryEmbedding):
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
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        # Get n-d magnitude scaling corrected for interpolation
        self.mscale = float(yarn_get_mscale(self.scaling_factor) * attn_factor)
        super().__init__(head_size, rotary_dim, max_position_embeddings, base,
                         is_neox_style)

    def _compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:
        pos_freqs = self.base**(
            torch.arange(0, self.rotary_dim, 2, dtype=torch.float) /
            self.rotary_dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low, high = yarn_find_correction_range(self.beta_fast, self.beta_slow,
                                               self.rotary_dim, self.base,
                                               self.max_position_embeddings)
        # Get n-d rotational scaling corrected for extrapolation
        inv_freq_mask = (1 - yarn_linear_ramp_mask(
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
    
    
    
# --------mrope--------------

@triton.jit
def _triton_qwen2vl_mrope_forward(
    q_ptr,
    k_ptr,
    cos,
    sin,
    num_tokens,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    rd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd: tl.constexpr,
    mrope_section_t: tl.constexpr,
    mrope_section_h: tl.constexpr,
):
    # Adapted from
    # https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/qwen2vl_mrope.py
    # This version supports flatten input tensors from vllm
    # and supports cos and sin cache with shape (3, num_tokens, head_dim // 2)
    # instead of (3, bsz, seq_len, head_dim)
    pid = tl.program_id(0)
    # locate start address
    q_ptr = q_ptr + pid * (n_qh * hd)
    k_ptr = k_ptr + pid * (n_kh * hd)

    # ####################################################################
    # get the cos(mθ_{i...d/2}) and sin(mθ_{i...d/2}) for token position
    # m of this program instance
    # ####################################################################
    # Note: cos and sin now have shape (3, num_tokens, head_dim // 2)

    t_end = mrope_section_t
    h_end = t_end + mrope_section_h

    # Updated stride calculation for half head_dim
    half_rd = rd // 2
    t_cos = cos + pid * half_rd
    h_cos = t_cos + num_tokens * half_rd
    w_cos = h_cos + num_tokens * half_rd
    t_sin = sin + pid * half_rd
    h_sin = t_sin + num_tokens * half_rd
    w_sin = h_sin + num_tokens * half_rd

    # Updated offsets for half head_dim
    cos_offsets = tl.arange(0, pad_hd // 2)
    t_mask = cos_offsets < t_end
    h_mask = (t_end <= cos_offsets) & (cos_offsets < h_end)
    w_mask = (h_end <= cos_offsets) & (cos_offsets < half_rd)

    t_cos_row = tl.load(t_cos + cos_offsets, mask=t_mask, other=0)
    h_cos_row = tl.load(h_cos + cos_offsets, mask=h_mask, other=0)
    w_cos_row = tl.load(w_cos + cos_offsets, mask=w_mask, other=0)
    t_sin_row = tl.load(t_sin + cos_offsets, mask=t_mask, other=0)
    h_sin_row = tl.load(h_sin + cos_offsets, mask=h_mask, other=0)
    w_sin_row = tl.load(w_sin + cos_offsets, mask=w_mask, other=0)

    cos_row = t_cos_row + h_cos_row + w_cos_row
    sin_row = t_sin_row + h_sin_row + w_sin_row

    # ####################################################################
    # Load the left and right half of q and k for the current
    # program instance (i.e. for the current token) separately
    # ####################################################################
    # left half of the head
    first_half_q_offsets = tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(
        0, pad_hd // 2)[None, :]
    first_half_k_offsets = tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(
        0, pad_hd // 2)[None, :]
    first_q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (tl.arange(
        0, pad_hd // 2)[None, :] < rd // 2)
    first_k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (tl.arange(
        0, pad_hd // 2)[None, :] < rd // 2)

    q_tile_1 = tl.load(q_ptr + first_half_q_offsets,
                       mask=first_q_mask,
                       other=0).to(sin_row.dtype)
    k_tile_1 = tl.load(k_ptr + first_half_k_offsets,
                       mask=first_k_mask,
                       other=0).to(sin_row.dtype)

    # right half of the head
    second_half_q_offsets = first_half_q_offsets + (rd // 2)
    second_half_k_offsets = first_half_k_offsets + (rd // 2)
    second_q_mask = first_q_mask
    second_k_mask = first_k_mask

    q_tile_2 = tl.load(q_ptr + second_half_q_offsets,
                       mask=second_q_mask,
                       other=0).to(sin_row.dtype)
    k_tile_2 = tl.load(k_ptr + second_half_k_offsets,
                       mask=second_k_mask,
                       other=0).to(sin_row.dtype)

    # y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
    # Since cos and sin are now half-size,
    # we use the same cos_row and sin_row for both halves
    new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
    tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
    new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
    tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)

    new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
    tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
    new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
    tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)


def triton_mrope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list[int],
    head_size: int,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Qwen2VL mrope kernel.

    Args:
        query: [num_tokens, num_heads * head_size]
        key: [num_tokens, num_kv_heads * head_size]
        cos: [3, num_tokens, head_size //2 ]
            (T/H/W positions with multimodal inputs)
        sin: [3, num_tokens, head_size //2 ]
            (T/H/W positions with multimodal inputs)
        mrope_section: [t, h, w]
        head_size: int
    """
    n_row, n_q_head_head_dim = q.shape
    n_q_head = n_q_head_head_dim // head_size
    n_kv_head = k.shape[1] // head_size
    pad_hd = triton.next_power_of_2(head_size)
    pad_n_q_head = triton.next_power_of_2(n_q_head)
    pad_n_kv_head = triton.next_power_of_2(n_kv_head)

    # ensure tensors passed into the kernel are contiguous.
    # It will be no-op if they are already contiguous
    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    _triton_qwen2vl_mrope_forward[(n_row, )](
        q,
        k,
        cos,
        sin,
        n_row,
        n_q_head,
        n_kv_head,
        head_size,
        rotary_dim,
        pad_n_q_head,
        pad_n_kv_head,
        pad_hd,
        mrope_section[0],
        mrope_section[1],
    )
    return q, k

def apply_rotary_emb_dispatch(x: torch.Tensor, cos: torch.Tensor,
                              sin: torch.Tensor,
                              is_neox_style: bool) -> torch.Tensor:
    """
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    """
    return apply_rotary_emb(x.unsqueeze(0), cos, sin,
                            not is_neox_style).squeeze(0)


class MRotaryEmbedding(RotaryEmbedding):
    """Rotary Embedding with Multimodal Sections."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        mrope_section: Optional[list[int]] = None,
    ) -> None:
        # In Qwen2.5-VL, the maximum index value is related to the duration of
        # the input video. We enlarge max_position_embeddings to 4 times to get
        # a larger the cos and sin cache.
        self.cache_max_position_num = max_position_embeddings * 4
        super().__init__(head_size, rotary_dim, self.cache_max_position_num,
                         base, is_neox_style)

        self.mrope_section = mrope_section
        if self.mrope_section:
            assert sum(self.mrope_section) == rotary_dim // 2

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """MRope forward.

        Args:
            positions:
                [num_tokens,] (text only) or
                [3, num_tokens] (T/H/W positions with multimodal inputs)
            query: [num_tokens, num_heads * head_size]
            key: [num_tokens, num_kv_heads * head_size]
        """
        assert positions.ndim == 1 or positions.ndim == 2
        assert key is not None

        num_tokens = positions.shape[-1]
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query_shape = query.shape
        key_shape = key.shape
        if positions.ndim == 2:
            assert self.mrope_section

            q, k = triton_mrope(
                query,
                key,
                cos,
                sin,
                self.mrope_section,
                self.head_size,
                self.rotary_dim,
            )

            return q.reshape(query_shape), k.reshape(key_shape)

        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        query_rot = apply_rotary_emb_dispatch(query_rot, cos, sin,
                                              self.is_neox_style)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., :self.rotary_dim]
        key_pass = key[..., self.rotary_dim:]
        key_rot = apply_rotary_emb_dispatch(key_rot, cos, sin,
                                            self.is_neox_style)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    @classmethod
    def get_input_positions(
        cls,
        input_tokens: list[int],
        hf_config,
        image_grid_thw: Optional[Union[list[list[int]], torch.Tensor]],
        video_grid_thw: Optional[Union[list[list[int]], torch.Tensor]],
        second_per_grid_ts: Optional[list[float]],
        context_len: int = 0,
        seq_len: Optional[int] = None,
    ) -> tuple[list[list[int]], int]:
        """Get mrope input positions and delta value."""

        image_grid_thw = [] if image_grid_thw is None else image_grid_thw
        video_grid_thw = [] if video_grid_thw is None else video_grid_thw
        second_per_grid_ts = [] if second_per_grid_ts is None else \
            second_per_grid_ts

        llm_positions, mrope_position_delta = \
            cls._vl_get_input_positions_tensor(
                input_tokens=input_tokens,
                hf_config=hf_config,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                context_len=context_len,
                seq_len=seq_len,
            )

        return llm_positions, mrope_position_delta

    @classmethod
    def _vl_get_input_positions_tensor(
        cls,
        input_tokens: list[int],
        hf_config,
        image_grid_thw: Union[list[list[int]], torch.Tensor],
        video_grid_thw: Union[list[list[int]], torch.Tensor],
        second_per_grid_ts: list[float],
        context_len: int = 0,
        seq_len: Optional[int] = None,
    ) -> tuple[torch.Tensor, int]:
        """Get mrope input positions and delta value."""

        image_token_id = hf_config.image_token_id
        video_token_id = hf_config.video_token_id
        vision_start_token_id = hf_config.vision_start_token_id
        spatial_merge_size = hf_config.vision_config.spatial_merge_size
        tokens_per_second = getattr(hf_config.vision_config,
                                    "tokens_per_second", 1.0)

        input_tokens_tensor = torch.tensor(input_tokens, device='cpu')
        vision_start_indices = torch.argwhere(
            input_tokens_tensor == vision_start_token_id).squeeze(1)
        vision_tokens = input_tokens_tensor[vision_start_indices + 1]
        image_nums = (vision_tokens == image_token_id).sum()
        video_nums = (vision_tokens == video_token_id).sum()
        llm_pos_ids_list: list = []

        st = 0
        remain_images, remain_videos = image_nums, video_nums

        image_index, video_index = 0, 0
        for _ in range(image_nums + video_nums):
            video_second_per_grid_t = 0.0
            if image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(image_token_id, st)
            else:
                ed_image = len(input_tokens) + 1
            if video_token_id in input_tokens and remain_videos > 0:
                ed_video = input_tokens.index(video_token_id, st)
            else:
                ed_video = len(input_tokens) + 1
            if ed_image < ed_video:
                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = (
                    video_grid_thw[video_index][0],
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )
                video_second_per_grid_t = 1.0
                if second_per_grid_ts:
                    video_second_per_grid_t = second_per_grid_ts[video_index]
                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t, llm_grid_h, llm_grid_w = \
                t, h // spatial_merge_size, w // spatial_merge_size
            text_len = ed - st

            st_idx = llm_pos_ids_list[-1].max() + 1 if len(
                llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(
                torch.arange(text_len, device='cpu').view(1, -1).expand(3, -1) + st_idx)

            t_index = (torch.arange(llm_grid_t, device='cpu').view(-1, 1).expand(
                -1, llm_grid_h * llm_grid_w) * video_second_per_grid_t *
                       tokens_per_second).long().flatten()

            h_index = torch.arange(llm_grid_h, device='cpu').view(1, -1, 1).expand(
                llm_grid_t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w, device='cpu').view(1, 1, -1).expand(
                llm_grid_t, llm_grid_h, -1).flatten()
            llm_pos_ids_list.append(
                torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(
                llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                torch.arange(text_len, device='cpu').view(1, -1).expand(3, -1) + st_idx)

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        mrope_position_delta = (llm_positions.max() + 1 -
                                len(input_tokens)).item()
        llm_positions = llm_positions[:, context_len:seq_len]

        return llm_positions, mrope_position_delta

    @staticmethod
    def get_next_input_positions(
        mrope_position_delta: int,
        context_len: int,
        seq_len: int,
    ) -> list[list[int]]:
        return [
            list(
                range(context_len + mrope_position_delta,
                      seq_len + mrope_position_delta)) for _ in range(3)
        ]

    @staticmethod
    def get_next_input_positions_tensor(out: np.ndarray, out_offset: int,
                                        mrope_position_delta: int,
                                        context_len: int, num_new_tokens: int):

        values = np.arange(mrope_position_delta + context_len,
                           mrope_position_delta + context_len + num_new_tokens,
                           dtype=out.dtype)
        out[:, out_offset:out_offset + num_new_tokens] = values