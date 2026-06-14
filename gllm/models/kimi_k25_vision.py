"""Kimi-K2.5 vision tower (MoonViT3d) + PatchMerger projector for gLLM.

This is a faithful re-implementation of the reference modules that ship with
the HF checkpoint (``modeling_kimi_k25.py``):

* ``MoonViT3dPretrainedModel``  -> :class:`KimiVisionTower`
* ``PatchMergerMLP``            -> :class:`KimiPatchMerger`

Design choices vs. the Qwen-VL towers already in gLLM:

* The ViT runs **replicated on every rank** (no tensor parallelism). It is only
  ~3.5B params and the goal here is bit-faithful parity with the HF reference;
  splitting QKV/MLP across ranks would add a class of subtle bugs (the 2D-RoPE
  + complex-valued rotation + fused ``wqkv`` packing) for little memory win on
  the machines this model targets. Every rank loads the full vision weights and
  produces identical image embeddings, so the downstream merge is rank-local.
* Submodule names mirror the checkpoint exactly (``patch_embed``, ``encoder``,
  ``encoder.blocks.N.{norm0,wqkv,wo,norm1,mlp.fc0,mlp.fc1}``,
  ``encoder.final_layernorm``) so weight loading is a verbatim ``copy_``.

The math (patch embed conv, learnable-2D-interpolated + sincos-temporal
positional embedding, 2D rotary embedding, varlen flash attention, 2x2 spatial
merge + temporal-mean pooling) is copied line-for-line from the reference.
"""

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional embeddings
# ---------------------------------------------------------------------------
def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


def _get_1d_sincos_pos_embed(embed_dim: int, t_size: int) -> np.ndarray:
    grid_t = np.arange(t_size, dtype=np.float32)
    return _get_1d_sincos_pos_embed_from_grid(embed_dim, grid_t)


class Learnable2DInterpPosEmbDivided(nn.Module):
    """Learnable 2D spatial grid (interpolated to the actual h/w) plus a fixed
    sincos temporal embedding. Mirrors
    ``Learnable2DInterpPosEmbDivided_fixed`` in the reference."""

    def __init__(
        self,
        height: int,
        width: int,
        num_frames: int,
        dim: int,
        interpolation_mode: str = "bicubic",
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.dim = dim
        self.interpolation_mode = interpolation_mode
        self.weight = nn.Parameter(torch.empty(height, width, dim))
        self.register_buffer(
            "time_weight",
            torch.from_numpy(_get_1d_sincos_pos_embed(dim, num_frames))
            .float()
            .unsqueeze(1),
            persistent=False,
        )

    def _interp(self, shape) -> torch.Tensor:
        return (
            F.interpolate(
                self.weight.permute((2, 0, 1)).unsqueeze(0),
                size=shape,
                mode=self.interpolation_mode,
            )
            .squeeze(0)
            .permute((1, 2, 0))
            .flatten(end_dim=1)
        )

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        pos_embs: List[torch.Tensor] = []
        for t, h, w in grid_thws.tolist():
            assert t <= self.num_frames, f"t:{t} > num_frames:{self.num_frames}"
            if (h, w) == tuple(self.weight.shape[:-1]):
                pos_emb_2d = self.weight.flatten(end_dim=1)
            else:
                pos_emb_2d = self._interp((h, w))
            if t == 1:
                pos_emb_3d = pos_emb_2d
            else:
                pos_emb_3d = pos_emb_2d.unsqueeze(0).repeat(t, 1, 1) + self.time_weight[0:t]
            pos_embs.append(pos_emb_3d.reshape(-1, pos_emb_3d.shape[-1]))
        return x + torch.cat(pos_embs).to(x.dtype)


class KimiVisionPatchEmbed(nn.Module):
    def __init__(
        self,
        out_dim: int,
        in_dim: int = 3,
        patch_size=(14, 14),
        pos_emb_height: int = 64,
        pos_emb_width: int = 64,
        pos_emb_time: int = 4,
    ):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_dim, out_dim, kernel_size=patch_size, stride=patch_size
        )
        self.pos_emb = Learnable2DInterpPosEmbDivided(
            height=pos_emb_height,
            width=pos_emb_width,
            num_frames=pos_emb_time,
            dim=out_dim,
        )

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).view(x.size(0), -1)
        x = self.pos_emb(x, grid_thws)
        return x


# ---------------------------------------------------------------------------
# 2D rotary position embedding
# ---------------------------------------------------------------------------
def _apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    freqs_cis = freqs_cis.unsqueeze(-2)  # (..., 1, head_dim/2)
    xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().view(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Rope2DPosEmb(nn.Module):
    """2D rotary position embedding (height + width axes interleaved).

    Mirrors ``Rope2DPosEmbRepeated``: returns a complex tensor of shape
    ``(sum(t*h*w), dim//2)``.
    """

    def __init__(self, dim: int, max_height: int, max_width: int, theta_base: float = 10000):
        super().__init__()
        assert dim % 4 == 0, "dim must be divisible by 4"
        self.dim = dim
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base

    def _precompute_freqs_cis(self, device: torch.device) -> torch.Tensor:
        N = self.max_height * self.max_width
        flat_pos = torch.arange(0, N).float().to(device)
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = torch.arange(0, self.dim, 4)[: (self.dim // 4)].float().to(device)
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = torch.outer(x_pos, freqs).float()
        y_freqs = torch.outer(y_pos, freqs).float()
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
        freqs_cis = torch.cat([x_cis.unsqueeze(-1), y_cis.unsqueeze(-1)], dim=-1)
        return freqs_cis.reshape(self.max_height, self.max_width, -1)

    def get_freqs_cis(self, grid_thws: torch.Tensor, device: torch.device) -> torch.Tensor:
        if not hasattr(self, "freqs_cis"):
            self.register_buffer(
                "freqs_cis", self._precompute_freqs_cis(device), persistent=False
            )
        shapes = grid_thws.tolist()
        return torch.cat(
            [
                self.freqs_cis[:h, :w].reshape(-1, self.dim // 2).repeat(t, 1)
                for t, h, w in shapes
            ],
            dim=0,
        )


# ---------------------------------------------------------------------------
# Encoder block
# ---------------------------------------------------------------------------
class KimiVisionMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.fc0 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.fc1 = nn.Linear(hidden_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reference uses PytorchGELUTanh (gelu, approximate="tanh").
        x = self.fc0(x)
        x = F.gelu(x, approximate="tanh")
        return self.fc1(x)


class KimiVisionBlock(nn.Module):
    def __init__(self, num_heads: int, hidden_dim: int, mlp_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.norm0 = nn.LayerNorm(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mlp = KimiVisionMLP(hidden_dim, mlp_dim, hidden_dim)
        self.wqkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=True)
        self.wo = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def _attn(self, x, cu_seqlens, max_seqlen, rope_freqs_cis):
        seqlen = x.shape[0]
        xqkv = self.wqkv(x)
        xqkv = xqkv.view(seqlen, 3, self.num_heads, self.head_dim)
        xq, xk, xv = torch.unbind(xqkv, dim=1)
        xq, xk = _apply_rope(xq, xk, rope_freqs_cis)

        from sgl_kernel.flash_attn import flash_attn_varlen_func

        out = flash_attn_varlen_func(
            xq,
            xk,
            xv,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=False,
        )
        out = out.reshape(seqlen, -1)
        return self.wo(out)

    def forward(self, hidden_states, cu_seqlens, max_seqlen, rope_freqs_cis):
        residual = hidden_states
        hidden_states = self.norm0(hidden_states)
        hidden_states = self._attn(hidden_states, cu_seqlens, max_seqlen, rope_freqs_cis)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class KimiVisionEncoder(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, num_heads: int, mlp_dim: int):
        super().__init__()
        self.rope_2d = Rope2DPosEmb(hidden_dim // num_heads, 512, 512)
        self.blocks = nn.ModuleList(
            [KimiVisionBlock(num_heads, hidden_dim, mlp_dim) for _ in range(num_layers)]
        )
        self.final_layernorm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        rope_freqs_cis = self.rope_2d.get_freqs_cis(grid_thws, hidden_states.device)
        lengths = torch.cat(
            (
                torch.zeros(1, dtype=grid_thws.dtype, device=grid_thws.device),
                grid_thws[:, 0] * grid_thws[:, 1] * grid_thws[:, 2],
            )
        )
        max_seqlen = int(lengths.max().item())
        cu_seqlens = lengths.to(hidden_states.device).cumsum(0, dtype=torch.int32)
        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens, max_seqlen, rope_freqs_cis)
        return self.final_layernorm(hidden_states)


def _tpool_patch_merger(x, grid_thws, merge_kernel_size=(2, 2)) -> List[torch.Tensor]:
    d_model = x.size(-1)
    outputs: List[torch.Tensor] = []
    pre_sum = 0
    kh, kw = merge_kernel_size
    for t, h, w in grid_thws.tolist():
        seq = x[pre_sum : pre_sum + t * h * w]
        new_h, new_w = h // kh, w // kw
        reshaped = seq.view(t, new_h, kh, new_w, kw, d_model)
        reshaped = reshaped.permute(0, 1, 3, 2, 4, 5).contiguous().mean(dim=0)
        outputs.append(reshaped.view(new_h * new_w, kh * kw, -1))
        pre_sum += t * h * w
    return outputs


class KimiVisionTower(nn.Module):
    """MoonViT3d: patch embed -> 27 encoder blocks -> 2x2 spatial merge + temporal pool.

    Returns a list of ``(num_merged_tokens, kh*kw, hidden)`` tensors, one per
    image, ready for :class:`KimiPatchMerger`.
    """

    def __init__(self, vision_config):
        super().__init__()
        c = vision_config
        self.hidden_size = c["vt_hidden_size"]
        self.num_heads = c["vt_num_attention_heads"]
        self.patch_size = c["patch_size"]
        self.merge_kernel_size = tuple(c["merge_kernel_size"])
        self.merge_type = c.get("merge_type", "sd2_tpool")

        self.patch_embed = KimiVisionPatchEmbed(
            out_dim=self.hidden_size,
            patch_size=self.patch_size,
            pos_emb_height=c["init_pos_emb_height"],
            pos_emb_width=c["init_pos_emb_width"],
            pos_emb_time=c["init_pos_emb_time"],
        )
        self.encoder = KimiVisionEncoder(
            hidden_dim=self.hidden_size,
            num_layers=c["vt_num_hidden_layers"],
            num_heads=self.num_heads,
            mlp_dim=c["vt_intermediate_size"],
        )

    def forward(self, pixel_values: torch.Tensor, grid_thws: torch.Tensor) -> List[torch.Tensor]:
        assert grid_thws.ndim == 2 and grid_thws.size(1) == 3
        hidden_states = self.patch_embed(pixel_values, grid_thws)
        hidden_states = self.encoder(hidden_states, grid_thws)
        assert self.merge_type == "sd2_tpool", f"unsupported merge_type {self.merge_type}"
        return _tpool_patch_merger(
            hidden_states, grid_thws, merge_kernel_size=self.merge_kernel_size
        )


class KimiPatchMerger(nn.Module):
    """PatchMergerMLP: LayerNorm -> Linear(mm*k -> mm*k) -> GELU -> Linear(mm*k -> text)."""

    def __init__(self, vision_config):
        super().__init__()
        c = vision_config
        eps = c.get("projector_ln_eps", 1e-5)
        mm_hidden = c["mm_hidden_size"]
        kh, kw = c["merge_kernel_size"]
        self.hidden_size = mm_hidden * (kh * kw)
        text_hidden = c["text_hidden_size"]
        self.pre_norm = nn.LayerNorm(mm_hidden, eps=eps)
        # nn.Sequential to keep checkpoint key names ``proj.0`` / ``proj.2``.
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, text_hidden),
        )

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        return [
            self.proj(self.pre_norm(item).view(item.shape[0], -1)) for item in x
        ]
