import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from collections.abc import Callable
from functools import partial, lru_cache

from gllm.input_data import InputData
from gllm.layers.vocab_parallel_embedding import ParallelLMHead
from gllm.models.utils import _merge_multimodal_embeddings
from gllm.models.weight_loader import (
    LoadContext,
    WeightRule,
    contains,
    hv_proj_dim0,
    hv_proj_dim1,
    hv_qkv_fused_split,
    run_vision_loader,
)
from gllm.layers.linear import ColumnParallelLinear, RowParallelLinear
from gllm.dist_utils import get_tp_size, is_first_pp_rank, is_last_pp_rank

from .qwen2_5_vl import (MultiModalEmbeddings, Qwen2_5_VLVideoEmbeddingInputs, Qwen2_5_VLVideoInputs, 
                         Qwen2_5_VLVideoPixelInputs, 
                         Qwen2_5_VisionAttention, Qwen2_5_VisionRotaryEmbedding, 
                         Qwen2_5_VLImageInputs, Qwen2_5_VLImagePixelInputs,
                         Qwen2_5_VLImageEmbeddingInputs)
from .qwen3 import Qwen3Model, Qwen3ForCausalLM


class Qwen3_VisionPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )
        self._use_linear = torch.__version__.startswith("2.9.")
        self._input_size = in_channels * temporal_patch_size * patch_size * patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L, C = x.shape
        x = x.view(L, -1, self.temporal_patch_size, self.patch_size, self.patch_size)
        if self._use_linear:
            from gllm.layers.conv import conv3d_patch_forward
            x = conv3d_patch_forward(x, self.proj.weight, self.proj.bias,
                                     self.hidden_size, self._input_size, self.kernel_size)
            x = x.view(L, self.hidden_size)
        else:
            x = self.proj(x).view(L, self.hidden_size)
        return x

    @property
    def kernel_size(self):
        return (self.temporal_patch_size, self.patch_size, self.patch_size)
    
class Qwen3_VisionMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        bias: bool = False,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        quant_config = None,
    ):
        super().__init__()
        self.linear_fc1 = ColumnParallelLinear(
            in_features,
            hidden_features,
            bias=bias,
            quant_config=quant_config,
            return_bias=False,
        )
        self.linear_fc2 = RowParallelLinear(
            hidden_features,
            in_features,
            bias=bias,
            quant_config=quant_config,
            return_bias=False,
        )
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor):
        mlp_output = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return mlp_output
    
class Qwen3_VisionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        norm_layer: Callable[[int], nn.Module] | None = None,
        quant_config = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = Qwen2_5_VisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            quant_config=quant_config,
        )
        self.mlp = Qwen3_VisionMLP(
            dim,
            mlp_hidden_dim,
            act_fn=act_fn,
            bias=True,
            quant_config=quant_config,
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        max_seqlen: torch.Tensor,  # Only used for Flash Attention
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            max_seqlen=max_seqlen,
        )

        x = x + self.mlp(self.norm2(x))
        return x
    
class Qwen3_VisionPatchMerger(nn.Module):
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        norm_layer: Callable[[int], nn.Module] | None = None,
        spatial_merge_size: int = 2,
        use_postshuffle_norm: bool = False,
        quant_config = None,
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)

        self.use_postshuffle_norm = use_postshuffle_norm
        if self.use_postshuffle_norm:
            context_dim = self.hidden_size

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(context_dim)
        self.linear_fc1 = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            quant_config=quant_config,
        )
        self.act_fn = nn.GELU()
        self.linear_fc2 = RowParallelLinear(
            self.hidden_size,
            d_model,
            bias=True,
            quant_config=quant_config,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.hidden_size))
        else:
            x = self.norm(x).view(-1, self.hidden_size)

        x_parallel = self.linear_fc1(x)
        x_parallel = self.act_fn(x_parallel)
        out = self.linear_fc2(x_parallel)
        return out
    
class Qwen3_VisionTransformer(nn.Module):
    def __init__(
        self,
        vision_config,
        norm_eps: float = 1e-6,
        quant_config = None,
    ) -> None:
        super().__init__()
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads
        self.num_position_embeddings = vision_config.num_position_embeddings
        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.temporal_patch_size = vision_config.temporal_patch_size
        self.deepstack_visual_indexes = (
            vision_config.deepstack_visual_indexes
            if hasattr(vision_config, "deepstack_visual_indexes")
            else []
        )
        self.num_grid_per_side = int(self.num_position_embeddings**0.5)

        self.tp_size = get_tp_size()

        # NOTE: This is used for creating empty tensor for all_gather for
        # DP ViT. Here out_hidden_size is enlarged due to deepstack
        self.out_hidden_size = vision_config.out_hidden_size * (
            1 + len(self.deepstack_visual_indexes)
        )

        self.patch_embed = Qwen3_VisionPatchEmbed(
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
            in_channels=vision_config.in_channels,
            hidden_size=self.hidden_size,
        )

        self.pos_embed = nn.Embedding(self.num_position_embeddings, self.hidden_size)

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        self.merger = Qwen3_VisionPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=self.hidden_size,
            norm_layer=norm_layer,
            spatial_merge_size=self.spatial_merge_size,
            quant_config=quant_config,
        )

        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3_VisionPatchMerger(
                    d_model=vision_config.out_hidden_size,
                    context_dim=self.hidden_size,
                    spatial_merge_size=self.spatial_merge_size,
                    use_postshuffle_norm=True,
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                )
                for layer_idx in range(len(self.deepstack_visual_indexes))
            ]
        )
        
        if vision_config.hidden_act == "silu":
            self.act_fn = F.silu
        elif vision_config.hidden_act == "gelu_pytorch_tanh":
            self.act_fn = F.gelu
        else:
            raise Exception(f"Unsupported activation function: {vision_config.hidden_act}")

        self.blocks = nn.ModuleList(
            [
                Qwen3_VisionBlock(
                    dim=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_hidden_dim=vision_config.intermediate_size,
                    act_fn=self.act_fn,
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                )
                for layer_idx in range(vision_config.depth)
            ]
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    @staticmethod
    @lru_cache(maxsize=1024)
    def rot_pos_ids(h: int, w: int, spatial_merge_size: int) -> torch.Tensor:
        hpos_ids = np.broadcast_to(np.arange(h).reshape(h, 1), (h, w))
        h_div = h // spatial_merge_size
        w_div = w // spatial_merge_size
        hpos_ids = hpos_ids.reshape(
            h_div,
            spatial_merge_size,
            w_div,
            spatial_merge_size,
        )
        hpos_ids = hpos_ids.transpose(0, 2, 1, 3)
        hpos_ids = hpos_ids.flatten()

        wpos_ids = np.broadcast_to(np.arange(w).reshape(1, w), (h, w))
        wpos_ids = wpos_ids.reshape(
            h_div,
            spatial_merge_size,
            w_div,
            spatial_merge_size,
        )
        wpos_ids = wpos_ids.transpose(0, 2, 1, 3)
        wpos_ids = wpos_ids.flatten()

        return torch.from_numpy(np.stack([hpos_ids, wpos_ids], axis=-1))

    def rot_pos_emb(self, grid_thw: list[list[int]]):
        max_grid_size = max(max(h, w) for _, h, w in grid_thw)
        pos_ids = [
            self.rot_pos_ids(h, w, self.spatial_merge_size)
            if t == 1
            else self.rot_pos_ids(h, w, self.spatial_merge_size).repeat(t, 1)
            for t, h, w in grid_thw
        ]
        pos_ids = torch.cat(pos_ids, dim=0).to(self.device, non_blocking=True)

        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def fast_pos_embed_interpolate(self, grid_thw: list[list[int]]) -> torch.Tensor:
        num_grid_per_side = self.num_grid_per_side
        m_size = self.spatial_merge_size
        hidden_dim = self.pos_embed.embedding_dim

        outputs = []
        for t, h, w in grid_thw:
            h_idxs = torch.linspace(
                0, num_grid_per_side - 1, h, dtype=torch.float32, device=self.device
            )
            w_idxs = torch.linspace(
                0, num_grid_per_side - 1, w, dtype=torch.float32, device=self.device
            )

            h_floor = h_idxs.to(torch.long)
            w_floor = w_idxs.to(torch.long)
            h_ceil = torch.clamp(h_floor + 1, max=num_grid_per_side - 1)
            w_ceil = torch.clamp(w_floor + 1, max=num_grid_per_side - 1)

            dh = h_idxs - h_floor
            dw = w_idxs - w_floor

            # Create meshgrid view for all h, w vars
            dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
            h_floor_grid, w_floor_grid = torch.meshgrid(h_floor, w_floor, indexing="ij")
            h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil, w_ceil, indexing="ij")

            # original computation of weights
            # w00 = (1 - dh_grid) * (1 - dw_grid)
            # w01 = (1 - dh_grid) * dw_grid
            # w10 = dh_grid * (1 - dw_grid)
            # w11 = dh_grid * dw_grid
            # we reuse w11 here to avoid duplicate
            # dh_grid * dw_grid computation
            w11 = dh_grid * dw_grid
            w10 = dh_grid - w11
            w01 = dw_grid - w11
            w00 = 1 - dh_grid - w01

            h_grid = torch.stack([h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid])
            w_grid = torch.stack([w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid])
            h_grid_idx = h_grid * num_grid_per_side

            indices = (h_grid_idx + w_grid).reshape(4, -1)
            weights = torch.stack([w00, w01, w10, w11], dim=0).reshape(4, -1, 1)
            weights = weights.to(dtype=self.dtype)

            embeds = self.pos_embed(indices)
            embeds *= weights
            combined = embeds.sum(dim=0)

            combined = combined.reshape(
                h // m_size, m_size, w // m_size, m_size, hidden_dim
            )
            combined = combined.permute(0, 2, 1, 3, 4).reshape(1, -1, hidden_dim)
            repeated = combined.expand(t, -1, -1).reshape(-1, hidden_dim)
            outputs.append(repeated)

        return torch.cat(outputs, dim=0)

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor | list[list[int]],
    ) -> torch.Tensor:
        hidden_states = x.to(device=self.device, dtype=self.dtype, non_blocking=True)
        hidden_states = self.patch_embed(hidden_states)

        if isinstance(grid_thw, list):
            grid_thw_list = grid_thw
            grid_thw = np.array(grid_thw, dtype=np.int32)
        else:
            grid_thw_list = grid_thw.tolist()
            grid_thw = grid_thw.cpu().numpy()

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw_list)
        hidden_states = hidden_states + pos_embeds
        rotary_pos_emb = self.rot_pos_emb(grid_thw_list)
        rotary_pos_emb = rotary_pos_emb.to(hidden_states.device, non_blocking=True)

        cu_seqlens = np.repeat(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            axis=0, dtype=np.int32
        )
        cu_seqlens = np.concatenate([np.zeros(1, dtype=np.int32), cu_seqlens])
        max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())
        cu_seqlens = torch.from_numpy(cu_seqlens).to(self.device, non_blocking=True)
        hidden_states = hidden_states.unsqueeze(1)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
                max_seqlen=max_seqlen,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_merger_idx = self.deepstack_visual_indexes.index(layer_num)
                deepstack_feature = self.deepstack_merger_list[deepstack_merger_idx](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)
        hidden_states = self.merger(hidden_states)
        hidden_states = torch.cat(
            [hidden_states] + deepstack_feature_lists, dim=1
        )  # [seq_len, hidden_size * (1 + depth_of_deepstack)]
        return hidden_states

class Qwen3LLMModel(Qwen3Model):
    def forward(
        self,
        input_data: InputData,
        hidden_states = None,
        residual = None,
        # args for deepstack
        deepstack_input_embeds = None,
    ):
        if is_first_pp_rank() and hidden_states is None:
            hidden_states = self.embed_tokens(input_data.get_tokens())
            
        for local_layer_idx, layer in enumerate(self.layers):
            layer_idx = local_layer_idx + self.start_layer

            hidden_states, residual = layer(
                input_data,
                hidden_states,
                residual,
            )

            if deepstack_input_embeds is not None and layer_idx in range(
                0, len(deepstack_input_embeds)
            ):
                hidden_states = (
                    hidden_states
                    + deepstack_input_embeds[f"deepstack_input_embeds_{layer_idx}"]
                )

        if not is_last_pp_rank():
            return hidden_states, residual
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3LLMForCausalLM(Qwen3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.model = Qwen3LLMModel(config)

        if is_last_pp_rank():
            if config.tie_word_embeddings:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                )
                self.lm_head.tie_weights(self.model.embed_tokens)
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                )

class Qwen3VLForConditionalGeneration(nn.Module):

    def __init__(self, config, language_model_type=Qwen3LLMForCausalLM):
        super().__init__()
        # Read the vision tower's quant config from ``vision_config`` only.
        # The top-level ``config.quantization_config`` is *not* used as a
        # fallback on purpose: today's quantized VL checkpoints (e.g.
        # Qwen3.5-MoE-FP8) put the FP8 field at the top level but ship a
        # bf16 vision tower, so a top-level fallback would force FP8 init
        # on a tower that has no FP8 scale tensors and crash weight load.
        # A future fully-quantized VL checkpoint should opt-in by calling
        # ``propagate_quantization_config(config, propagate_to=("text_config", "vision_config"))``
        # in the model loader, which fills ``vision_config.quantization_config``
        # and lets this read pick it up naturally. Mixed-precision (text
        # FP8 + vision INT4, etc.) is also supported because the propagate
        # helper never overwrites an explicit per-sub-config setting.
        quant_config = getattr(config.vision_config, "quantization_config", None)

        self.config = config

        self.use_deepstack = hasattr(config.vision_config, "deepstack_visual_indexes")
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack
            else 0
        )
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

        # Encoder-disaggregation: the LM node does not own the vision tower
        # (the visual embeddings arrive over NIXL from a separate encoder
        # process; see docs/encoder_disaggregation_design.md §4.3). We still
        # keep ``visual_dim`` / ``multiscale_dim`` / ``deepstack_*`` above and
        # the deepstack buffers below because ``embed_input_ids`` /
        # ``_compute_deepstack_embeds`` only need those config scalars, never
        # the vision weights. ``self.visual = None`` saves the tower's memory
        # and weight load on the LM side.
        self.skip_visual = getattr(config, "skip_visual", False)
        if self.skip_visual:
            self.visual = None
        else:
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
            )

        # register buffer for deepstack
        if self.use_deepstack:
            self.deepstack_input_embeds = [
                torch.zeros(
                    config.max_num_batched_tokens,
                    config.text_config.hidden_size,
                )
                for _ in range(self.deepstack_num_level)
            ]

        # Encoder-disaggregation: the encoder process owns ONLY the vision
        # tower and never runs the language model (no KV cache / scheduler /
        # sampler). ``skip_language`` lets it construct just ``self.visual``
        # and reuse the exact ``embed_multimodal`` path so its ViT output is
        # numerically identical to the monolith's.
        self.skip_language = getattr(config, "skip_language", False)
        if self.skip_language:
            self.language_model = None
            self.num_layers = 0
            self.num_kv_heads = 0
            self.head_dim = 0
            self.ret_residual = False
            return

        self.language_model = language_model_type(
            config.text_config,
        )

        if not is_first_pp_rank() and hasattr(
            config.vision_config, "deepstack_visual_indexes"
        ):
            assert self.language_model.start_layer >= len(
                config.vision_config.deepstack_visual_indexes
            ), (
                "start_layer should be greater than or equal to "
                "len(deepstack_visual_indexes)"
            )
            
        self.num_layers = self.language_model.num_layers
        self.num_kv_heads = self.language_model.num_kv_heads
        self.head_dim = self.language_model.head_dim
        self.ret_residual = self.language_model.ret_residual

    def _get_deepstack_input_embeds(
        self,
        num_tokens: int,
    ):
        if not getattr(self, "deepstack_input_embeds", None):
            return None  # If vision tower is skipped

        # get deepstack_input_embeds from buffer, and clear the buffer
        return  {
            f"deepstack_input_embeds_{idx}": self.deepstack_input_embeds[idx][
                :num_tokens
            ]
            for idx in range(self.deepstack_num_level)
        }

    def _set_deepstack_input_embeds(
        self,
        deepstack_input_embeds: torch.Tensor,
        offset: int = 0,
    ) -> None:
        """Write a chunk of deepstack residuals into the per-batch buffer.

        ``deepstack_input_embeds`` is ``[L, N, hidden]`` and is copied into
        rows ``[offset : offset + N]`` of the buffer (one row per token
        in the *final batch layout*, i.e. ``hidden_states.size(0)``).

        With prefix caching + chunked prefill, a single forward batch can
        contain ``[decode_rows ‖ seq1_chunk ‖ seq2_chunk ‖ ...]`` where
        each prefill seq only contributes the un-cached tail of its
        prompt. The deepstack buffer must mirror that layout exactly,
        otherwise rows produced by the embed-time vision merger get added
        to the wrong tokens at every ``deepstack_visual_indexes`` layer
        and the LM output degrades (silently — embed_tokens still
        works, but the residual stream picks up bogus per-token visual
        deltas). The caller (model runner) is responsible for zeroing
        positions that no chunk will write to (decode rows, text-only
        prefill chunks); see ``_clear_deepstack_input_embeds``.
        """
        if not getattr(self, "deepstack_input_embeds", None):
            return

        num_tokens = deepstack_input_embeds.size(1)
        end = offset + num_tokens
        cur_len = self.deepstack_input_embeds[0].size(0)
        if end > cur_len:
            # Grow to at least ``end`` rows. We don't preserve the existing
            # contents because the runner zeros the buffer before writing
            # chunks anyway; a fresh ``zeros`` is cheaper than a copy.
            self.deepstack_input_embeds = [
                torch.zeros(
                    end,
                    self.config.text_config.hidden_size,
                    device=self.deepstack_input_embeds[0].device,
                    dtype=self.deepstack_input_embeds[0].dtype,
                )
                for _ in range(self.deepstack_num_level)
            ]
        for idx in range(self.deepstack_num_level):
            self.deepstack_input_embeds[idx][offset:end].copy_(
                deepstack_input_embeds[idx]
            )

    def _clear_deepstack_input_embeds(self, num_tokens: int) -> None:
        if not getattr(self, "deepstack_input_embeds", None):
            return

        # clear deepstack_input_embeds in buffer
        if num_tokens > 0:
            for idx in range(self.deepstack_num_level):
                self.deepstack_input_embeds[idx][:num_tokens].zero_()

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> Qwen2_5_VLImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            return Qwen2_5_VLImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

        if image_embeds is not None:
            return Qwen2_5_VLImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw,
            )

    def _parse_and_validate_video_input(
        self, **kwargs: object
    ) -> Qwen2_5_VLVideoInputs | None:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        second_per_grid_ts = kwargs.pop("second_per_grid_ts", None)
        timestamps = kwargs.pop("timestamps", None)

        if pixel_values_videos is None and video_embeds is None:
            return None

        if pixel_values_videos is not None:
            return Qwen2_5_VLVideoPixelInputs(
                type="pixel_values_videos",
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                timestamps=timestamps,
            )

        if video_embeds is not None:
            return Qwen2_5_VLVideoEmbeddingInputs(
                type="video_embeds",
                video_embeds=video_embeds,
                video_grid_thw=video_grid_thw,
                timestamps=timestamps,
            )

    def _process_image_input(
        self, image_input: Qwen2_5_VLImageInputs
    ) -> tuple[torch.Tensor, ...]:
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
        else:
            pixel_values = image_input["pixel_values"].type(self.visual.dtype)
            image_embeds = self.visual(pixel_values, grid_thw=grid_thw)

        # Split concatenated embeddings for each image item.
        merge_size = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        return image_embeds.split(sizes)

    def _process_video_input(
        self, video_input: Qwen2_5_VLVideoInputs
    ) -> tuple[torch.Tensor, ...]:
        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2

        if video_input["type"] == "video_embeds":
            video_embeds = video_input["video_embeds"].type(self.visual.dtype)
        else:
            pixel_values_videos = video_input["pixel_values_videos"].type(
                self.visual.dtype
            )
            video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw)

        # Split concatenated embeddings for each video item.
        merge_size = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        return video_embeds.split(sizes)

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        mm_input_by_modality = {}
        for input_key in kwargs:
            if (
                input_key in ("pixel_values", "image_embeds")
                and "image" not in mm_input_by_modality
            ):
                mm_input_by_modality["image"] = self._parse_and_validate_image_input(
                    **kwargs
                )
            if (
                input_key in ("pixel_values_videos", "video_embeds")
                and "video" not in mm_input_by_modality
            ):
                mm_input_by_modality["video"] = self._parse_and_validate_video_input(
                    **kwargs
                )
        return mm_input_by_modality

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return None

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor corresponding to a multimodal data item (image or video).
        multimodal_embeddings: list[torch.Tensor] = []

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                image_embeddings = self._process_image_input(multimodal_input)
                multimodal_embeddings.extend(image_embeddings)
            if modality == "video":
                video_embeddings = self._process_video_input(multimodal_input)
                multimodal_embeddings.extend(video_embeddings)

        embeddings_tuple = tuple(multimodal_embeddings)
        return embeddings_tuple

    def _compute_deepstack_embeds(
        self,
        inputs_embeds: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings,
        is_multimodal: torch.Tensor,
    ) -> tuple[torch.Tensor, MultiModalEmbeddings]:
        visual_lens = [len(x) for x in multimodal_embeddings]
        multimodal_embeddings_cat = torch.cat(multimodal_embeddings, dim=0)

        (
            multimodal_embeddings_main,
            multimodal_embeddings_multiscale,
        ) = torch.split(
            multimodal_embeddings_cat,
            [self.visual_dim, self.multiscale_dim],
            dim=-1,
        )

        multimodal_embeddings = torch.split(
            multimodal_embeddings_main, visual_lens, dim=0
        )
        multimodal_embeddings_multiscale = torch.split(
            multimodal_embeddings_multiscale, visual_lens, dim=0
        )

        deepstack_input_embeds = inputs_embeds.new_zeros(
            inputs_embeds.size(0), self.deepstack_num_level * inputs_embeds.size(1)
        )

        deepstack_input_embeds = _merge_multimodal_embeddings(
            inputs_embeds=deepstack_input_embeds,
            multimodal_embeddings=multimodal_embeddings_multiscale,
            is_multimodal=is_multimodal,
        )
        deepstack_input_embeds = deepstack_input_embeds.view(
            inputs_embeds.shape[0], self.deepstack_num_level, self.visual_dim
        )
        deepstack_input_embeds = deepstack_input_embeds.permute(1, 0, 2)

        return deepstack_input_embeds, multimodal_embeddings
    
    def _embed_text_input_ids(
        self,
        input_ids: torch.Tensor,
        embed_input_ids: Callable[[torch.Tensor], torch.Tensor],
        *,
        is_multimodal: torch.Tensor | None,
    ) -> torch.Tensor:
        if is_multimodal is not None:
            in_vocab_ids = input_ids.masked_fill(is_multimodal, 0)
            return embed_input_ids(in_vocab_ids)

        return embed_input_ids(input_ids)
    
    def get_mm_placeholder_token_ids(self):
        return [self.config.image_token_id, self.config.video_token_id]

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return ``(inputs_embeds, deepstack_input_embeds_or_None)``.

        ``deepstack_input_embeds`` is the full-prompt per-token deepstack
        residual ``[deepstack_num_level, N, hidden]`` produced from the
        ViT's multiscale features. Previously this was eagerly copied into
        a shared model buffer here, but with prefix caching the caller
        only feeds the *un-cached* tail (``prompt_embeddings[computed:seq_len]``)
        to the model -- the eagerly-written buffer (sized to the full prompt
        and starting at row 0) then mis-aligns with hidden_states by exactly
        the prefix-hit length, and ``Qwen3LLMModel.forward`` adds the wrong
        deepstack delta to every multimodal layer until the issue manifests
        as garbage output (this was the root cause of the multimodal
        prefix-cache regression that previously forced
        ``--no-enable-prefix-caching``).

        We now return the tensor to the caller, which slices it the same
        way it slices ``inputs_embeds``, concatenates across the batch's
        decode + per-seq prefill chunks, and writes the final layout into
        the buffer via :meth:`_set_deepstack_input_embeds`. Text-only
        prompts and non-deepstack models return ``None`` here.
        """
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
        )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds, None

        if self.use_deepstack:
            (
                deepstack_input_embeds,
                multimodal_embeddings,
            ) = self._compute_deepstack_embeds(
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=multimodal_embeddings,
                is_multimodal=is_multimodal,
            )
        else:
            deepstack_input_embeds = None

        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

        return inputs_embeds, deepstack_input_embeds

    def forward(
        self,
        input_data: InputData,
        hidden_states=None,
        residual=None,
    ):
        if hidden_states is not None and is_first_pp_rank():
            deepstack_input_embeds = self._get_deepstack_input_embeds(
                hidden_states.size(0)
            )
        else:
            deepstack_input_embeds = None

        result = self.language_model.model(
            input_data,
            hidden_states,
            residual,
            # args for deepstack
            deepstack_input_embeds=deepstack_input_embeds,
        )
        residual = None
        if isinstance(result, tuple):
            hidden_states, residual = result
        else:
            hidden_states = result

        if hidden_states is not None and is_first_pp_rank():
            self._clear_deepstack_input_embeds(hidden_states.size(0))

        if is_last_pp_rank():
            return hidden_states
        else:
            assert residual is not None
            return hidden_states, residual

    def compute_logits(
        self,
        input_data: InputData,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(input_data, hidden_states)

    def logits_from_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.language_model.logits_from_hidden(hidden_states)

    def embed_multimodal_single(self, **mm_input) -> torch.Tensor:
        """Encode exactly one mm item and return its raw visual embedding.

        Thin wrapper over :meth:`embed_multimodal` for the per-item encoder
        path (design §4.2.1): ``mm_input`` carries a single image/video item
        (``pixel_values`` + ``image_grid_thw`` for one item, or the video
        equivalents). Returns the ``[N_vis_i, visual_dim * (1 + L)]`` tensor
        that is the i-th element of the monolith's ``embed_multimodal`` tuple.
        """
        out = self.embed_multimodal(**mm_input)
        assert out is not None and len(out) == 1, (
            f"embed_multimodal_single expected exactly one item, got "
            f"{0 if out is None else len(out)}"
        )
        return out[0]

    def load_weights(self, weights, mp_load_progress=None):
        if not getattr(self, "skip_language", False) and self.language_model is not None:
            self.language_model.load_weights(weights, mp_load_progress)

        if not is_first_pp_rank():
            return

        # Encoder-disaggregation LM node: vision weights live in the encoder
        # process, so there is nothing to load here.
        if getattr(self, "skip_visual", False) or self.visual is None:
            return

        ctx = LoadContext(
            weights=weights,
            num_heads=self.visual.num_heads // get_tp_size(),
            head_dim=self.visual.hidden_size // self.visual.num_heads,
            extra={"prefix": "visual."},
        )
        run_vision_loader(self.visual, weights, self._vision_rules(), ctx)

    def _vision_rules(self):
        return [
            WeightRule(contains("attn.qkv"), hv_qkv_fused_split, "v_qkv"),
            WeightRule(
                contains("attn.proj.weight", "linear_fc2.weight"),
                hv_proj_dim1,
                "v_proj_dim1",
            ),
            WeightRule(contains("linear_fc1"), hv_proj_dim0, "v_fc1"),
        ]
