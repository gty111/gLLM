import torch
from torch import nn

from gllm.dist_utils import is_first_pp_rank
from gllm.models.deepseek_v2 import DeepseekV2ForCausalLM
from gllm.models.kimi_k25_vision import KimiPatchMerger, KimiVisionTower
from gllm.models.utils import _merge_multimodal_embeddings


class KimiK25ForConditionalGeneration(nn.Module):
    """Kimi-K2.5 multimodal runtime: MoonViT3d vision tower + PatchMerger
    projector on top of a DeepSeek-V3 language backbone.

    The vision tower runs replicated on every rank (see
    :mod:`gllm.models.kimi_k25_vision`); each rank produces identical image
    embeddings and merges them locally into the text stream at the
    ``<|media_pad|>`` placeholder positions (token id
    ``media_placeholder_token_id``).

    Note on placeholder expansion: Kimi's chat template emits a *single*
    ``<|media_pad|>`` per image and its HF processor does NOT pre-expand it
    to N tokens (unlike Qwen-VL). gLLM's merge requires the placeholder run
    to already be N tokens long, so ``ModelRunner`` expands it before the
    ``is_multimodal`` mask is built (see ``_mm_expand_kimi_placeholders``).
    """

    def __init__(self, config):
        super().__init__()
        self.full_config = config
        self.config = config.text_config
        # DeepSeek text backbone expects these runtime flags on its local config.
        self.config.use_mla = getattr(config, "use_mla", True)
        self.config.use_hybrid_state = getattr(config, "use_hybrid_state", False)
        self.config.max_num_batched_tokens = getattr(
            config,
            "max_num_batched_tokens",
            getattr(self.config, "max_num_batched_tokens", 0),
        )

        # ``vision_config`` ships as a plain dict on the top-level config.
        vc = config.vision_config
        self.vision_config = vc if isinstance(vc, dict) else vc.to_dict()
        self.media_placeholder_token_id = config.media_placeholder_token_id
        # Spatial merge collapses (kh*kw) patches into one token; temporal pool
        # collapses T frames. Used by ModelRunner to size placeholder runs.
        self.spatial_merge_size = int(self.vision_config["merge_kernel_size"][0]) * int(
            self.vision_config["merge_kernel_size"][1]
        )

        # Encoder-disaggregation parity with the Qwen-VL wrappers: the LM node
        # skips the vision tower; the encoder node skips the LM. Vision also
        # only lives on PP0 (the embed merge happens before the first layer).
        self.skip_visual = getattr(config, "skip_visual", False)
        if self.skip_visual or not is_first_pp_rank():
            self.vision_tower = None
            self.mm_projector = None
        else:
            self.vision_tower = KimiVisionTower(self.vision_config)
            self.mm_projector = KimiPatchMerger(self.vision_config)

        self.skip_language = getattr(config, "skip_language", False)
        if self.skip_language:
            self.language_model = None
            self.num_layers = 0
            self.num_kv_heads = 0
            self.head_dim = 0
            self.ret_residual = False
            return

        self.language_model = DeepseekV2ForCausalLM(self.config)
        self.num_layers = self.language_model.num_layers
        self.num_kv_heads = self.language_model.num_kv_heads
        self.head_dim = self.language_model.head_dim
        self.ret_residual = self.language_model.ret_residual

    # ------------------------------------------------------------------
    # Multimodal contract (mirrors the Qwen-VL wrappers)
    # ------------------------------------------------------------------
    def get_mm_placeholder_token_ids(self):
        return [self.media_placeholder_token_id]

    def embed_multimodal(self, **kwargs):
        """Run the vision tower + projector. Returns a tuple of per-image
        ``[N_i, text_hidden]`` tensors (one per image), matching the order
        of ``pixel_values`` / ``grid_thws``.
        """
        pixel_values = kwargs.get("pixel_values")
        grid_thws = kwargs.get("grid_thws")
        if pixel_values is None or grid_thws is None:
            return ()

        device = self.vision_tower.patch_embed.proj.weight.device
        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        if not isinstance(pixel_values, torch.Tensor):
            pixel_values = torch.as_tensor(pixel_values)
        if not isinstance(grid_thws, torch.Tensor):
            grid_thws = torch.as_tensor(grid_thws)
        pixel_values = pixel_values.to(device=device, dtype=dtype)
        grid_thws = grid_thws.to(device=device, dtype=torch.int64)

        merged = self.vision_tower(pixel_values, grid_thws)  # list[(N_i, k, C)]
        projected = self.mm_projector(merged)  # list[(N_i, text_hidden)]
        return tuple(projected)

    def embed_multimodal_single(self, **mm_input) -> torch.Tensor:
        out = self.embed_multimodal(**mm_input)
        assert out is not None and len(out) == 1, (
            f"embed_multimodal_single expected exactly one item, got "
            f"{0 if out is None else len(out)}"
        )
        return out[0]

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal: torch.Tensor = None,
    ) -> torch.Tensor:
        # The placeholder id (e.g. 163605) lives outside the LM vocab table,
        # so zero it out before the embedding lookup to avoid an OOB gather;
        # those rows are overwritten by the vision embeddings below.
        if is_multimodal is not None:
            in_vocab_ids = input_ids.masked_fill(is_multimodal, 0)
        else:
            in_vocab_ids = input_ids
        inputs_embeds = self.language_model.model.embed_tokens(in_vocab_ids)

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            is_multimodal=is_multimodal,
            multimodal_embeddings=multimodal_embeddings,
        )

    # ------------------------------------------------------------------
    def forward(self, input_data, hidden_states=None, residual=None):
        return self.language_model(input_data, hidden_states, residual)

    def compute_logits(self, input_data, hidden_states: torch.Tensor):
        return self.language_model.compute_logits(input_data, hidden_states)

    def load_weights(self, weights, mp_load_progress=None):
        # 1) Language model: keys are prefixed ``language_model.``.
        if not self.skip_language and self.language_model is not None:
            language_weights = {}
            prefix = "language_model."
            for key, value in weights.items():
                if key.startswith(prefix):
                    language_weights[key[len(prefix) :]] = value
            if not language_weights:
                # Fallback for already-stripped language-only checkpoints.
                language_weights = weights
            self.language_model.load_weights(language_weights, mp_load_progress)

        if not is_first_pp_rank():
            return
        if self.skip_visual or self.vision_tower is None:
            return

        # 2) Vision tower + projector: verbatim copy (replicated, no TP). The
        # checkpoint key names match our submodule names exactly, so a plain
        # ``copy_`` over named_parameters suffices.
        self._load_replicated(self.vision_tower, weights, "vision_tower.")
        self._load_replicated(self.mm_projector, weights, "mm_projector.")

    @staticmethod
    def _load_replicated(module: nn.Module, weights, prefix: str) -> None:
        for name, param in module.named_parameters():
            src_key = f"{prefix}{name}"
            if src_key not in weights:
                raise KeyError(
                    f"Kimi-K2.5 vision weight missing from checkpoint: {src_key}"
                )
            src = weights[src_key]
            assert param.shape == src.shape, (
                f"shape mismatch for {src_key}: param {tuple(param.shape)} "
                f"vs checkpoint {tuple(src.shape)}"
            )
            param.data.copy_(src.to(param.dtype))
