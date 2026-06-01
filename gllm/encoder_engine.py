"""Encoder engine: vision-only ViT inference for encoder disaggregation.

A single Encoder replica (one process, one GPU) owns the full visual stack:

    raw mm_content  --processor-->  pixel_values + grid_thw
                    --hash------->  content_hash (prefix-cache key, §5.4.4)
                    --ViT--------->  [N_vis_i, visual_dim*(1+L)] embedding

and nothing else: no language model, no KV cache, no scheduler, no sampler
(design §4.2). The embedding is then NIXL-written straight to the LM PP0
worker (wired in later phases); this module is purely the compute side.

Numerical equivalence with the monolith is preserved by reusing the exact
``model.embed_multimodal`` code path (via ``embed_multimodal_single``) and the
exact processor + content-hash helpers from :mod:`gllm.model_runner`.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch
from logger import logger
from transformers import AutoProcessor
from transformers.image_utils import load_images
from transformers.video_utils import load_video

from gllm.model_loader import ModelLoader
from gllm.model_runner import (
    MultiModalEmbeddingCache,
    _build_item_content_hash,
)


class EncoderEngine:
    """Loads the vision tower + processor and encodes one mm item at a time."""

    def __init__(
        self,
        model_path: str,
        load_format: str = "auto",
        mm_processor_min_pixels: Optional[int] = None,
        mm_processor_max_pixels: Optional[int] = None,
        mm_embed_cache_mb: float = 256.0,
        max_num_batched_tokens: int = 8192,
    ):
        # Tell the model loader to construct ONLY the vision tower. Set before
        # ModelLoader reads the config so ``config.skip_language`` is picked up.
        os.environ["GLLM_SKIP_LANGUAGE"] = "1"
        os.environ.setdefault("GLLM_SKIP_VISUAL", "0")

        self.model_path = model_path
        self.model_loader = ModelLoader(load_format, model_path, max_num_batched_tokens)
        assert self.model_loader.use_mm, (
            f"{model_path} is not a multimodal model; nothing for the encoder "
            "to do"
        )

        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        self.image_processor = self.processor.image_processor
        self.video_processor = self.processor.video_processor
        if mm_processor_min_pixels is not None:
            self.image_processor.min_pixels = mm_processor_min_pixels
            self.video_processor.min_pixels = mm_processor_min_pixels
            self.image_processor.size["shortest_edge"] = mm_processor_min_pixels
            self.video_processor.size["shortest_edge"] = mm_processor_min_pixels
        if mm_processor_max_pixels is not None:
            self.image_processor.max_pixels = mm_processor_max_pixels
            self.video_processor.max_pixels = mm_processor_max_pixels
            self.image_processor.size["longest_edge"] = mm_processor_max_pixels
            self.video_processor.size["longest_edge"] = mm_processor_max_pixels

        # Per-replica content-hash -> embedding dedup cache (design §4.2.1).
        self.mm_embed_cache = MultiModalEmbeddingCache(
            max_entries=256, max_mb=mm_embed_cache_mb
        )

        self.model: torch.nn.Module = None
        self.dtype = self.model_loader.dtype
        # spatial_merge_size is needed to compute N_vis from grid_thw; cache it
        # off the config so we don't have to reach into the (TP-sharded) tower.
        self.spatial_merge_size = self.model_loader.config.vision_config.spatial_merge_size

    def init(self, mp_load_progress=None) -> None:
        self.model = self.model_loader.load_model(mp_load_progress)
        self.model.eval()
        assert getattr(self.model, "visual", None) is not None, (
            "encoder model has no vision tower"
        )
        logger.info("EncoderEngine ready: vision tower loaded, language model skipped")

    # ------------------------------------------------------------------
    # CPU: processor + grid + token count + content hash (per item)
    # ------------------------------------------------------------------
    def run_processor(
        self, content, modality: str
    ) -> Tuple[Dict, torch.Tensor]:
        """Run the image/video processor for a SINGLE mm item.

        Returns ``(mm_input, grid_thw)`` where ``mm_input`` has the kwargs that
        ``embed_multimodal`` consumes (``pixel_values``/``image_grid_thw`` or
        the video equivalents) and ``grid_thw`` is the per-item ``[1, 3]`` grid
        tensor (CPU).
        """
        if modality == "image":
            images = load_images([content])
            out = self.image_processor(images=images)
            grid_thw = out["image_grid_thw"]
            if isinstance(grid_thw, torch.Tensor):
                grid_thw = grid_thw.cpu()
            mm_input = {
                "pixel_values": out["pixel_values"],
                "image_grid_thw": grid_thw,
            }
            return mm_input, grid_thw
        elif modality == "video":
            video_data, metadata = load_video(content)
            out = self.video_processor(videos=[video_data], video_metadata=[metadata])
            grid_thw = out["video_grid_thw"]
            if isinstance(grid_thw, torch.Tensor):
                grid_thw = grid_thw.cpu()
            mm_input = {
                "pixel_values_videos": out["pixel_values_videos"],
                "video_grid_thw": grid_thw,
            }
            # carry through optional video timing kwargs if present
            for k in ("second_per_grid_ts", "timestamps"):
                if k in out:
                    mm_input[k] = out[k]
            return mm_input, grid_thw
        raise ValueError(f"unknown modality {modality!r}")

    def num_vis_tokens(self, grid_thw: torch.Tensor) -> int:
        """N_vis = prod(grid_thw) / spatial_merge_size**2 (design §2.1)."""
        merge = self.spatial_merge_size
        return int(grid_thw.prod().item()) // (merge * merge)

    def content_hash(self, mm_input: Dict, grid_thw: torch.Tensor) -> bytes:
        pixel = mm_input.get("pixel_values")
        if pixel is None:
            pixel = mm_input.get("pixel_values_videos")
        return _build_item_content_hash(pixel, grid_thw)

    # ------------------------------------------------------------------
    # GPU: ViT (per item), with per-replica dedup cache
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def encode(self, mm_input: Dict, content_hash: bytes) -> torch.Tensor:
        cached = self.mm_embed_cache.get(content_hash)
        if cached is not None:
            return cached[0]
        vis_item = self.model.embed_multimodal_single(**mm_input)
        # Store as a 1-tuple to match MultiModalEmbeddingCache's value shape.
        self.mm_embed_cache.put(content_hash, (vis_item,))
        return vis_item

    @torch.inference_mode()
    def encode_item(
        self, content, modality: str
    ) -> Dict:
        """Full per-item path: processor -> hash -> ViT. Convenience wrapper."""
        mm_input, grid_thw = self.run_processor(content, modality)
        chash = self.content_hash(mm_input, grid_thw)
        num_tokens = self.num_vis_tokens(grid_thw)
        vis = self.encode(mm_input, chash)
        return {
            "embedding": vis,
            "grid_thw": tuple(grid_thw.flatten().tolist()),
            "num_tokens": num_tokens,
            "content_hash": chash,
            "modality": modality,
            "feat_dim": vis.shape[-1],
        }
