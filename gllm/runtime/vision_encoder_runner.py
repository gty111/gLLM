"""Vision model runner for encoder-disaggregated inference.

A single Encoder replica (one process, one GPU) owns the full visual stack:

    raw mm_content  --processor-->  pixel_values + grid_thw
                    --hash------->  content_hash (prefix-cache key, §5.4.4)
                    --ViT--------->  [N_vis_i, visual_dim*(1+L)] embedding

and nothing else: no language model, no KV cache, no scheduler, no sampler. The embedding is then NIXL-written straight to the LM PP0
worker (wired in later phases); this module is purely the compute side.

Numerical equivalence with the monolith is preserved by reusing the exact
``model.embed_multimodal`` code path (via ``embed_multimodal_single``) and the
exact processor + content-hash helpers from :mod:`gllm.runtime.model_runner`.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Optional, Tuple

import torch
from logger import logger
from transformers import AutoProcessor
from transformers.image_utils import load_images

from gllm.multimodal.video_io import (
    VideoLoader,
    preprocess_segment,
    source_fingerprint,
    video_target_size,
)
from gllm.runtime.model_loader import ModelLoader
from gllm.runtime.model_runner import (
    MultiModalEmbeddingCache,
    _build_item_content_hash,
    apply_mm_processor_pixels,
)


class VisionEncoderRunner:
    """Loads the vision tower + processor and encodes one mm item at a time."""

    def __init__(
        self,
        model_path: str,
        load_format: str = "auto",
        mm_processor_min_pixels: Optional[int] = None,
        mm_processor_max_pixels: Optional[int] = None,
        mm_embed_cache_mb: float = 256.0,
        max_num_batched_tokens: int = 8192,
        video_decode_device: str = "auto",
        video_decode_threads: int = 0,
        video_decode_segments: int = 1,
        video_decode_workers: Optional[int] = None,
    ):
        self.model_path = model_path
        # Construct ONLY the vision tower: ``skip_language=True`` so the loader
        # builds no language model / KV cache / scheduler / sampler, and
        # ``skip_visual=False`` so the tower itself is loaded. Passed explicitly
        # (see gllm.disagg.config.DisaggConfig) instead of via env.
        self.model_loader = ModelLoader(
            load_format,
            model_path,
            max_num_batched_tokens,
            skip_visual=False,
            skip_language=True,
        )
        assert self.model_loader.use_mm, (
            f"{model_path} is not a multimodal model; nothing for the encoder "
            "to do"
        )

        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        self.image_processor = self.processor.image_processor
        self.video_processor = self.processor.video_processor
        apply_mm_processor_pixels(
            self.image_processor,
            self.video_processor,
            min_pixels=mm_processor_min_pixels,
            max_pixels=mm_processor_max_pixels,
        )
        # The encoder owns its GPU (no KV cache, no LM), so video decoding
        # defaults to NVDEC with frames kept on-device for the processor.
        self.video_loader = VideoLoader(
            self.video_processor,
            device=video_decode_device,
            num_threads=video_decode_threads,
            num_segments=video_decode_segments,
            num_workers=video_decode_workers,
        )
        logger.info(
            f"Video decode device: {self.video_loader.device}, "
            f"segments={self.video_loader.num_segments}, "
            f"workers={self.video_loader.num_workers}"
        )

        # Per-replica content-hash -> embedding dedup cache.
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
        logger.info(
            "VisionEncoderRunner ready: vision tower loaded, language model skipped"
        )

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
            video_data, metadata = self.video_loader.load(content)
            # Frames are already the sampled subset (metadata.frames_indices).
            out = self.video_processor(
                videos=[video_data],
                video_metadata=[metadata],
                do_sample_frames=False,
            )
            grid_thw = out["video_grid_thw"]
            if isinstance(grid_thw, torch.Tensor):
                grid_thw = grid_thw.cpu()
            mm_input = {
                "pixel_values_videos": out["pixel_values_videos"],
                "video_grid_thw": grid_thw,
            }
            for k in ("second_per_grid_ts", "timestamps"):
                if k in out:
                    mm_input[k] = out[k]
            return mm_input, grid_thw
        raise ValueError(f"unknown modality {modality!r}")

    def num_vis_tokens(self, grid_thw: torch.Tensor) -> int:
        """N_vis = prod(grid_thw) / spatial_merge_size**2."""
        merge = self.spatial_merge_size
        return int(grid_thw.prod().item()) // (merge * merge)

    def content_hash(self, mm_input: Dict, grid_thw: torch.Tensor) -> bytes:
        pixel = mm_input.get("pixel_values")
        if pixel is None:
            pixel = mm_input.get("pixel_values_videos")
        return _build_item_content_hash(pixel, grid_thw)

    # ------------------------------------------------------------------
    # Segment-streamed video: decode -> preprocess -> ViT per time segment
    # ------------------------------------------------------------------
    def open_video_stream(self, content) -> Optional[Dict]:
        """Plan a segment-streamed video encode *without decoding it*.

        The grid (hence ``num_tokens``) follows from the container metadata and
        the processor's resize rule, so ``MmItemMeta`` can be sent before any
        frame is decoded. The content hash is derived from the source and the
        sampling/resize plan rather than from pixel values (which do not exist
        yet); it is stable for the same video and processor settings, but not
        equal to the monolith's pixel-based hash. ``None`` if the content cannot
        be streamed (e.g. a list of frame images).
        """
        video = self.video_loader.open(content)
        if video is None:
            return None
        meta = video.metadata
        vp = self.video_processor
        num_frames = len(meta.frames_indices)
        size = video_target_size(vp, num_frames, meta.height, meta.width)
        tps = vp.temporal_patch_size
        grid_thw = torch.tensor(
            [[-(-num_frames // tps), size[0] // vp.patch_size, size[1] // vp.patch_size]]
        )
        h = hashlib.sha256(b"gllm-video-stream-v1")
        h.update(source_fingerprint(video.src))
        h.update(repr((meta.frames_indices, size, grid_thw.tolist())).encode())
        h.update(repr(sorted(vp.to_dict().items())).encode())
        return {
            "video": video,
            "size": size,
            "grid_thw": grid_thw,
            "content_hash": h.digest(),
        }

    def video_segment_bounds(self, plan: Dict):
        """``[lo, hi)`` sampled-frame ranges, one per streamed segment."""
        return self.video_loader.segment_bounds(plan["video"])

    def prepare_video_segment(self, plan: Dict, lo: int, hi: int) -> Dict:
        """Decode + preprocess one segment into ViT inputs. Thread-safe: the
        decoder is created on the calling thread; preprocessing runs where
        the frames live (GPU for NVDEC, CPU otherwise)."""
        video = plan["video"]
        frames = self.video_loader.decode_segment(video, lo, hi)
        if tuple(frames.shape[1:3]) != (video.metadata.height, video.metadata.width):
            raise RuntimeError(
                f"decoded frame size {tuple(frames.shape[1:3])} differs from "
                "container metadata (rotated video?); cannot stream"
            )
        out = preprocess_segment(self.video_processor, frames, plan["size"])
        return {
            "pixel_values_videos": out["pixel_values_videos"],
            "video_grid_thw": out["video_grid_thw"].cpu(),
        }

    @torch.inference_mode()
    def encode_segment(self, mm_input: Dict) -> torch.Tensor:
        """ViT on one video segment (no dedup cache). Qwen-VL vision towers
        attend within a temporal group, so concatenated segment outputs equal
        the whole-video embedding."""
        return self.model.embed_multimodal_single(**mm_input)

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
