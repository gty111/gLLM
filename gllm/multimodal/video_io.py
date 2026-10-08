"""Video decoding for multimodal requests, optionally on the GPU via NVDEC.

``transformers.video_utils.load_video(content)`` decodes *every* frame of the
video on the CPU (single-threaded PyAV) and leaves frame sampling to the
video processor. For long videos that is both slow (tens of seconds) and
memory hungry (all decoded frames are materialized in host RAM).

:class:`VideoLoader` instead asks the video processor which frames it is going
to keep (``video_processor.sample_frames``) and decodes only those with
torchcodec:

* ``device="cuda"``: NVDEC hardware decoding. Frames stay on the GPU, so the
  HF video processor (resize / normalize / patchify) runs on the GPU too and
  ``pixel_values_videos`` is produced on-device. NVDEC needs the driver's
  ``libnvcuvid.so``; when it is missing torchcodec reports a CPU fallback at
  decoder construction and the loader switches to multi-threaded CPU decoding.
* ``device="cpu"``: multi-threaded FFmpeg decoding of the sampled frames.

The selected frames and the ``VideoMetadata`` (including ``frames_indices``)
are the same as the processor would pick from a full decode, so the token
layout and timestamps are unchanged. Pixel values are not bit-identical to the
PyAV path (YUV->RGB conversion differs by at most one level per channel).
"""

from __future__ import annotations

import base64
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import torch
from logger import logger
from transformers.video_utils import VideoMetadata, load_video

VIDEO_DECODE_DEVICES = ("auto", "cuda", "cpu")


def torchcodec_available() -> bool:
    try:
        from torchcodec.decoders import VideoDecoder  # noqa: F401
    except Exception:  # ImportError, or a missing/incompatible FFmpeg runtime
        return False
    return True


def resolve_video_decode_device(device: str) -> str:
    """``auto`` -> ``cuda`` when torchcodec can be used, else ``cpu``."""
    if device not in VIDEO_DECODE_DEVICES:
        raise ValueError(
            f"video decode device must be one of {VIDEO_DECODE_DEVICES}, got {device!r}"
        )
    if device == "auto":
        return "cuda" if torch.cuda.is_available() and torchcodec_available() else "cpu"
    return device


def _video_source(content):
    """Map a request's video reference to something torchcodec can open.

    Returns a local path or raw bytes, or ``None`` for inputs torchcodec does
    not handle (e.g. a list of frame images), which keep the HF loader path.
    """
    if isinstance(content, (bytes, bytearray)):
        return bytes(content)
    if not isinstance(content, str):
        return None
    if content.startswith("data:"):
        _, _, payload = content.partition(",")
        return base64.b64decode(payload)
    if content.startswith(("http://", "https://")):
        import requests

        resp = requests.get(content, timeout=60)
        resp.raise_for_status()
        return resp.content
    if content.startswith("file://"):
        content = content[len("file://") :]
    return content if os.path.exists(content) else None


def source_fingerprint(src, edge_bytes: int = 4 << 20) -> bytes:
    """Cheap, stable identity of a video source for cache keys.

    In-memory bytes are hashed in full. For a local file, hashing hundreds of
    MB would delay the item's metadata, so we hash the resolved path, size and
    mtime plus the first and last ``edge_bytes`` (container headers/index).
    """
    h = hashlib.sha256()
    if isinstance(src, (bytes, bytearray)):
        h.update(b"bytes")
        h.update(src)
        return h.digest()
    path = os.path.realpath(src)
    st = os.stat(path)
    h.update(repr((path, st.st_size, st.st_mtime_ns)).encode())
    with open(path, "rb") as f:
        h.update(f.read(edge_bytes))
        if st.st_size > edge_bytes:
            f.seek(max(edge_bytes, st.st_size - edge_bytes))
            h.update(f.read(edge_bytes))
    return h.digest()


def video_target_size(video_processor, num_frames: int, height: int, width: int):
    """``(height, width)`` the processor resizes every frame of a video to.

    Mirrors the processor's own ``smart_resize``: Qwen3-VL budgets pixels over
    the whole clip (depends on ``num_frames``), Qwen2/2.5-VL per frame. Used to
    preprocess a video segment by segment with exactly the whole-video size.
    """
    import importlib
    import inspect

    vp = video_processor
    smart_resize = importlib.import_module(type(vp).__module__).smart_resize
    kwargs = dict(
        factor=vp.patch_size * vp.merge_size,
        min_pixels=vp.size["shortest_edge"],
        max_pixels=vp.size["longest_edge"],
    )
    if "num_frames" in inspect.signature(smart_resize).parameters:
        kwargs.update(num_frames=num_frames, temporal_factor=vp.temporal_patch_size)
    return smart_resize(height=height, width=width, **kwargs)


def preprocess_segment(video_processor, frames: torch.Tensor, size: Tuple[int, int]):
    """Run the video processor on one segment, resized to the whole-video
    ``size``. Concatenating the segments' ``pixel_values_videos`` reproduces
    the whole-video output bit for bit (patches are temporal-group major and
    segments are aligned to ``temporal_patch_size``)."""
    from transformers.image_utils import SizeDict

    vp = video_processor
    x = frames.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW
    x = vp.resize(x, size=SizeDict(height=size[0], width=size[1]), resample=vp.resample)
    return vp(videos=[x], do_resize=False, do_sample_frames=False, return_tensors="pt")


def split_segments(n: int, k: int, align: int = 1) -> List[Tuple[int, int]]:
    """Split ``[0, n)`` into at most ``k`` contiguous, near-equal ``[lo, hi)``
    ranges whose boundaries are multiples of ``align`` (the last one may be
    shorter)."""
    units = -(-n // align)  # ceil
    k = max(1, min(k, units))
    base, rem = divmod(units, k)
    bounds, lo = [], 0
    for i in range(k):
        hi = min(n, lo + (base + (1 if i < rem else 0)) * align)
        bounds.append((lo, hi))
        lo = hi
    return bounds


@dataclass
class OpenedVideo:
    """A video whose sampled frames are known but not yet decoded."""

    src: object
    device: str
    metadata: VideoMetadata  # frames_indices = the sampled frames


class VideoLoader:
    """Decode the frames a video processor will keep, on GPU or CPU.

    With ``num_segments > 1`` the sampled frames are split into contiguous time
    segments, each decoded by its own decoder (its own NVDEC session, or its
    own FFmpeg decoder) on a pool of ``num_workers`` threads. Every decoder
    seeks to the keyframe preceding its first frame, so the frames are exactly
    those a single decoder returns. :meth:`iter_segments` yields segments in
    time order as soon as each one is decoded, which lets callers pipeline
    preprocessing / encoding with the decoding of later segments.
    """

    def __init__(
        self,
        video_processor,
        device: str = "auto",
        num_threads: int = 0,
        num_segments: int = 1,
        num_workers: Optional[int] = None,
    ):
        self.video_processor = video_processor
        self.device = resolve_video_decode_device(device)
        if self.device == "cuda" and not torchcodec_available():
            raise RuntimeError(
                "video decode device 'cuda' requires torchcodec (CUDA build)"
            )
        self.use_torchcodec = torchcodec_available()
        if not self.use_torchcodec:
            logger.warning(
                "torchcodec is unavailable (not installed, or its FFmpeg "
                "libraries libavcodec/libavformat are not on the library "
                "path): videos are decoded whole on the CPU with PyAV, without "
                "NVDEC, segmenting or streaming."
            )
        self.num_threads = num_threads
        self.num_segments = max(1, num_segments)
        self.num_workers = max(1, num_workers or self.num_segments)
        self._pool: Optional[ThreadPoolExecutor] = None
        # Segment boundaries stay on temporal-patch boundaries so a segment
        # maps to whole ViT temporal groups.
        self.align = int(getattr(video_processor, "temporal_patch_size", 1) or 1)

    @property
    def pool(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(
                max_workers=self.num_workers, thread_name_prefix="video-decode"
            )
        return self._pool

    def _decoder(self, src, device: str):
        from torchcodec.decoders import VideoDecoder

        return VideoDecoder(
            src,
            dimension_order="NHWC",
            seek_mode="exact",
            device=device,
            num_ffmpeg_threads=self.num_threads,
        )

    def _sample_indices(self, metadata: VideoMetadata):
        vp = self.video_processor
        if getattr(vp, "do_sample_frames", False):
            return vp.sample_frames(metadata=metadata)
        return list(range(metadata.total_num_frames))

    def open(self, content) -> Optional[OpenedVideo]:
        """Read container metadata and pick the sampled frames, without
        decoding. ``None`` if torchcodec cannot handle ``content``."""
        src = _video_source(content) if self.use_torchcodec else None
        if src is None:
            return None
        # Resolve the ordinal here: decode pool threads do not inherit the
        # caller's current CUDA device.
        device = (
            f"cuda:{torch.cuda.current_device()}" if self.device == "cuda" else "cpu"
        )
        decoder = self._decoder(src, device)
        if device != "cpu" and bool(getattr(decoder, "cpu_fallback", False)):
            # Typically "NVcuvid unavailable": the driver's libnvcuvid.so is not
            # installed / not on LD_LIBRARY_PATH. torchcodec's CUDA fallback is
            # a slow single-threaded CPU decode, so switch this loader to the
            # multi-threaded CPU decoder for this and all later videos.
            logger.warning(
                f"NVDEC unavailable ({decoder.cpu_fallback}); falling back to "
                "multi-threaded CPU video decoding. Make the driver's "
                "libnvcuvid.so visible (e.g. via LD_LIBRARY_PATH) to enable it."
            )
            self.device = device = "cpu"
            decoder = self._decoder(src, device)

        m = decoder.metadata
        metadata = VideoMetadata(
            total_num_frames=m.num_frames,
            fps=m.average_fps,
            width=m.width,
            height=m.height,
            duration=m.duration_seconds,
            video_backend="torchcodec",
        )
        metadata.frames_indices = [int(i) for i in self._sample_indices(metadata)]
        return OpenedVideo(src=src, device=device, metadata=metadata)

    def segment_bounds(self, video: OpenedVideo) -> List[Tuple[int, int]]:
        """``[lo, hi)`` ranges over ``metadata.frames_indices``."""
        n = len(video.metadata.frames_indices)
        return split_segments(n, self.num_segments, self.align)

    def decode_segment(self, video: OpenedVideo, lo: int, hi: int) -> torch.Tensor:
        """Decode sampled frames ``[lo, hi)`` (indices into
        ``metadata.frames_indices``) with a decoder created on the calling
        thread: an NVDEC decoder created on one thread and driven from another
        fails ("Could not receive frame from decoder")."""
        indices = video.metadata.frames_indices[lo:hi]
        try:
            dec = self._decoder(video.src, video.device)
            frames = dec.get_frames_at(indices).data
        except RuntimeError as e:
            if video.device == "cpu":
                raise
            # NVDEC sessions occasionally fail under concurrent load; redo
            # this segment on the CPU rather than failing the request.
            logger.warning(f"NVDEC segment decode failed ({e}); retrying on CPU")
            dec = self._decoder(video.src, "cpu")
            frames = dec.get_frames_at(indices).data.to(video.device)
        if frames.is_cuda:
            # The consumer may run on another stream / thread.
            torch.cuda.current_stream(frames.device).synchronize()
        return frames

    def iter_segments(self, video: OpenedVideo) -> Iterator[torch.Tensor]:
        """Yield each segment's frames in time order; all segments are decoded
        concurrently on the pool, so later ones overlap the caller's work."""
        futures = [
            self.pool.submit(self.decode_segment, video, lo, hi)
            for lo, hi in self.segment_bounds(video)
        ]
        for fut in futures:
            yield fut.result()

    def load(self, content) -> Tuple[object, VideoMetadata]:
        """Return ``(frames, metadata)`` ready for the HF video processor.

        Frames are a CUDA tensor when decoded with NVDEC, otherwise a CPU
        tensor (torchcodec) or numpy array (HF loader).
        """
        video = self.open(content)
        if video is not None:
            segments = list(self.iter_segments(video))
            frames = segments[0] if len(segments) == 1 else torch.cat(segments)
            return frames, video.metadata

        # No torchcodec, or an input it cannot open: HF loader, but still only
        # decode the frames the processor will keep.
        def sample_fn(metadata, **kwargs):
            return self._sample_indices(metadata)

        if isinstance(content, str) and content.startswith("file://"):
            content = content[len("file://") :]
        return load_video(content, sample_indices_fn=sample_fn)
