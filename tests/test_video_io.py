"""VideoLoader: sampled-frame decoding on CPU / NVDEC with CPU fallback."""
import base64
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gllm.multimodal import video_io
from gllm.multimodal.video_io import (
    VideoLoader,
    _video_source,
    preprocess_segment,
    split_segments,
    video_target_size,
)

NUM_FRAMES = 48
KEEP = 8

requires_torchcodec = pytest.mark.skipif(
    not video_io.torchcodec_available(), reason="torchcodec not installed"
)


class _SamplingProcessor:
    """Stands in for an HF video processor: keeps KEEP evenly spaced frames."""

    do_sample_frames = True

    def sample_frames(self, metadata):
        return np.linspace(0, metadata.total_num_frames - 1, KEEP).round().astype(int)


@pytest.fixture(scope="module")
def h264_video(tmp_path_factory):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg CLI not available")
    path = tmp_path_factory.mktemp("video") / "testsrc.mp4"
    cmd = [
        "ffmpeg", "-loglevel", "error", "-f", "lavfi",
        "-i", f"testsrc=size=320x240:rate=24:duration={NUM_FRAMES / 24}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "12", str(path),
    ]
    if subprocess.run(cmd).returncode != 0:
        pytest.skip("ffmpeg cannot encode H.264")
    return str(path)


def _pyav_frames(path, indices):
    import av

    wanted = set(int(i) for i in indices)
    out = {}
    with av.open(path) as c:
        for i, frame in enumerate(c.decode(video=0)):
            if i in wanted:
                out[i] = frame.to_ndarray(format="rgb24")
    return np.stack([out[int(i)] for i in indices])


def test_video_source_forms(tmp_path):
    f = tmp_path / "v.mp4"
    f.write_bytes(b"\x00\x01")
    assert _video_source(str(f)) == str(f)
    assert _video_source(f"file://{f}") == str(f)
    assert _video_source(b"\x00\x01") == b"\x00\x01"
    data_url = "data:video/mp4;base64," + base64.b64encode(b"abc").decode()
    assert _video_source(data_url) == b"abc"
    assert _video_source(str(tmp_path / "missing.mp4")) is None
    assert _video_source(["frame0.jpg", "frame1.jpg"]) is None


def test_resolve_device_rejects_unknown():
    with pytest.raises(ValueError):
        video_io.resolve_video_decode_device("tpu")
    assert video_io.resolve_video_decode_device("cpu") == "cpu"


@requires_torchcodec
def test_cpu_decodes_only_sampled_frames(h264_video):
    loader = VideoLoader(_SamplingProcessor(), device="cpu")
    frames, meta = loader.load(h264_video)

    expected_idx = _SamplingProcessor().sample_frames(meta).tolist()
    assert meta.frames_indices == expected_idx
    assert meta.total_num_frames == NUM_FRAMES
    assert tuple(frames.shape) == (KEEP, 240, 320, 3)
    assert frames.device.type == "cpu"
    ref = _pyav_frames(h264_video, expected_idx).astype(np.int16)
    # Same frames as PyAV; FFmpeg's YUV->RGB conversion is configured
    # differently in torchcodec and PyAV, so pixels differ by a few levels.
    diff = np.abs(frames.numpy().astype(np.int16) - ref)
    assert diff.max() <= 4 and diff.mean() < 1.5


@requires_torchcodec
def test_cuda_fallback_switches_loader_to_cpu(h264_video, monkeypatch):
    loader = VideoLoader(_SamplingProcessor(), device="cpu")
    loader.device = "cuda"
    real_decoder = loader._decoder
    devices = []

    def fake_decoder(src, device):
        devices.append(device)
        if device.startswith("cuda"):
            return SimpleNamespace(cpu_fallback="Falling back due to: NVcuvid unavailable")
        return real_decoder(src, device)

    monkeypatch.setattr(loader, "_decoder", fake_decoder)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    frames, _ = loader.load(h264_video)
    # NVDEC probe at open -> CPU metadata decoder -> CPU segment decoder.
    assert [d.split(":")[0] for d in devices] == ["cuda", "cpu", "cpu"]
    assert loader.device == "cpu"
    assert frames.device.type == "cpu"


@requires_torchcodec
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_nvdec_matches_cpu(h264_video):
    loader = VideoLoader(_SamplingProcessor(), device="cuda")
    frames, meta = loader.load(h264_video)
    if loader.device != "cuda":
        pytest.skip("NVDEC unavailable (libnvcuvid.so not found)")
    assert frames.device.type == "cuda"
    cpu_frames, cpu_meta = VideoLoader(_SamplingProcessor(), device="cpu").load(h264_video)
    assert meta.frames_indices == cpu_meta.frames_indices
    diff = (frames.cpu().to(torch.int16) - cpu_frames.to(torch.int16)).abs()
    # NVDEC and swscale YUV->RGB conversions round differently.
    assert diff.float().mean() < 2.0


def test_split_segments_aligned_and_covering():
    assert split_segments(10, 3, align=2) == [(0, 4), (4, 8), (8, 10)]
    assert split_segments(7, 4, align=2) == [(0, 2), (2, 4), (4, 6), (6, 7)]
    assert split_segments(3, 8, align=2) == [(0, 2), (2, 3)]
    assert split_segments(5, 1) == [(0, 5)]
    for n, k, a in [(768, 16, 2), (87, 8, 2), (1, 4, 2)]:
        b = split_segments(n, k, a)
        assert b[0][0] == 0 and b[-1][1] == n
        assert all(lo % a == 0 and lo < hi for lo, hi in b)
        assert all(x[1] == y[0] for x, y in zip(b, b[1:]))


@requires_torchcodec
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_segmented_decode_matches_single_decoder(h264_video, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    one = VideoLoader(_SamplingProcessor(), device=device)
    frames1, meta1 = one.load(h264_video)
    if device == "cuda" and one.device != "cuda":
        pytest.skip("NVDEC unavailable (libnvcuvid.so not found)")
    seg = VideoLoader(_SamplingProcessor(), device=device, num_segments=3, num_workers=2)
    video = seg.open(h264_video)
    parts = list(seg.iter_segments(video))
    assert [p.shape[0] for p in parts] == [hi - lo for lo, hi in seg.segment_bounds(video)]
    assert video.metadata.frames_indices == meta1.frames_indices
    assert torch.equal(torch.cat(parts), frames1)


@requires_torchcodec
def test_segment_preprocessing_matches_whole_video(h264_video):
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
        Qwen3VLVideoProcessor,
    )

    vp = Qwen3VLVideoProcessor()
    loader = VideoLoader(vp, device="cpu", num_segments=3)
    video = loader.open(h264_video)
    parts = list(loader.iter_segments(video))
    whole = vp(
        videos=[torch.cat(parts)],
        video_metadata=[video.metadata],
        do_sample_frames=False,
        return_tensors="pt",
    )
    meta = video.metadata
    size = video_target_size(vp, len(meta.frames_indices), meta.height, meta.width)
    outs = [preprocess_segment(vp, p, size) for p in parts]
    assert torch.equal(
        torch.cat([o["pixel_values_videos"] for o in outs]), whole["pixel_values_videos"]
    )
    grid = whole["video_grid_thw"][0].tolist()
    assert grid[1:] == [size[0] // vp.patch_size, size[1] // vp.patch_size]
    assert sum(int(o["video_grid_thw"][0, 0]) for o in outs) == grid[0]


def test_hf_fallback_accepts_file_url(h264_video, monkeypatch):
    # Without torchcodec the loader falls back to transformers' PyAV loader,
    # which only takes plain paths / http URLs.
    monkeypatch.setattr(video_io, "torchcodec_available", lambda: False)
    loader = VideoLoader(_SamplingProcessor(), device="cpu")
    frames, meta = loader.load("file://" + h264_video)
    assert len(frames) == KEEP and len(meta.frames_indices) == KEEP
