"""Cross-request pipelined encoder: plan -> decode -> ViT -> NIXL, overlapped.

Serving one item at a time (decode, preprocess, ViT, NIXL write, wait) makes
the decoders, the GPU and the transport take turns under concurrency (measured:
~25-30% lower throughput at 4-16 concurrent videos). This module runs those
stages concurrently across
requests, at the granularity of *segments* (a time slice of a video, or a
whole image):

    main thread    ZMQ job intake, MmItemMeta out, LM page targets in, every
                   NIXL call (posting writes, polling completions, in-order
                   notifications)
    plan pool      open the video container / run the image processor; yields
                   the grid (token count) and content hash -- no decoding
    decode workers pop the highest-priority pending segment of *any* request,
                   decode + preprocess it (NVDEC/CPU), hand it to the GPU stage
    GPU thread     pops the highest-priority decoded segment, runs the ViT and
                   stages the rows in pages of a registered send buffer

Segments of one item may finish out of order; the LM is notified of each
item's landed rows strictly as a growing prefix (``embp`` partial notifs, then
the final ``emb`` notif), which is what its in-order prefill gate needs.
Scheduling is FCFS by item arrival, then segment order.
"""

from __future__ import annotations

import heapq
import itertools
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from logger import logger

from gllm.disagg.paging import embed_layout, num_pages, row_index
from gllm.runtime.cache_arena import CacheArena


class PagedStaging:
    """Paged staging rows over the registered send buffer.

    The buffer is a :class:`CacheArena` of fixed-size row pages, so an item or
    segment takes any free pages (no contiguous run, no fragmentation).
    Allocations block until enough pages are free, which also back-pressures
    the GPU stage when the transport (or the LM's page supply) falls behind.
    """

    def __init__(self, backing: torch.Tensor, dtype: torch.dtype, feat_dim: int,
                 rows_per_page: int):
        layout = embed_layout("staging", feat_dim, dtype, rows_per_page)
        page_bytes = layout.entry_bytes
        if page_bytes != rows_per_page * feat_dim * torch.empty((), dtype=dtype).element_size():
            raise ValueError("staging rows must pack pages exactly")
        self.arena = CacheArena(backing, page_bytes)
        self.cache = self.arena.register_cache(layout)
        self.rows_per_page = rows_per_page
        self.num_pages = self.cache.num_slots
        self.capacity = self.num_pages * rows_per_page
        # Page ``p`` holds flat rows ``[p * rows_per_page, (p + 1) * rows_per_page)``.
        self.rows = backing.view(dtype).view(-1, feat_dim)
        self.row_bytes = self.rows.stride(0) * self.rows.element_size()
        self._cv = threading.Condition()

    def alloc(self, rows: int) -> List[int]:
        if rows > self.capacity:
            raise ValueError(f"{rows} rows exceed staging capacity {self.capacity}")
        n = num_pages(rows, self.rows_per_page)
        with self._cv:
            while True:
                pages = self.arena.allocator.allocate("staging", n)
                if pages is not None:
                    return pages
                self._cv.wait()

    def free(self, pages: List[int]) -> None:
        with self._cv:
            self.arena.allocator.free("staging", pages)
            self._cv.notify_all()

    def row_index(self, pages: List[int], rows: int) -> torch.Tensor:
        """Flat row ids of rows ``[0, rows)`` staged in ``pages``."""
        page, off = row_index(pages, rows, self.rows_per_page)
        return torch.tensor(page) * self.rows_per_page + torch.tensor(off)

    @property
    def free_rows(self) -> int:
        with self._cv:
            return self.arena.allocator.num_free_slots("staging") * self.rows_per_page


class _PriorityQueue:
    """Blocking min-heap keyed by ``(priority, tiebreak)``."""

    def __init__(self):
        self._heap: list = []
        self._cv = threading.Condition()
        self._count = itertools.count()
        self._closed = False

    def put(self, priority, item) -> None:
        with self._cv:
            heapq.heappush(self._heap, (priority, next(self._count), item))
            self._cv.notify()

    def get(self):
        """Highest-priority item, or ``None`` once closed and drained."""
        with self._cv:
            while not self._heap and not self._closed:
                self._cv.wait()
            if not self._heap:
                return None
            return heapq.heappop(self._heap)[2]

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()


@dataclass
class WorkItem:
    """One EncoderJob flowing through the pipeline."""

    order: int  # arrival order (FCFS priority)
    job: object  # EncoderJob
    num_tokens: int
    chash: bytes
    grid_thw: Optional[torch.Tensor] = None
    plan: Optional[dict] = None  # segment-streamed video plan
    mm_input: Optional[dict] = None  # whole-item ViT input (image / fallback)
    cached: Optional[torch.Tensor] = None  # embed-cache hit
    seg_bounds: List[Tuple[int, int]] = field(default_factory=list)  # frame ranges
    seg_rows: List[Tuple[int, int]] = field(default_factory=list)  # embedding rows
    seg_done: List[bool] = field(default_factory=list)
    seg_embeds: Dict[int, torch.Tensor] = field(default_factory=dict)
    notified_rows: int = 0
    failed: bool = False
    t_arrive: float = 0.0

    @property
    def num_segments(self) -> int:
        return len(self.seg_rows)

    def priority(self, seg: int) -> Tuple[int, int]:
        return (self.order, seg)

    def done_prefix_rows(self) -> int:
        """Rows covered by the leading run of finished segments."""
        rows = 0
        for done, (_, hi) in zip(self.seg_done, self.seg_rows):
            if not done:
                break
            rows = hi
        return rows


def video_segment_rows(plan: dict, bounds, runner) -> List[Tuple[int, int]]:
    """Embedding-row range of each frame segment of a streamed video."""
    vp = runner.video_processor
    tps = vp.temporal_patch_size
    _, gh, gw = (int(x) for x in plan["grid_thw"][0])
    rows_per_t = gh * gw // (runner.spatial_merge_size**2)
    out, off = [], 0
    for lo, hi in bounds:
        n = -(-(hi - lo) // tps) * rows_per_t
        out.append((off, off + n))
        off += n
    return out


class EncoderPipeline:
    """Runs the plan / decode / GPU stages for :class:`gllm.engine.encoder.Encoder`.

    The owning encoder keeps the main loop (ZMQ, discovery, NIXL) and calls
    :meth:`submit`, :meth:`poll_planned` and :meth:`poll_staged` from it.
    """

    def __init__(
        self,
        runner,
        staging: PagedStaging,
        *,
        decode_workers: int = 8,
        plan_workers: int = 4,
        max_ready_segments: Optional[int] = None,
    ):
        self.runner = runner
        self.staging = staging
        self.device = staging.rows.device
        self._order = itertools.count()
        self._plan_pool = ThreadPoolExecutor(plan_workers, thread_name_prefix="enc-plan")
        self._planned: List[Tuple[object, Future]] = []
        self.dropped: List[object] = []  # jobs whose plan failed
        self.failed: "queue.Queue[WorkItem]" = queue.Queue()  # failed in decode / ViT
        self._decode_q = _PriorityQueue()
        self._ready_q = _PriorityQueue()
        self._staged: "queue.Queue[tuple]" = queue.Queue()
        self._lock = threading.Lock()
        self._outstanding = 0  # segments started but not yet staged/dropped
        # Decoded-but-not-encoded segments hold frames / pixel values (on the
        # GPU with NVDEC); bound them so a burst cannot exhaust device memory.
        self._ready_budget = threading.BoundedSemaphore(
            max_ready_segments or 2 * decode_workers
        )
        self._threads = [
            threading.Thread(target=self._decode_worker, name=f"enc-decode-{i}", daemon=True)
            for i in range(decode_workers)
        ]
        self._threads.append(
            threading.Thread(target=self._gpu_worker, name="enc-gpu", daemon=True)
        )
        for t in self._threads:
            t.start()

    # ------------------------------------------------------------------
    # main thread API
    # ------------------------------------------------------------------
    def submit(self, job) -> None:
        """Start planning a job (container open / image processor)."""
        self._planned.append((job, self._plan_pool.submit(self._plan, job)))

    def poll_planned(self) -> List[WorkItem]:
        """Planned items, in submission order; the caller sends their meta and
        then calls :meth:`start`. Failed plans are logged and their jobs
        appended to :attr:`dropped`."""
        out = []
        while self._planned and self._planned[0][1].done():
            job, fut = self._planned.pop(0)
            try:
                out.append(fut.result())
            except Exception as e:
                self.dropped.append(job)
                logger.error(
                    f"[encoder] job seq={job.seq_id} item={job.item_idx} dropped "
                    f"in plan: {type(e).__name__}: {e}"
                )
        return out

    def start(self, w: WorkItem) -> None:
        """Queue the item's segments for decoding (or straight for the GPU)."""
        with self._lock:
            self._outstanding += w.num_segments
        if w.plan is not None and w.cached is None:
            for seg in range(w.num_segments):
                self._decode_q.put(w.priority(seg), (w, seg))
        else:
            # Already-prepared input: no frames to hold, so it bypasses the
            # ready budget (the main thread must never block on it).
            self._ready_q.put(w.priority(0), (w, 0, None, None, False))

    def poll_staged(self) -> List[tuple]:
        """``(work, seg, staging_pages, rows)`` whose embedding rows are in
        the send buffer and visible to the transport."""
        out = []
        while True:
            try:
                out.append(self._staged.get_nowait())
            except queue.Empty:
                return out

    def segment_written(self, w: WorkItem, seg: int, pages: List[int]) -> None:
        """The segment's NIXL writes landed: recycle its staging pages."""
        self.staging.free(pages)
        w.seg_done[seg] = True

    @property
    def busy(self) -> bool:
        """Anything planned, decoding, encoding or staged."""
        return bool(self._planned) or self._outstanding > 0 or not self._staged.empty()

    def _segment_left_pipeline(self) -> None:
        with self._lock:
            self._outstanding -= 1

    def close(self) -> None:
        self._decode_q.close()
        self._ready_q.close()
        self._plan_pool.shutdown(wait=False, cancel_futures=True)

    def _bind_device(self) -> None:
        # The CUDA current device is per thread; pool threads start on GPU 0.
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

    # ------------------------------------------------------------------
    # plan pool
    # ------------------------------------------------------------------
    def _plan(self, job) -> WorkItem:
        self._bind_device()
        with torch.inference_mode():
            r = self.runner
            plan = r.open_video_stream(job.content) if job.modality == "video" else None
            if plan is not None:
                w = WorkItem(
                    order=next(self._order), job=job, plan=plan,
                    grid_thw=plan["grid_thw"],
                    num_tokens=r.num_vis_tokens(plan["grid_thw"]),
                    chash=plan["content_hash"], t_arrive=time.monotonic(),
                )
                cached = r.mm_embed_cache.get(w.chash)
                if cached is not None:
                    w.cached = cached[0]
                    w.seg_rows = [(0, w.num_tokens)]
                else:
                    w.seg_bounds = r.video_segment_bounds(plan)
                    w.seg_rows = video_segment_rows(plan, w.seg_bounds, r)
            else:
                mm_input, grid_thw = r.run_processor(job.content, job.modality)
                w = WorkItem(
                    order=next(self._order), job=job, mm_input=mm_input,
                    grid_thw=grid_thw,
                    num_tokens=r.num_vis_tokens(grid_thw),
                    chash=r.content_hash(mm_input, grid_thw), t_arrive=time.monotonic(),
                )
                w.seg_rows = [(0, w.num_tokens)]
            w.seg_done = [False] * w.num_segments
            return w

    # ------------------------------------------------------------------
    # decode workers
    # ------------------------------------------------------------------
    def _decode_worker(self) -> None:
        self._bind_device()
        with torch.inference_mode():
            while True:
                task = self._decode_q.get()
                if task is None:
                    return
                w, seg = task
                if w.failed:
                    self._segment_left_pipeline()
                    continue
                self._ready_budget.acquire()
                try:
                    lo, hi = w.seg_bounds[seg]
                    mm = self.runner.prepare_video_segment(w.plan, lo, hi)
                    event = None
                    if mm["pixel_values_videos"].is_cuda:
                        event = torch.cuda.Event()
                        event.record()
                    self._ready_q.put(w.priority(seg), (w, seg, mm, event, True))
                except Exception as e:
                    self._ready_budget.release()
                    self._segment_left_pipeline()
                    self._fail(w, f"decode segment {seg}", e)

    # ------------------------------------------------------------------
    # GPU thread
    # ------------------------------------------------------------------
    def _gpu_worker(self) -> None:
        self._bind_device()
        with torch.inference_mode():
            while True:
                task = self._ready_q.get()
                if task is None:
                    return
                w, seg, mm, event, budgeted = task
                try:
                    if w.failed:
                        continue
                    if event is not None:
                        event.wait()
                    if w.cached is not None:
                        vis = w.cached
                    elif w.plan is not None:
                        vis = self.runner.encode_segment(mm)
                    else:
                        vis = self.runner.encode(w.mm_input, w.chash)
                        w.mm_input = None
                    lo, hi = w.seg_rows[seg]
                    if vis.shape[0] != hi - lo:
                        raise RuntimeError(
                            f"segment {seg} produced {vis.shape[0]} rows, planned {hi - lo}"
                        )
                    if w.plan is not None and w.cached is None:
                        w.seg_embeds[seg] = vis
                        if len(w.seg_embeds) == w.num_segments:
                            full = torch.cat([w.seg_embeds[i] for i in range(w.num_segments)])
                            self.runner.mm_embed_cache.put(w.chash, (full,))
                            w.seg_embeds = {}
                    rows = hi - lo
                    pages = self.staging.alloc(rows)
                    st = self.staging
                    st.rows.index_copy_(
                        0, st.row_index(pages, rows).to(self.device), vis.to(st.rows.dtype)
                    )
                    # NIXL reads the raw device pointer outside CUDA stream
                    # order; make the staged rows visible before handing off.
                    if self.device.type == "cuda":
                        torch.cuda.current_stream().synchronize()
                    self._staged.put((w, seg, pages, rows))
                except Exception as e:
                    self._fail(w, f"encode segment {seg}", e)
                finally:
                    if budgeted:
                        self._ready_budget.release()
                    self._segment_left_pipeline()

    def _fail(self, w: WorkItem, what: str, e: Exception) -> None:
        if not w.failed:
            w.failed = True
            w.seg_embeds = {}
            self.failed.put(w)
            logger.error(
                f"[encoder] job seq={w.job.seq_id} item={w.job.item_idx} failed "
                f"({what}): {type(e).__name__}: {e}; LM will re-dispatch"
            )
