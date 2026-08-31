"""Breakable piecewise CUDA graphs for dynamic mixed forwards.

Static model regions are captured into CUDA graph segments. Attention/GDN
modules are eager breaks which consume only the real token prefix and publish
their result into a persistent bucket-sized tensor. The next graph segment
therefore sees a stable address even though the dynamic core ran eagerly.
"""

from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from logger import logger


class PiecewiseCapture:
    """A sequence of CUDA graphs interleaved with eager callables."""

    def __init__(self, graph_pool: list | None = None):
        # All piecewise graphs may use one pool when replay is serialized on
        # one stream and no graph-owned tensor is live across a segment. The
        # runner enforces the latter by copying every eager input, passthrough,
        # and final output into persistent external buffers. This is the same
        # lifetime rule used by vLLM's global CUDA graph pool: alternative
        # bucket graphs may then reuse scratch addresses without retaining the
        # sum of every bucket's peak activation footprint.
        self.graph_pool = graph_pool if graph_pool is not None else []
        self.segments: list[Callable[[], object]] = []
        self.num_graphs = 0
        self.num_eager_breaks = 0
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._capturing = False

    def _begin(self):
        graph = torch.cuda.CUDAGraph()
        pool = self.graph_pool[0] if self.graph_pool else None
        graph.capture_begin(pool=pool)
        self._graph = graph
        self._capturing = True

    def _end(self):
        if not self._capturing:
            return
        assert self._graph is not None
        self._graph.capture_end()
        if not self.graph_pool:
            self.graph_pool.append(self._graph.pool())
        # Do not replay during capture. TP graph segments may contain the
        # repository's registered custom all-reduce, whose cross-rank graph
        # buffers are exchanged only after the outer capture context closes.
        # Replaying here dereferences unregistered peer pointers and can
        # segfault. Adjacent segments are connected through persistent
        # placeholder/boundary buffers, so their values need not be initialized
        # while kernels are merely being recorded.
        self.segments.append(self._graph.replay)
        self.num_graphs += 1
        self._graph = None
        self._capturing = False

    def __enter__(self):
        self._begin()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._end()

    def add_eager(
        self,
        fn: Callable[[], object],
        *,
        capture_result: object | None = None,
    ):
        self._end()
        # Attention/SSM is intentionally not executed while graphs are being
        # captured. Its output shape matches the hidden-state input, so the
        # runtime can supply a persistent zero buffer to connect the adjacent
        # static segments. The callable itself remains in the replay sequence
        # and executes with real metadata and the real token count at runtime.
        # This avoids an O(sequence_length^2) prefill attention pass for every
        # startup bucket.
        result = fn() if capture_result is None else capture_result
        self.segments.append(fn)
        self.num_eager_breaks += 1
        self._begin()
        return result

    def replay(self):
        for segment in self.segments:
            segment()


class PiecewiseRuntime:
    """Thread-local layer hook active during warmup/capture."""

    _tls = threading.local()

    def __init__(
        self,
        bucket: int,
        num_tokens: int,
        *,
        warmup: bool = False,
        workspace_tokens: Optional[int] = None,
        workspace_token_sizes: Optional[list[int]] = None,
        break_buffers: Optional[list[torch.Tensor]] = None,
        break_input_buffers: Optional[list[torch.Tensor]] = None,
        passthrough_buffers: Optional[list[torch.Tensor]] = None,
        workspaces: Optional[dict] = None,
    ):
        self.bucket = bucket
        self.num_tokens = num_tokens
        self.warmup = warmup
        self.workspace_tokens = int(workspace_tokens or bucket)
        self.workspace_token_sizes = tuple(
            int(size) for size in (workspace_token_sizes or [self.workspace_tokens])
        )
        self.capture: Optional[PiecewiseCapture] = None
        self.break_buffers = break_buffers if break_buffers is not None else []
        self.break_input_buffers = (
            break_input_buffers if break_input_buffers is not None else []
        )
        self.passthrough_buffers = (
            passthrough_buffers if passthrough_buffers is not None else []
        )
        self.workspaces = workspaces if workspaces is not None else {}
        self._break_index = 0

    @classmethod
    def current(cls) -> Optional["PiecewiseRuntime"]:
        return getattr(cls._tls, "current", None)

    @contextlib.contextmanager
    def activate(self):
        previous = getattr(self._tls, "current", None)
        if previous is not None:
            raise RuntimeError("nested piecewise CUDA graph runtime")
        self._tls.current = self
        try:
            yield self
        finally:
            self._tls.current = previous

    def dynamic_tensor(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor],
        x,
        *passthrough: torch.Tensor,
    ):
        """Execute a dynamic layer and bridge every live cross-boundary tensor."""
        if self.warmup:
            # Warm only the graph-resident regions without advancing KV/GDN.
            output = torch.zeros_like(x)
            return (output, *passthrough) if passthrough else output
        if self.capture is None:
            output = fn(x[: self.num_tokens])
            return (output, *passthrough) if passthrough else output

        break_index = self._break_index
        self._break_index += 1
        expected_shape = (self.workspace_tokens, *x.shape[1:])
        if self.break_input_buffers:
            input_buffer = self.break_input_buffers[0]
            if (
                input_buffer.shape != expected_shape
                or input_buffer.dtype != x.dtype
                or input_buffer.device != x.device
            ):
                raise RuntimeError(
                    "piecewise eager-input signature changed across buckets: "
                    f"break={break_index}, expected={expected_shape}/{x.dtype}/"
                    f"{x.device}, actual={tuple(input_buffer.shape)}/"
                    f"{input_buffer.dtype}/{input_buffer.device}"
                )
        else:
            input_buffer = torch.zeros(
                expected_shape,
                dtype=x.dtype,
                device=x.device,
            )
            self.break_input_buffers.append(input_buffer)
        boundary_input = input_buffer[: self.bucket]
        # Make the graph publish its boundary into one persistent address.
        # The eager callable then no longer retains the graph-owned ``x``
        # tensor. Segments sharing this bucket's CUDA graph pool can therefore
        # recycle internal activations instead of keeping one live output per
        # layer and token bucket.
        boundary_input.copy_(x)

        bridged_passthrough = []
        for index, value in enumerate(passthrough):
            passthrough_shape = (self.workspace_tokens, *value.shape[1:])
            if index < len(self.passthrough_buffers):
                passthrough_buffer = self.passthrough_buffers[index]
                if (
                    passthrough_buffer.shape != passthrough_shape
                    or passthrough_buffer.dtype != value.dtype
                    or passthrough_buffer.device != value.device
                ):
                    raise RuntimeError(
                        "piecewise passthrough signature changed: "
                        f"break={break_index}, slot={index}, "
                        f"expected={passthrough_shape}/{value.dtype}/"
                        f"{value.device}, actual="
                        f"{tuple(passthrough_buffer.shape)}/"
                        f"{passthrough_buffer.dtype}/{passthrough_buffer.device}"
                    )
            else:
                passthrough_buffer = torch.zeros(
                    passthrough_shape,
                    dtype=value.dtype,
                    device=value.device,
                )
                self.passthrough_buffers.append(passthrough_buffer)
            bridged = passthrough_buffer[: self.bucket]
            # Residual/skip tensors bypass the eager operator but remain live
            # in the following graph segment. Publish them just like the
            # dynamic operator input so a shared graph pool cannot overwrite
            # their storage before the consumer runs.
            bridged.copy_(value)
            bridged_passthrough.append(bridged)

        # Adjacent boundaries use different buffers; boundary i+2 may safely
        # reuse boundary i's storage because graph i+1 and its eager call are
        # ordered on the same replay stream. This bounds persistent eager
        # activations at two max-token buffers regardless of layer count.
        buffer_slot = break_index % 2
        if buffer_slot < len(self.break_buffers):
            output_buffer = self.break_buffers[buffer_slot]
            if (
                output_buffer.shape != expected_shape
                or output_buffer.dtype != x.dtype
                or output_buffer.device != x.device
            ):
                raise RuntimeError(
                    "piecewise eager-break signature changed across buckets: "
                    f"break={break_index}, expected={expected_shape}/{x.dtype}/"
                    f"{x.device}, actual={tuple(output_buffer.shape)}/"
                    f"{output_buffer.dtype}/{output_buffer.device}"
                )
        else:
            output_buffer = torch.zeros(
                (self.workspace_tokens, *x.shape[1:]),
                dtype=x.dtype,
                device=x.device,
            )
            self.break_buffers.append(output_buffer)
        output = output_buffer[: self.bucket]

        def eager_call():
            n = self.num_tokens
            value = fn(boundary_input[:n])
            if value.shape != output[:n].shape:
                raise RuntimeError(
                    "piecewise eager boundary must preserve hidden shape: "
                    f"input={tuple(boundary_input[:n].shape)}, "
                    f"output={tuple(value.shape)}"
                )
            output[:n].copy_(value)
            if n < self.bucket:
                output[n:].zero_()
            return output

        output = self.capture.add_eager(
            eager_call,
            capture_result=output,
        )
        if passthrough:
            return (output, *bridged_passthrough)
        return output


def piecewise_dynamic_tensor(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x,
    *passthrough: torch.Tensor,
):
    """Layer-side hook; eager outside a piecewise capture."""
    runtime = PiecewiseRuntime.current()
    if runtime is None:
        output = fn(x)
        return (output, *passthrough) if passthrough else output
    return runtime.dynamic_tensor(fn, x, *passthrough)


@dataclass
class _PiecewiseGraph:
    bucket: int
    static_input: torch.Tensor
    runtime: PiecewiseRuntime
    capture: PiecewiseCapture
    output: torch.Tensor


class PiecewiseGraphRunner:
    """Breakable graphs dispatched with the model runner's fixed buckets.

    A geometric ``capture_sizes`` table supplies the smallest captured size
    which can contain the real batch. All buckets are captured during model
    initialization; request-time execution is replay-only, with eager fallback
    if initialization did not produce a graph for a bucket.
    """

    def __init__(
        self,
        model: Callable,
        capture_sizes: list[int],
    ):
        self.model = model
        self.capture_sizes = sorted(
            {int(size) for size in capture_sizes if int(size) > 0}
        )
        self.size_to_graph: dict[int, _PiecewiseGraph] = {}
        self.graph_pool: list = []
        # Buckets replay serially, so they can safely alias one maximum-size
        # model input, one boundary input, two ping-pong eager outputs, one
        # passthrough per live skip tensor, and one final output. Since those
        # buffers externalize every value that crosses a segment boundary, all
        # buckets can share one graph scratch pool under serialized replay.
        self.static_input: Optional[torch.Tensor] = None
        self.break_buffers: list[torch.Tensor] = []
        self.break_input_buffers: list[torch.Tensor] = []
        self.passthrough_buffers: list[torch.Tensor] = []
        self.static_output: Optional[torch.Tensor] = None
        self.workspaces: dict = {}

    @staticmethod
    def build_capture_sizes(max_capture_size: int) -> list[int]:
        """Build CUDA Graph token buckets up to the full token budget.

        Keep fine-grained buckets where padding is most visible: exact tiny
        sizes, stride 8 below 256, stride 16 through 512, then four buckets per
        power-of-two interval. This is the original piecewise bucket policy;
        startup/memory optimizations must not trade away its steady-state
        padding efficiency.
        """
        max_capture_size = int(max_capture_size)
        if max_capture_size <= 0:
            return []
        sizes = [size for size in (1, 2, 4) if size <= max_capture_size]
        if max_capture_size >= 8:
            sizes.extend(range(8, min(max_capture_size + 1, 256), 8))
        dense_max = min(max_capture_size, 512)
        if dense_max >= 256:
            sizes.extend(range(256, dense_max + 1, 16))
        interval_start = 512
        while max_capture_size > interval_start:
            interval_end = min(max_capture_size, interval_start * 2)
            step = interval_start // 4
            sizes.extend(
                range(interval_start + step, interval_end + 1, step)
            )
            if sizes[-1] != interval_end:
                sizes.append(interval_end)
            interval_start *= 2
        return sorted(set(sizes))

    def bucket_for(self, num_tokens: int) -> Optional[int]:
        return next(
            (size for size in self.capture_sizes if size >= num_tokens), None
        )

    def can_run(self, num_tokens: int) -> bool:
        return self.bucket_for(num_tokens) is not None

    @torch.inference_mode()
    def capture_bucket(self, input_data, hidden_states: torch.Tensor):
        n = int(hidden_states.shape[0])
        bucket = self.bucket_for(n)
        if bucket is None:
            return None
        if n != bucket:
            raise ValueError(
                f"initial piecewise capture needs an exact bucket, got {n}->{bucket}"
            )
        if bucket in self.size_to_graph:
            return self.size_to_graph[bucket].output[:n]

        if self.static_input is None:
            self.static_input = torch.zeros(
                (self.capture_sizes[-1], hidden_states.shape[-1]),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        elif (
            self.static_input.shape[1:] != hidden_states.shape[1:]
            or self.static_input.dtype != hidden_states.dtype
            or self.static_input.device != hidden_states.device
        ):
            raise RuntimeError("piecewise model input signature changed")
        static_input = self.static_input[:bucket]
        static_input[:n].copy_(hidden_states)

        # Shape/JIT warmup without touching attention or recurrent state.
        warmup = PiecewiseRuntime(
            bucket,
            n,
            warmup=True,
            workspace_tokens=self.capture_sizes[-1],
            workspace_token_sizes=self.capture_sizes,
            break_buffers=self.break_buffers,
            break_input_buffers=self.break_input_buffers,
            passthrough_buffers=self.passthrough_buffers,
            workspaces=self.workspaces,
        )
        with warmup.activate():
            self.model(input_data, static_input)
        torch.cuda.synchronize()
        static_input[:n].copy_(hidden_states)
        if n < bucket:
            static_input[n:].zero_()

        runtime = PiecewiseRuntime(
            bucket,
            n,
            workspace_tokens=self.capture_sizes[-1],
            workspace_token_sizes=self.capture_sizes,
            break_buffers=self.break_buffers,
            break_input_buffers=self.break_input_buffers,
            passthrough_buffers=self.passthrough_buffers,
            workspaces=self.workspaces,
        )
        capture = PiecewiseCapture(graph_pool=self.graph_pool)
        runtime.capture = capture
        with runtime.activate(), capture:
            output = self.model(input_data, static_input)
            if self.static_output is None:
                self.static_output = torch.zeros(
                    (self.capture_sizes[-1], *output.shape[1:]),
                    dtype=output.dtype,
                    device=output.device,
                )
            static_output = self.static_output[:bucket]
            static_output.copy_(output)
        output = static_output

        graph = _PiecewiseGraph(bucket, static_input, runtime, capture, output)
        self.size_to_graph[bucket] = graph
        logger.debug(
            "Captured piecewise CUDA graph: token_bucket=%d real_tokens=%d "
            "graph_segments=%d eager_breaks=%d",
            bucket,
            n,
            capture.num_graphs,
            capture.num_eager_breaks,
        )
        return output[:n]

    @torch.inference_mode()
    def run(self, input_data, hidden_states: torch.Tensor):
        n = int(hidden_states.shape[0])
        bucket = self.bucket_for(n)
        if bucket is None:
            return None
        graph = self.size_to_graph.get(bucket)
        if graph is None:
            # All configured buckets are captured during model initialization.
            # A miss is a safe eager fallback, never an online capture.
            return None
        graph.runtime.num_tokens = n
        graph.static_input[:n].copy_(hidden_states)
        if n < bucket:
            graph.static_input[n:].zero_()
        graph.capture.replay()
        return graph.output[:n]
