"""Custom NVLink P2P all-reduce for small TP messages.

Plugs in front of NCCL's RING_LL kernel for the (TP, small-message) regime
where the collective is bound by per-launch overhead rather than bandwidth.
Profile of Qwen3-0.6B (hidden=1024) TP=2 showed ~6.5 us per NCCL AR for
~64 KB messages plus a ~3.5 us host-side launch -- a two-shot P2P kernel
hits ~3 us end-to-end on NVLink-connected H100s. With 28 layers x 2 ARs
per forward this compounds to a real chunk of forward time
(~35 % of GPU time in the baseline profile).

The kernel itself lives in sgl-kernel (``torch.ops.sgl_kernel.all_reduce``).
This module is the Python orchestration: cudaMalloc + cudaIpcGetMemHandle
the shared metadata/staging buffers, exchange those handles across TP
ranks via byte-tensor all_gather over the existing NCCL group (no gloo
group needed), init the kernel-side allreduce object, and provide a
``capture()`` context manager for CUDA-graph integration.

References:
- sgl_kernel.allreduce (the underlying kernel + Python bindings)
- sglang/srt/distributed/device_communicators/custom_all_reduce.py
- vllm/distributed/device_communicators/custom_all_reduce.py
"""

import ctypes
import os
import pickle
from contextlib import contextmanager
from typing import List, Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

try:
    from sgl_kernel.allreduce import (
        all_reduce as _kernel_all_reduce,
        dispose as _kernel_dispose,
        get_graph_buffer_ipc_meta as _kernel_get_graph_buffer_ipc_meta,
        init_custom_ar as _kernel_init_custom_ar,
        meta_size as _kernel_meta_size,
        register_buffer as _kernel_register_buffer,
        register_graph_buffers as _kernel_register_graph_buffers,
    )

    _CUSTOM_AR_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when sgl-kernel is absent
    _CUSTOM_AR_AVAILABLE = False

from logger import logger

from gllm.distributed.cuda_wrapper import CudaRTLibrary, cudaIpcMemHandle_t


_IPC_HANDLE_BYTES = 128  # cudaIpcMemHandle_t is always 128 bytes


def _all_gather_bytes(
    payload: bytes, group: ProcessGroup, device: torch.device
) -> List[bytes]:
    """All-gather a fixed-length byte payload across ``group``.

    We avoid ``dist.all_gather_object`` here because gLLM only initializes
    the NCCL backend and ``all_gather_object`` then pickles through GPU
    intermediaries with known edge cases under ``inference_mode`` (see
    https://github.com/pytorch/pytorch/issues/126032). Going via a raw
    ``uint8`` GPU tensor is reliable on the NCCL backend and is also
    cheap -- each rank ships at most a few hundred bytes during init.
    """
    world_size = dist.get_world_size(group=group)
    n = len(payload)
    # ``torch.frombuffer`` keeps the underlying ``bytearray`` alive, but the
    # ``.to(device)`` clones it onto the GPU so the original buffer can be
    # freed immediately after the call returns.
    src = torch.frombuffer(bytearray(payload), dtype=torch.uint8).to(device)
    gathered = torch.empty(world_size * n, dtype=torch.uint8, device=device)
    dist.all_gather_into_tensor(gathered, src, group=group)
    raw = bytes(gathered.cpu().numpy().tobytes())
    return [raw[i * n : (i + 1) * n] for i in range(world_size)]


class CustomAllreduce:
    """NVLink P2P fast path for small TP all-reduces.

    Constructor params:
    - ``device``: the local CUDA device this rank lives on.
    - ``group``: the TP ``ProcessGroup`` (NCCL-backed). All handle
      exchange happens via byte tensors over this group so we don't
      need a separate gloo group.
    - ``rank`` / ``world_size``: TP rank/size within ``group``.
    - ``max_size``: maximum AR message size in bytes that we'll handle
      via the custom kernel. Larger messages fall back to NCCL. Default
      8 MB matches sglang/vLLM and is well above typical TP small-AR
      sizes (per-layer hidden tensor in Qwen3-0.6B = 64 KB).

    The instance is ``disabled = True`` whenever any precondition fails
    (missing sgl-kernel, unsupported world size, no NVLink, P2P access
    denied, etc.). In that case callers should fall back to NCCL.
    """

    _SUPPORTED_WORLD_SIZES = (2, 4, 6, 8)
    _DEFAULT_MAX_SIZE = 8 * 1024 * 1024  # 8 MB

    def __init__(
        self,
        device: torch.device,
        group: ProcessGroup,
        rank: int,
        world_size: int,
        max_size: Optional[int] = None,
        full_nvlink: bool = True,
    ) -> None:
        self.disabled = True
        self._capturing = False
        self._meta_ptrs: List[int] = []
        self._buffer_ptrs: List[int] = []
        # Local ``cudaMalloc`` results we own and must ``cudaFree`` on
        # close. Tracked separately from ``_meta_ptrs`` / ``_buffer_ptrs``
        # because those also contain peer-opened ``cudaIpcOpenMemHandle``
        # pointers that must be ``cudaIpcCloseMemHandle``'d instead.
        self._owned_local_ptrs: List[ctypes.c_void_p] = []
        self._opened_peer_ptrs: List[ctypes.c_void_p] = []
        self._rank_data: Optional[torch.Tensor] = None
        self._handle: int = 0

        if not _CUSTOM_AR_AVAILABLE:
            logger.warning(
                "Custom allreduce: sgl_kernel.allreduce not importable; "
                "falling back to NCCL."
            )
            return

        if world_size not in self._SUPPORTED_WORLD_SIZES:
            logger.warning(
                f"Custom allreduce: unsupported world_size={world_size}; "
                f"supported sizes are {self._SUPPORTED_WORLD_SIZES}. Falling "
                "back to NCCL."
            )
            return

        if not torch.cuda.is_available():
            return

        self.device = device
        self.group = group
        self.rank = rank
        self.world_size = world_size
        self.full_nvlink = full_nvlink
        self.max_size = max_size if max_size is not None else self._DEFAULT_MAX_SIZE

        # ``meta_size`` is the per-rank sync/header region. The remainder
        # (``max_size`` bytes) is the staging buffer that the kernel uses
        # for the second-shot of two-shot AR. We allocate one big slab so
        # the metadata and the staging always come from the same
        # contiguous IPC-shared region (kernel expects them packed).
        meta_size = _kernel_meta_size()
        try:
            self._meta_ptrs = self._create_shared_buffer(meta_size + self.max_size)
            self._buffer_ptrs = self._create_shared_buffer(self.max_size)
        except Exception as e:
            logger.warning(
                f"Custom allreduce: failed to allocate shared buffers "
                f"({e!r}); falling back to NCCL."
            )
            self._free_shared_buffers()
            return

        # ``rank_data`` is per-rank scratch used by the kernel for stashing
        # registered-buffer tuples (one tuple per CUDA-graph buffer set we
        # ever register). 8 MB easily covers thousands of buckets and is
        # well below our memory budget.
        self._rank_data = torch.empty(
            8 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )

        try:
            self._handle = _kernel_init_custom_ar(
                self._meta_ptrs, self._rank_data, self.rank, self.full_nvlink
            )
            _kernel_register_buffer(self._handle, self._buffer_ptrs)
        except Exception as e:
            logger.warning(
                f"Custom allreduce: kernel init failed ({e!r}); falling "
                "back to NCCL."
            )
            self._free_shared_buffers()
            self._rank_data = None
            return

        self.disabled = False
        logger.info(
            f"Custom allreduce enabled (rank={self.rank}/{self.world_size}, "
            f"full_nvlink={self.full_nvlink}, max_size={self.max_size}B)."
        )

    # -------------------------------------------------------------- buffers
    def _create_shared_buffer(self, size_in_bytes: int) -> List[int]:
        """Allocate ``size_in_bytes`` on this rank, share via cudaIPC.

        Returns a length-``world_size`` list of device pointers (ints):
        the entry at index ``self.rank`` is the local cudaMalloc result;
        the others are ``cudaIpcOpenMemHandle`` results from the peer
        ranks' handles. The kernel then walks this table by ``rank`` to
        find each peer's metadata/staging region.
        """
        lib = CudaRTLibrary()
        local_ptr = lib.cudaMalloc(size_in_bytes)
        lib.cudaMemset(local_ptr, 0, size_in_bytes)

        local_handle = lib.cudaIpcGetMemHandle(local_ptr)
        local_bytes = bytes(local_handle.internal)
        gathered = _all_gather_bytes(local_bytes, self.group, self.device)

        ptrs: List[int] = []
        for i, h in enumerate(gathered):
            if i == self.rank:
                ptrs.append(local_ptr.value)
                continue
            peer_handle = cudaIpcMemHandle_t()
            ctypes.memmove(
                peer_handle.internal, h, _IPC_HANDLE_BYTES
            )
            peer_ptr = lib.cudaIpcOpenMemHandle(peer_handle)
            ptrs.append(peer_ptr.value)
            self._opened_peer_ptrs.append(peer_ptr)
        # Remember the local pointer so we can ``cudaFree`` it on shutdown.
        self._owned_local_ptrs.append(local_ptr)
        return ptrs

    def _free_shared_buffers(self) -> None:
        lib = CudaRTLibrary()
        for ptr in self._opened_peer_ptrs:
            try:
                lib.cudaIpcCloseMemHandle(ptr)
            except Exception:
                pass
        self._opened_peer_ptrs.clear()
        for ptr in self._owned_local_ptrs:
            try:
                lib.cudaFree(ptr)
            except Exception:
                pass
        self._owned_local_ptrs.clear()
        self._meta_ptrs = []
        self._buffer_ptrs = []

    # -------------------------------------------------- API used by callers
    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        """Return True iff this tensor is eligible for the custom kernel.

        Kernel constraints:
        - byte size must be a multiple of 16 (vectorized loads);
        - tensor must be (weakly) contiguous in memory;
        - byte size must be <= ``self.max_size``.
        """
        if self.disabled:
            return False
        nbytes = inp.numel() * inp.element_size()
        if nbytes == 0 or nbytes % 16 != 0:
            return False
        if not inp.is_contiguous():
            # contiguous() would copy; if the caller passed a non-contig
            # tensor we let NCCL handle it (it does an internal staging).
            return False
        return nbytes <= self.max_size

    def all_reduce(
        self,
        inp: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the custom AR kernel; out-of-place.

        Two regimes:
        - ``inp`` is being captured into a CUDA graph (``_capturing``
          set AND current stream is capturing): use the *registered*
          path (``reg_buffer=0``). The kernel records ``inp.data_ptr()``
          / ``out.data_ptr()`` as cross-rank addresses; the
          ``register_graph_buffers`` call we make on capture exit then
          broadcasts the IPC handles so the kernel can resolve those
          local addresses to peer pointers at replay time. Saves one
          ``cudaMemcpyAsync(inp -> staging)`` per AR (~2 % extra
          throughput on decode-heavy workloads on top of the eager
          custom-AR win).
        - Otherwise (eager, including pre-capture warmup): use the
          *unregistered* / staging-buffer path. The kernel first stages
          ``inp`` into ``self._buffer_ptrs[rank]`` (an ``max_size``-byte
          IPC slot we already shared at ``__init__``) before reducing.
          Works for any contiguous tensor without per-call setup.

        Both paths assume ``inp`` is contiguous and ``inp.nbytes`` is
        a multiple of 16 -- ``should_custom_ar`` is the caller-side
        gate that filters out the rest.
        """
        if out is None:
            out = torch.empty_like(inp)
        # ``is_current_stream_capturing`` is a cheap pair of cudart
        # calls; we still short-circuit on the ``_capturing`` Python
        # flag so the hot eager path is a single bool check.
        use_registered = (
            self._capturing and torch.cuda.is_current_stream_capturing()
        )
        if use_registered:
            _kernel_all_reduce(self._handle, inp, out, 0, 0)
        else:
            _kernel_all_reduce(
                self._handle,
                inp,
                out,
                self._buffer_ptrs[self.rank],
                self.max_size,
            )
        return out

    @contextmanager
    def capture(self):
        """Mark a CUDA-graph capture region.

        While inside this context (``_capturing=True``), ``all_reduce``
        switches to the *registered* path -- the kernel reads / writes
        ``inp.data_ptr()`` directly cross-rank, skipping the
        ``cudaMemcpyAsync(inp -> staging)`` that the eager path needs.
        This saves ~30 % per AR for the tiny decode-time messages.
        On context exit we call ``register_graph_buffers``, which
        broadcasts the captured-buffer IPC handles so the kernel can
        translate those captured local addresses into cross-rank
        pointers at replay time.

        If ``register_graph_buffers`` fails we **must** disable custom
        AR -- the graphs were captured with ``reg_buffer=0`` so replay
        would dereference unmapped peer pointers. In that case all
        future ARs go through NCCL on the *eager* path (the captured
        graphs are still safe because they go through the registered
        kernel which now has the pre-init ``register_buffer`` IPC set;
        but to be on the safe side we disable the whole instance and
        let callers rebuild graphs from a clean state on next start).
        """
        if self.disabled:
            yield
            return
        self._capturing = True
        try:
            yield
        finally:
            self._capturing = False
            # The graphs we just captured use the registered AR kernel
            # (``reg_buffer=0``), so they need the cross-rank IPC table
            # built by ``register_graph_buffers`` to be safe to replay.
            # If the handshake fails we have to bail loudly -- silently
            # disabling custom AR wouldn't unbake the kernel from the
            # captured graphs, so replay would crash later with a much
            # uglier error.
            self._register_graph_buffers()

    def _register_graph_buffers(self) -> None:
        """After cuda-graph capture, share captured-buffer IPC handles.

        ``get_graph_buffer_ipc_meta`` returns a pair of Python lists --
        ``(handles, offsets)`` -- describing every buffer the AR kernel
        saw in the captured graph(s). The exact element types are
        opaque (the kernel may return raw pointer-sized ints in
        ``handles`` or a packed byte-list; sgl-kernel changed this once
        already), so rather than reaching into the structure we pickle
        the whole ``(handles, offsets)`` tuple and exchange the pickled
        bytes across ranks via byte-tensor all-gather. This mirrors how
        vLLM/sglang use ``broadcast_object_list`` (which is just pickle
        under the hood) and is robust to layout changes in the kernel.
        """
        local_handles, local_offsets = _kernel_get_graph_buffer_ipc_meta(
            self._handle
        )
        payload = pickle.dumps((local_handles, local_offsets))

        # Per-rank pickle sizes may differ slightly (different small-int
        # caches etc.); pad to the max so the all-gather stays rectangular.
        local_len = len(payload)
        max_len_t = torch.tensor([local_len], dtype=torch.int64, device=self.device)
        dist.all_reduce(max_len_t, op=dist.ReduceOp.MAX, group=self.group)
        max_len = int(max_len_t.item())
        framed = local_len.to_bytes(8, "little") + payload
        if len(framed) < max_len + 8:
            framed = framed + b"\x00" * (max_len + 8 - len(framed))
        gathered = _all_gather_bytes(framed, self.group, self.device)

        all_handles: List = []
        all_offsets: List = []
        for buf in gathered:
            real_len = int.from_bytes(buf[:8], "little")
            handles_i, offsets_i = pickle.loads(buf[8 : 8 + real_len])
            all_handles.append(handles_i)
            all_offsets.append(offsets_i)
        _kernel_register_graph_buffers(self._handle, all_handles, all_offsets)

    # ------------------------------------------------------------- shutdown
    def close(self) -> None:
        if self.disabled and self._handle == 0:
            return
        if self._handle != 0:
            try:
                _kernel_dispose(self._handle)
            except Exception:
                pass
            self._handle = 0
        self._free_shared_buffers()
        self._rank_data = None
        self.disabled = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ------------------------------------------------------------------- module-level


_CUSTOM_AR: Optional[CustomAllreduce] = None


def init_custom_allreduce(
    device: torch.device,
    group: ProcessGroup,
    rank: int,
    world_size: int,
    max_size: Optional[int] = None,
    full_nvlink: bool = True,
) -> Optional[CustomAllreduce]:
    """Initialize the process-global custom AR instance.

    Idempotent: calling twice on the same process is a no-op (we return
    the already-built instance). Returns ``None`` if creation failed --
    in that case callers should keep using NCCL.
    """
    global _CUSTOM_AR
    if _CUSTOM_AR is not None:
        return _CUSTOM_AR
    if os.environ.get("GLLM_DISABLE_CUSTOM_AR", "0") == "1":
        logger.info("Custom allreduce disabled by GLLM_DISABLE_CUSTOM_AR=1.")
        return None
    car = CustomAllreduce(
        device=device,
        group=group,
        rank=rank,
        world_size=world_size,
        max_size=max_size,
        full_nvlink=full_nvlink,
    )
    if car.disabled:
        return None
    _CUSTOM_AR = car
    return car


def get_custom_allreduce() -> Optional[CustomAllreduce]:
    return _CUSTOM_AR


def shutdown_custom_allreduce() -> None:
    global _CUSTOM_AR
    if _CUSTOM_AR is not None:
        _CUSTOM_AR.close()
        _CUSTOM_AR = None
