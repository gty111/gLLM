"""Pure-Python ctypes wrapper around the cudart APIs we need for IPC.

Adapted (trimmed) from sglang's
``device_communicators/cuda_wrapper.py`` (itself derived from vLLM's
``cuda_wrapper.py``). We only need a handful of calls
(``cudaMalloc`` / ``cudaFree`` / ``cudaIpcGetMemHandle`` /
``cudaIpcOpenMemHandle`` / ``cudaMemset``) to set up the shared GPU
buffers that the custom-allreduce kernels read/write across ranks, and
inlining them via ``ctypes`` lets us avoid pulling in a separate C++
extension just for these.

``libcudart`` is loaded once via ``find_loaded_library`` (matching it
against ``/proc/self/maps``) so we always bind to the exact runtime
PyTorch already loaded -- mixing libcudart versions inside a process is
a recipe for cryptic ``cudaErrorInvalidValue`` returns.
"""

import ctypes
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


cudaError_t = ctypes.c_int
cudaMemcpyKind = ctypes.c_int


class cudaIpcMemHandle_t(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


@dataclass
class _Function:
    name: str
    restype: Any
    argtypes: List[Any]


def _find_loaded_library(lib_name: str) -> Optional[str]:
    """Locate an already-loaded shared library by scanning ``/proc/self/maps``.

    We require the library to already be in-process (PyTorch loads it for
    us). Returning ``None`` lets the caller raise a sensible error instead
    of trying to ``dlopen`` a random copy off the filesystem.
    """
    line = None
    with open("/proc/self/maps") as f:
        for raw in f:
            if lib_name in raw:
                line = raw
                break
    if line is None:
        return None
    start = line.index("/")
    path = line[start:].strip()
    filename = path.split("/")[-1]
    assert filename.rpartition(".so")[0].startswith(
        lib_name
    ), f"Unexpected filename: {filename} for library {lib_name}"
    return path


class CudaRTLibrary:
    """Thin ctypes wrapper exposing the cudart entrypoints we need.

    The ``exported_functions`` table mirrors the function declarations in
    the cudart headers; missing entries here are intentional -- this is
    not meant to be a general cudart binding.
    """

    exported_functions = [
        _Function("cudaSetDevice", cudaError_t, [ctypes.c_int]),
        _Function("cudaDeviceSynchronize", cudaError_t, []),
        _Function("cudaGetErrorString", ctypes.c_char_p, [cudaError_t]),
        _Function(
            "cudaMalloc",
            cudaError_t,
            [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t],
        ),
        _Function("cudaFree", cudaError_t, [ctypes.c_void_p]),
        _Function(
            "cudaMemset",
            cudaError_t,
            [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t],
        ),
        _Function(
            "cudaIpcGetMemHandle",
            cudaError_t,
            [ctypes.POINTER(cudaIpcMemHandle_t), ctypes.c_void_p],
        ),
        _Function(
            "cudaIpcOpenMemHandle",
            cudaError_t,
            [
                ctypes.POINTER(ctypes.c_void_p),
                cudaIpcMemHandle_t,
                ctypes.c_uint,
            ],
        ),
        _Function("cudaIpcCloseMemHandle", cudaError_t, [ctypes.c_void_p]),
    ]

    path_to_library_cache: Dict[str, Any] = {}
    path_to_dict_mapping: Dict[str, Dict[str, Any]] = {}

    def __init__(self, so_file: Optional[str] = None):
        import torch  # noqa: F401 - ensures libcudart is loaded

        if so_file is None:
            so_file = _find_loaded_library("libcudart")
            assert (
                so_file is not None
            ), "libcudart is not loaded; cannot bind cudart wrapper"
        if so_file not in CudaRTLibrary.path_to_library_cache:
            CudaRTLibrary.path_to_library_cache[so_file] = ctypes.CDLL(so_file)
        self.lib = CudaRTLibrary.path_to_library_cache[so_file]

        if so_file not in CudaRTLibrary.path_to_dict_mapping:
            funcs: Dict[str, Any] = {}
            for fn in CudaRTLibrary.exported_functions:
                f = getattr(self.lib, fn.name)
                f.restype = fn.restype
                f.argtypes = fn.argtypes
                funcs[fn.name] = f
            CudaRTLibrary.path_to_dict_mapping[so_file] = funcs
        self.funcs = CudaRTLibrary.path_to_dict_mapping[so_file]

    # ---- error handling --------------------------------------------------
    def _check(self, result: cudaError_t) -> None:
        if result != 0:
            err = self.funcs["cudaGetErrorString"](result).decode("utf-8")
            raise RuntimeError(f"CUDART error: {err}")

    # ---- thin wrappers ---------------------------------------------------
    def cudaSetDevice(self, device: int) -> None:
        self._check(self.funcs["cudaSetDevice"](device))

    def cudaDeviceSynchronize(self) -> None:
        self._check(self.funcs["cudaDeviceSynchronize"]())

    def cudaMalloc(self, size: int) -> ctypes.c_void_p:
        devPtr = ctypes.c_void_p()
        self._check(self.funcs["cudaMalloc"](ctypes.byref(devPtr), size))
        return devPtr

    def cudaFree(self, devPtr: ctypes.c_void_p) -> None:
        self._check(self.funcs["cudaFree"](devPtr))

    def cudaMemset(self, devPtr: ctypes.c_void_p, value: int, count: int) -> None:
        self._check(self.funcs["cudaMemset"](devPtr, value, count))

    def cudaIpcGetMemHandle(self, devPtr: ctypes.c_void_p) -> cudaIpcMemHandle_t:
        handle = cudaIpcMemHandle_t()
        self._check(self.funcs["cudaIpcGetMemHandle"](ctypes.byref(handle), devPtr))
        return handle

    def cudaIpcOpenMemHandle(self, handle: cudaIpcMemHandle_t) -> ctypes.c_void_p:
        # cudaIpcMemLazyEnablePeerAccess == 1
        devPtr = ctypes.c_void_p()
        self._check(
            self.funcs["cudaIpcOpenMemHandle"](ctypes.byref(devPtr), handle, 1)
        )
        return devPtr

    def cudaIpcCloseMemHandle(self, devPtr: ctypes.c_void_p) -> None:
        self._check(self.funcs["cudaIpcCloseMemHandle"](devPtr))
