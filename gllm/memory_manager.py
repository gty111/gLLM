from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import torch
import torch.distributed as dist
from logger import logger

from collections import deque

from gllm.dist_utils import get_pp_size
from gllm.id_allocator import IDAllocator
from gllm.sequence import Sequence
from gllm.utils import async_tensor_h2d, get_dtype_bytes

# DeepSeek Sparse Attention FP8 MLA cache: the nope latent is quantized in
# 128-wide tiles (one fp32 scale per tile), matching FlashMLA's packed layout.
_DSA_FP8_TILE = 128


@dataclass
class SSMCacheConfig:
    """Layout description for the recurrent-state cache used by linear-attention
    (Mamba / Gated DeltaNet) layers.

    ``num_layers`` is the count of *linear-attention* layers on this PP rank,
    *not* the total decoder depth. The full-attention layers continue to use
    the regular paged KV cache (``Segment.k_cache`` / ``v_cache``) and do not
    consume slots here.

    Shapes (per layer, after TP sharding on the head dim):

    * ``conv_state``  : ``(pool_size, conv_dim, conv_kernel - 1)``
    * ``temporal_state``: ``(pool_size, num_v_heads, head_v_dim, head_k_dim)``

    Slot 0 in the working pool is reserved as the CUDA-graph dummy slot
    (mirrors how :class:`MemoryManager` reserves a dummy KV page) so a padded
    decode row can write into it without polluting any real request's state.
    """
    num_layers: int
    conv_dim: int
    conv_kernel: int
    num_v_heads: int
    head_v_dim: int
    head_k_dim: int
    # Recurrent (temporal) SSM state dtype. Mamba/GDN papers accumulate the
    # delta-rule recurrence in fp32 even when the rest of the model is
    # bf16/fp16, controlled by ``mamba_ssm_dtype`` in the HF config.
    dtype: torch.dtype = torch.float32
    # Conv-state dtype tracks the model's activation dtype because the vendored
    # ``causal_conv1d_*`` kernels do a typed ``tl.load(...)`` from the conv
    # state buffer and need it to match the input ``mixed_qkv`` dtype to keep
    # the Triton ``tl.if`` branches type-consistent.
    conv_state_dtype: torch.dtype = torch.bfloat16

    # ``layer_id`` here is the PP-local *decoder* layer id (0..num_local_layers-1).
    # ``ssm_layer_ids`` is the subset that should hit the SSM cache. Both are
    # populated by the model file at construction time.
    ssm_layer_ids: List[int] = field(default_factory=list)

    def conv_state_shape_per_slot(self):
        return (self.conv_dim, self.conv_kernel - 1)

    def temporal_state_shape_per_slot(self):
        return (self.num_v_heads, self.head_v_dim, self.head_k_dim)

    def per_slot_bytes(self) -> int:
        """Memory footprint of a *single* pool slot, summed across all linear
        layers, post TP sharding. Used for SSM cache sizing logs.
        """
        conv_bytes = get_dtype_bytes(self.conv_state_dtype) * \
            self.conv_dim * (self.conv_kernel - 1)
        temp_bytes = get_dtype_bytes(self.dtype) * \
            self.num_v_heads * self.head_v_dim * self.head_k_dim
        return self.num_layers * (conv_bytes + temp_bytes)


class SSMSegment:
    """Twin tensor banks for the GDN/Mamba recurrent state.

    Two independent pools share the same per-slot tensor layout:

    * **Working pool**: one slot per *live* request. Holds the conv + temporal
      state that gets mutated in place by every forward.
    * **Snapshot pool**: one slot per *cached prefix page*. Holds a frozen copy
      of the working state at a page boundary so a future prefix-cache hit can
      restore it into a fresh working slot via :meth:`copy_state`.

    Each pool exposes its own :class:`IDAllocator`; slot ids are independent
    across the two pools. Both pools always reserve slot 0 as a CUDA-graph
    padding dummy.
    """

    def __init__(
        self,
        cfg: SSMCacheConfig,
        working_pool_size: int,
        snapshot_pool_size: int,
    ):
        self.cfg = cfg
        # +1 so slot 0 stays available as the CUDA-graph dummy slot.
        self.working_pool_size = working_pool_size + 1
        self.snapshot_pool_size = max(snapshot_pool_size, 0) + 1

        conv_shape = cfg.conv_state_shape_per_slot()
        temp_shape = cfg.temporal_state_shape_per_slot()

        # Layout: ``[num_layers, pool_size, *per_slot]``. Indexing always uses
        # ``[layer_id, slot_id, ...]`` so layers can do batched gather via the
        # per-token slot mapping. ``torch.zeros`` (not ``empty``) because the
        # SSM kernels read from slot 0 / freshly-allocated slots before the
        # first write and require a clean initial state (h_0 = 0).
        self.conv_state = [
            torch.zeros((self.working_pool_size, *conv_shape),
                        dtype=cfg.conv_state_dtype)
            for _ in range(cfg.num_layers)
        ]
        self.temporal_state = [
            torch.zeros((self.working_pool_size, *temp_shape), dtype=cfg.dtype)
            for _ in range(cfg.num_layers)
        ]

        if self.snapshot_pool_size > 1:
            self.conv_state_snap = [
                torch.zeros((self.snapshot_pool_size, *conv_shape),
                            dtype=cfg.conv_state_dtype)
                for _ in range(cfg.num_layers)
            ]
            self.temporal_state_snap = [
                torch.zeros((self.snapshot_pool_size, *temp_shape), dtype=cfg.dtype)
                for _ in range(cfg.num_layers)
            ]
        else:
            self.conv_state_snap = None
            self.temporal_state_snap = None

        # Slot 0 reserved for both pools.
        self.working_alloc = IDAllocator(1, self.working_pool_size - 1)
        if self.snapshot_pool_size > 1:
            self.snapshot_alloc = IDAllocator(1, self.snapshot_pool_size - 1)
        else:
            self.snapshot_alloc = None

        # Dummy slots that padded rows / unused snapshot pointers can refer to
        # without aliasing any real state.
        self.dummy_working_slot: int = 0
        self.dummy_snapshot_slot: int = 0

        # Optional CUDA stream that ``copy_state`` (the prefix-cache restore)
        # must run on. Under overlap scheduling the snapshot WRITE happens
        # inside the model forward on ``forward_stream``, while the restore is
        # issued from the scheduler on the CPU thread (default stream). With no
        # shared stream the restore could read a snapshot the in-flight forward
        # has not finished writing. ``OverlapModelRunner`` sets this to
        # ``forward_stream`` so the restore is FIFO-ordered after the forward's
        # snapshot write (it is always enqueued after the forward launch).
        # ``None`` (non-overlap) keeps the restore on the single default stream,
        # where it is already serialized with the forward.
        self.restore_stream: Optional["torch.cuda.Stream"] = None

    # --- working pool ---------------------------------------------------

    def allocate_working(self) -> int:
        return self.working_alloc.allocate()

    def free_working(self, slot: int) -> None:
        if slot is None or slot == self.dummy_working_slot:
            return
        # Zero before returning so the next request starts from h_0 = 0
        # without needing an explicit "reset state" pass through every layer.
        for layer_id in range(self.cfg.num_layers):
            self.conv_state[layer_id][slot].zero_()
            self.temporal_state[layer_id][slot].zero_()
        self.working_alloc.free(slot)

    def num_free_working(self) -> int:
        return self.working_alloc.get_num_free_ids()

    # --- snapshot pool --------------------------------------------------

    def allocate_snapshot(self) -> Optional[int]:
        if self.snapshot_alloc is None or self.snapshot_alloc.get_num_free_ids() == 0:
            return None
        return self.snapshot_alloc.allocate()

    def free_snapshot(self, slot: int) -> None:
        if (
            slot is None
            or self.snapshot_alloc is None
            or slot == self.dummy_snapshot_slot
        ):
            return
        self.snapshot_alloc.free(slot)

    def num_free_snapshot(self) -> int:
        if self.snapshot_alloc is None:
            return 0
        return self.snapshot_alloc.get_num_free_ids()

    # --- transfer -------------------------------------------------------

    def copy_state(
        self,
        src_kind: str,
        src_slot: int,
        dst_kind: str,
        dst_slot: int,
    ) -> None:
        """Copy a full multi-layer state snapshot between pools.

        ``kind`` is one of ``"working"`` / ``"snapshot"``. Used in two places:

        * Prefill snapshot capture: ``copy_state("working", req_slot,
          "snapshot", page_slot)`` after the GDN layer finishes a chunk that
          crosses a page boundary.
        * Prefix-cache hit restore: ``copy_state("snapshot", page_slot,
          "working", req_slot)`` before the new request runs its first
          forward.
        """
        src_conv, src_temp = self._pool(src_kind)
        dst_conv, dst_temp = self._pool(dst_kind)
        if src_conv is None or dst_conv is None:
            return

        def _do_copies():
            for layer_id in range(self.cfg.num_layers):
                dst_conv[layer_id][dst_slot].copy_(src_conv[layer_id][src_slot])
                dst_temp[layer_id][dst_slot].copy_(src_temp[layer_id][src_slot])

        # Pin the copies to ``restore_stream`` when set (overlap scheduling) so
        # a restore that reads a snapshot written by the in-flight forward is
        # ordered after that write. The restore is always enqueued after the
        # forward launch on the CPU thread, so same-stream FIFO is sufficient
        # -- no explicit event needed. ``None`` -> current (default) stream.
        if self.restore_stream is not None:
            with torch.cuda.stream(self.restore_stream):
                _do_copies()
        else:
            _do_copies()

    def _pool(self, kind: str):
        if kind == "working":
            return self.conv_state, self.temporal_state
        if kind == "snapshot":
            return self.conv_state_snap, self.temporal_state_snap
        raise ValueError(f"unknown ssm pool kind: {kind!r}")


class Segment:
    def __init__(
        self,
        num_layers: int,
        num_pages: int,
        page_size: int,
        kv_head_num: int,
        kv_head_dim: int,
        use_mla: bool,
        index_head_dim: int = 0,
        qk_rope_head_dim: int = 0,
        mla_cache_fp8: bool = False,
    ):
        """``num_layers`` here is the number of layers that *actually consume
        KV pages*. For text-only / non-hybrid models that's the full decoder
        depth; for Qwen3.5 and other hybrid GDN models it's the count of
        ``full_attention`` layers (the linear-attention layers route their
        recurrent state through :class:`SSMSegment` instead).

        ``index_head_dim`` (> 0 only for DeepSeek Sparse Attention / V3.2)
        allocates a parallel per-layer **indexer key cache** of shape
        ``[num_pages, page_size, index_head_dim]``. The lightning indexer's
        post-norm+rope key is a single-head ``index_head_dim`` (128) vector per
        token that cannot be derived from the MLA latent (it comes from a
        separate ``wk`` projection), so it needs its own paged cache written by
        the same ``slot_mapping`` as the MLA latent.
        """
        self.num_layers = num_layers
        self.num_pages = num_pages
        self.page_size = page_size
        self.kv_head_num = kv_head_num
        self.kv_head_dim = kv_head_dim
        self.index_head_dim = index_head_dim
        # DeepSeek Sparse Attention: the MLA latent cache is stored in FlashMLA's
        # FP8 packed layout (656 bytes/token) only when ``mla_cache_fp8`` is
        # explicitly enabled -- that layout is what the SM90 *sparse* decode
        # kernel reads (bf16 sparse decode is Blackwell-only). Default is a plain
        # bf16 latent cache + dense decode, which is exact for prompts <=
        # index_topk. Every non-DSA model keeps its bf16 latent cache unchanged.
        self.mla_cache_fp8 = use_mla and index_head_dim > 0 and mla_cache_fp8
        # Packed FP8 layout size: kv_lora_rank(=kv_head_dim - qk_rope) FP8 bytes
        # + (kv_lora_rank/128) fp32 scale bytes + qk_rope_head_dim bf16 bytes.
        # For MLA, kv_head_dim = kv_lora_rank + qk_rope_head_dim.

        if not use_mla:
            # We don't need zero initialization here
            self.k_cache = [
                torch.ones((num_pages, page_size, kv_head_num, kv_head_dim))
                for _ in range(num_layers)
            ]
            self.v_cache = [
                torch.ones((num_pages, page_size, kv_head_num, kv_head_dim))
                for _ in range(num_layers)
            ]
        elif self.mla_cache_fp8:
            # kv_head_dim is kv_lora_rank + qk_rope_head_dim (e.g. 512 + 64).
            qk_rope = qk_rope_head_dim
            kv_lora = kv_head_dim - qk_rope
            assert kv_lora % _DSA_FP8_TILE == 0, (
                f"kv_lora_rank {kv_lora} must be divisible by FP8 tile "
                f"{_DSA_FP8_TILE} for the DSA FP8 MLA cache"
            )
            num_tiles = kv_lora // _DSA_FP8_TILE
            self.mla_fp8_dim = kv_lora + num_tiles * 4 + qk_rope * 2  # 656
            self.kv_cache = [
                torch.zeros(
                    (num_pages, page_size, 1, self.mla_fp8_dim),
                    dtype=torch.float8_e4m3fn,
                )
                for _ in range(num_layers)
            ]
        else:
            self.kv_cache = [
                torch.ones((num_pages, page_size, kv_head_dim))
                for _ in range(num_layers)
            ]
        # DeepSeek Sparse Attention: parallel indexer key cache (bf16, one
        # single-head index_head_dim vector per token per layer). Only
        # allocated when index_head_dim > 0.
        if index_head_dim > 0:
            self.index_k_cache = [
                torch.zeros((num_pages, page_size, index_head_dim))
                for _ in range(num_layers)
            ]
        else:
            self.index_k_cache = None
        self.id_allocator = IDAllocator(0, num_pages - 1)

    def allocate(self):
        pagenum = self.id_allocator.allocate()
        return pagenum

    def free(self, page_num: int):
        self.id_allocator.free(page_num)

    def get_num_free_pages(self):
        return self.id_allocator.get_num_free_ids()

    # return percent of used memory
    def get_memory_util(self):
        return round(
            100 * self.id_allocator.get_num_used_ids() / self.id_allocator.size, 2
        )


class MemoryManager:
    def __init__(
        self,
        gpu_memory_util: float,
        num_layers: int,
        dtype: torch.dtype,
        page_size: int,
        kv_head_num: int,
        kv_head_dim: int,
        vocab_size: int,
        use_mla: bool = False,
        ssm_cache_config: Optional[SSMCacheConfig] = None,
        max_working_ssm_slots: int = 0,
        max_snapshot_ssm_slots: int = 0,
        max_running_seqs: int = 256,
        index_head_dim: int = 0,
        qk_rope_head_dim: int = 0,
        mla_cache_fp8: bool = False,
    ):
        """
        Args:
            num_layers: number of decoder layers *that consume KV cache pages*.
                For text-only models that's every layer; for hybrid GDN
                models (Qwen3.5) it's only the full-attention layers.
            page_size: number of tokens in a KV page.
            kv_head_num: number of k/v heads (post-TP-shard).
            kv_head_dim: dimension of one k/v head.
            ssm_cache_config: layout for the recurrent (Mamba/GDN) state
                cache. ``None`` disables the SSM segment entirely; the rest
                of gllm behaves exactly as before (this is the path used by
                every non-hybrid model).
            max_working_ssm_slots: number of live request slots in the SSM
                working pool. Should be ``>= max_running_seqs`` so the
                scheduler always finds room.
            max_snapshot_ssm_slots: number of cached-prefix slots in the SSM
                snapshot pool. Set to 0 to disable SSM prefix caching while
                keeping per-request SSM state. Otherwise this is the budget
                for cross-request state reuse (mirrors sglang's
                ``--max-mamba-cache-size``).
        """
        self.gpu_memory_util = gpu_memory_util
        self.num_layers = num_layers
        self.page_size = page_size
        self.kv_head_num = kv_head_num
        self.kv_head_dim = kv_head_dim
        self.dtype = dtype
        self.vocab_size = vocab_size
        self.use_mla = use_mla
        # DeepSeek Sparse Attention indexer key cache head dim (0 = disabled).
        self.index_head_dim = index_head_dim
        # MLA rope head dim, needed to size the native FP8 MLA cache layout.
        self.qk_rope_head_dim = qk_rope_head_dim
        # Whether the MLA latent cache is stored natively in FP8 (DSA). Default
        # is bf16 (full precision, dense decode); FP8-packed is opt-in and only
        # needed to drive FlashMLA's *sparse* decode kernel on SM90 for long
        # context (> index_topk). DSA on bf16 runs dense decode, which is exact
        # for prompts <= index_topk (the sparse top-k would select every key).
        self.mla_cache_fp8 = use_mla and index_head_dim > 0 and mla_cache_fp8
        self.ssm_cache_config = ssm_cache_config
        self.max_working_ssm_slots = max_working_ssm_slots
        self.max_snapshot_ssm_slots = max_snapshot_ssm_slots
        # Upper bound on the share of util-scaled free memory the SSM pools may
        # occupy before the KV cache is sized. The snapshot pool (best-effort)
        # is clamped to fit; the working pool (mandatory) is always honored.
        # TODO: replace with a derived formula based on per_slot_bytes vs
        #       kv_bytes_per_page so the split is model-aware.
        self.ssm_pool_budget_frac: float = 0.5
        # Populated by :meth:`init`; ``None`` when the model is not hybrid.
        self.ssm_segment: Optional[SSMSegment] = None
        self.segment: Union[Segment, PrefixSegment] = None

        # --- Persistent repetition-penalty mask pool --------------------
        # Lazily allocated on the first batch that actually uses a non-1.0
        # ``repetition_penalty`` (so workloads that never set one pay nothing,
        # not even GPU memory). ``_rep_pool`` is a ``[num_slots + 1, vocab]``
        # tensor: row 0 is an immutable all-ones sentinel reused for every
        # seq with ``repetition_penalty == 1.0`` (multiplying through it is a
        # no-op), rows ``1..num_slots`` are per-seq persistent rows. Each seq
        # incrementally scatters only its newly generated token(s) into its
        # row, and the per-step ``[batch, vocab]`` mask is a single
        # ``index_select`` gather over the slot ids -- O(batch) work per step
        # instead of the previous O(sum(len(token_ids))) full rebuild.
        self.max_running_seqs = max_running_seqs
        self._rep_pool: Optional[torch.Tensor] = None
        self._rep_free_slots: Optional[deque] = None

    @property
    def use_ssm_cache(self) -> bool:
        return self.ssm_cache_config is not None

    def consume_pending_ssm_restores(self) -> Dict[int, int]:
        """No SSM prefix caching without a snapshot pool (base manager)."""
        return {}

    def init(self, segment_cls=Segment, reserve_dummy_page: bool = False):
        # Allocate SSM pools before sizing the KV cache so ``mem_get_info``
        # reflects the true post-SSM free memory. Do not subtract an estimated
        # byte count again afterward -- the tensors are already on CUDA.
        self._init_ssm_segment_if_needed()

        free_mem_size, _ = torch.cuda.mem_get_info()
        num_max_pages = free_mem_size // self.get_sizeof_KV_per_page()
        num_pages = int(num_max_pages * self.gpu_memory_util)

        if not dist.is_initialized():
            self.num_pages = num_pages
        else:
            num_pages_all = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(num_pages_all, num_pages)
            self.num_pages = min(num_pages_all)

        # KV cache element precision: native FP8 for DeepSeek Sparse Attention
        # (packed 656-byte MLA latent), otherwise the model dtype (e.g. bf16).
        if self.mla_cache_fp8:
            kv_dtype_str = "fp8_e4m3 (nope) + bf16 (rope)"
        else:
            kv_dtype_str = str(self.dtype).replace("torch.", "")
        logger.info(
            f"KV cache: {self.num_pages} pages ({self.page_size} tokens/page), "
            f"dtype {kv_dtype_str}, "
            f"{round(self.get_sizeof_KV_per_page()/(2**10*self.page_size),2)} KB (per token), "
            f"{round(self.num_pages*self.get_sizeof_KV_per_page()/(2**30),2)} GB (total)"
        )

        self.segment = segment_cls(
            self.num_layers,
            self.num_pages,
            self.page_size,
            self.kv_head_num,
            self.kv_head_dim,
            self.use_mla,
            index_head_dim=self.index_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            mla_cache_fp8=self.mla_cache_fp8,
        )

        # Reserve a dedicated dummy page for CUDA graph padding only when
        # CUDA graphs are enabled.  This page is never returned to normal use,
        # so real sequences will never overwrite it, and padding dummy tokens
        # can safely write here.
        self.dummy_page: int = self.segment.allocate() if reserve_dummy_page else None

        self.kv_cache_dtype = "auto"
        self.k_scale = torch.tensor(1.0, dtype=torch.float32)
        self.v_scale = self.k_scale

    def _init_ssm_segment_if_needed(self) -> None:
        """Allocate the SSM working+snapshot pools when the model needs them.

        For hybrid GDN/Mamba models each pool slot holds the full per-layer
        recurrent state, so the requested pools (``maxd`` working +
        ``4*maxd`` snapshot) can be tens of GB and -- allocated eagerly before
        the KV cache is sized -- exhaust the device and OOM right here. To make
        startup robust we size the pools against the *currently free* memory:

        * the **working pool** (one slot per concurrently-running seq) is
          mandatory for correctness and always allocated; if even it cannot
          fit we raise a clear, actionable error instead of a bare CUDA OOM;
        * the **snapshot pool** (best-effort prefix-cache state reuse) is
          clamped so the total SSM footprint stays within
          ``ssm_pool_budget_frac`` of the util-scaled free memory, leaving the
          remainder for the KV cache. When memory is ample the full requested
          snapshot pool is honored (behavior unchanged); when tight it shrinks,
          down to 0 (SSM prefix caching disabled) before anything OOMs.
        """
        if self.ssm_cache_config is None:
            return
        cfg = self.ssm_cache_config
        per_slot = cfg.per_slot_bytes()

        free_mem, _ = torch.cuda.mem_get_info()
        budget = int(free_mem * self.gpu_memory_util * self.ssm_pool_budget_frac)

        # +1 mirrors SSMSegment's reserved CUDA-graph dummy slot in each pool.
        working_slots = self.max_working_ssm_slots
        working_bytes = (working_slots + 1) * per_slot
        if working_bytes >= free_mem:
            raise RuntimeError(
                f"SSM working pool needs {working_bytes / (1 << 30):.1f} GB "
                f"({working_slots} slots x {per_slot / (1 << 20):.1f} MB) but only "
                f"{free_mem / (1 << 30):.1f} GB is free after loading weights. "
                f"Lower --maxd (currently {working_slots}) or use more "
                f"tensor-parallel GPUs (--tp) to shrink the per-rank state."
            )

        requested_snapshot = self.max_snapshot_ssm_slots
        affordable_snapshot = max(0, (budget - working_bytes) // per_slot - 1)
        snapshot_slots = min(requested_snapshot, int(affordable_snapshot))

        # Keep every TP rank's pool layout identical (state is sharded, not
        # replicated, but the slot *count* must match across ranks); free
        # memory can differ slightly per rank, so agree on the minimum.
        if dist.is_initialized():
            gathered = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered, snapshot_slots)
            snapshot_slots = min(gathered)

        if snapshot_slots < requested_snapshot:
            logger.warning(
                "SSM snapshot pool clamped %d -> %d slots to fit the memory "
                "budget (%.1f GB free, %.0f%% util, %.0f%% SSM share); prefix-cache "
                "state reuse is %s. Lower --maxd or raise --tp for the full pool.",
                requested_snapshot,
                snapshot_slots,
                free_mem / (1 << 30),
                self.gpu_memory_util * 100,
                self.ssm_pool_budget_frac * 100,
                "reduced" if snapshot_slots > 0 else "disabled",
            )

        self.ssm_segment = SSMSegment(
            cfg,
            working_pool_size=working_slots,
            snapshot_pool_size=snapshot_slots,
        )
        total = per_slot * (
            self.ssm_segment.working_pool_size + self.ssm_segment.snapshot_pool_size
        )
        logger.info(
            "SSM cache: %d working slots, %d snapshot slots, "
            "%.2f KB/slot, %.2f GB total (linear-attn layers: %d)",
            self.ssm_segment.working_pool_size,
            self.ssm_segment.snapshot_pool_size,
            per_slot / 1024,
            total / (1 << 30),
            cfg.num_layers,
        )

    def get_sizeof_KV_per_page(self):  # Bytes
        if not self.use_mla:
            # 2: K cache and V cache
            return (
                2
                * self.num_layers
                * self.page_size
                * self.kv_head_num
                * self.kv_head_dim
                * get_dtype_bytes(self.dtype)
            )
        else:
            # Per-token MLA latent bytes. Native FP8 (DSA) uses the packed
            # 656-byte layout (1 byte/elem, computed in Segment as mla_fp8_dim);
            # otherwise bf16 kv_head_dim. The index key cache adds its own
            # per-token bytes (bf16) on top.
            if self.mla_cache_fp8:
                qk_rope = self.qk_rope_head_dim
                kv_lora = self.kv_head_dim - qk_rope
                num_tiles = kv_lora // _DSA_FP8_TILE
                mla_bytes = kv_lora + num_tiles * 4 + qk_rope * 2  # 656, 1 B/elem
            else:
                mla_bytes = self.kv_head_dim * get_dtype_bytes(self.dtype)
            index_bytes = self.index_head_dim * get_dtype_bytes(self.dtype)
            return self.num_layers * self.page_size * (mla_bytes + index_bytes)

    def store_index_k(
        self,
        layer_idx: int,
        index_k: torch.Tensor,
        slot_mapping_tensor: torch.Tensor,
    ):
        """Write the DSA indexer key into the paged index cache by slot.

        ``index_k`` is ``[num_tokens, index_head_dim]`` (post norm+rope, single
        head). The paged cache is ``[num_pages, page_size, index_head_dim]``;
        ``slot_mapping_tensor`` gives the flattened ``page*page_size + offset``
        slot for each token, identical to the MLA latent's slot mapping. A plain
        indexed scatter into the flattened (num_slots, dim) view is enough here
        -- the indexer is not the throughput bottleneck and this keeps the write
        dtype-agnostic and kernel-free.
        """
        cache = self.segment.index_k_cache[layer_idx]
        num_pages, page_size, dim = cache.shape
        flat = cache.view(num_pages * page_size, dim)
        flat[slot_mapping_tensor] = index_k.to(flat.dtype)

    def batch_store(
        self,
        layer_idx: int,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_mapping_tensor: torch.Tensor,
    ):
        from gllm import _custom_ops as ops

        ops.reshape_and_cache_flash(
            k_cache,
            v_cache,
            self.segment.k_cache[layer_idx],
            self.segment.v_cache[layer_idx],
            slot_mapping_tensor,
            self.kv_cache_dtype,
            self.k_scale,
            self.v_scale,
        )

    def pre_allocate_page(self, seqs: List[Sequence]):
        for seq in seqs:
            num_page = (seq.seq_len + self.page_size - 1) // self.page_size - len(
                seq.page_table
            )
            for _ in range(num_page):
                seq.page_table.append(self.segment.allocate())

    def register_decode_boundary(self, seq: Sequence, pos: int) -> None:
        """No-op without a prefix cache; overridden by ``PrefixMemoryManager``."""
        return

    def free(self, seq: Sequence):
        for page_num in seq.page_table:
            self.segment.free(page_num)
        self.free_ssm_slot(seq)
        self.free_rep_slot(seq)

    # --- Repetition-penalty mask pool lifecycle ---------------------------

    def _ensure_rep_pool(self) -> None:
        if self._rep_pool is not None:
            return
        num_slots = max(self.max_running_seqs, 1)
        # +1 for the row-0 all-ones sentinel.
        self._rep_pool = torch.ones(
            (num_slots + 1, self.vocab_size), dtype=self.dtype, device="cuda"
        )
        self._rep_free_slots = deque(range(1, num_slots + 1))

    def _grow_rep_pool(self, extra: int) -> None:
        """Append ``extra`` fresh all-ones rows to the pool.

        Concurrent decode is normally bounded by ``max_running_seqs`` (the
        scheduler caps each batch at that many rows), so this is a rare
        safety valve rather than a steady-state path.
        """
        old_rows = self._rep_pool.shape[0]
        new_rows = torch.ones(
            (extra, self.vocab_size), dtype=self.dtype, device="cuda"
        )
        self._rep_pool = torch.cat([self._rep_pool, new_rows], dim=0)
        self._rep_free_slots.extend(range(old_rows, old_rows + extra))

    def free_rep_slot(self, seq: Sequence) -> None:
        if seq.rep_slot is None:
            return
        # Lazy reset: the row is re-filled with ones when the slot is handed
        # to the next seq (see ``build_repetition_penalty_mask``), so we only
        # need to return the id and clear the per-seq bookkeeping here.
        if self._rep_free_slots is not None:
            self._rep_free_slots.append(seq.rep_slot)
        seq.rep_slot = None
        seq.rep_filled = 0

    def build_repetition_penalty_mask(self, seqs: List[Sequence]):
        """Return a ``[batch, vocab]`` scaling-penalty mask, or ``None``.

        Incremental + persistent: every seq with ``repetition_penalty != 1.0``
        owns a pool row that is updated with only its newly appended tokens
        each step; the batch mask is gathered from those rows in one op.
        Mirrors the semantics of the old from-scratch builder (penalty value
        at already-seen token positions, 1.0 elsewhere).
        """
        active = [
            seq
            for seq in seqs
            if getattr(seq, "repetition_penalty", 1.0) != 1.0
            and seq.token_ids is not None
        ]
        if not active:
            return None

        self._ensure_rep_pool()

        # 1) Allocate slots for new seqs and collect the (slot, token) pairs
        #    that still need scattering -- only the suffix of token_ids that
        #    has not been seen yet (one token per decode step in steady state).
        new_slots: List[int] = []
        new_tokens: List[int] = []
        new_pens: List[float] = []
        reset_slots: List[int] = []
        for seq in active:
            if seq.rep_slot is None:
                if not self._rep_free_slots:
                    self._grow_rep_pool(self.max_running_seqs)
                seq.rep_slot = self._rep_free_slots.popleft()
                # Reset a (possibly reused) row back to the all-ones baseline
                # (done in bulk below, after ``_grow_rep_pool`` may have
                # rebuilt ``self._rep_pool`` via torch.cat).
                reset_slots.append(seq.rep_slot)
                seq.rep_filled = 0
            n_total = len(seq.token_ids)
            if n_total > seq.rep_filled:
                suffix = seq.token_ids[seq.rep_filled :]
                new_slots.extend([seq.rep_slot] * len(suffix))
                new_tokens.extend(suffix)
                new_pens.extend([seq.repetition_penalty] * len(suffix))
                seq.rep_filled = n_total

        # ``self._rep_pool`` is only stable to capture *after* the allocation
        # loop, since ``_grow_rep_pool`` may have replaced it via torch.cat.
        pool = self._rep_pool
        if reset_slots:
            pool[reset_slots] = 1.0
        if new_slots:
            slot_t = async_tensor_h2d(new_slots, torch.long, "cuda", True)
            token_t = async_tensor_h2d(new_tokens, torch.long, "cuda", True)
            pen_t = async_tensor_h2d(new_pens, self.dtype, "cuda", True)
            pool[slot_t, token_t] = pen_t

        # 2) Gather the per-batch rows in a single op. Seqs with penalty 1.0
        #    (or no slot) map to the row-0 all-ones sentinel.
        batch_slots = [
            seq.rep_slot
            if (
                getattr(seq, "repetition_penalty", 1.0) != 1.0
                and seq.rep_slot is not None
            )
            else 0
            for seq in seqs
        ]
        batch_slots_t = async_tensor_h2d(batch_slots, torch.long, "cuda", True)
        return pool.index_select(0, batch_slots_t)

    # --- SSM working slot lifecycle ---------------------------------------
    #
    # These are no-ops for non-hybrid models (``ssm_segment is None``). For
    # hybrid models the scheduler calls ``allocate_ssm_slot`` on the first
    # schedule of a sequence (mirroring how KV pages are pre-allocated) and
    # ``free_ssm_slot`` when the sequence finishes or is aborted/preempted.

    def allocate_ssm_slot(self, seq: Sequence) -> None:
        if self.ssm_segment is None or seq.ssm_state_slot is not None:
            return
        seq.ssm_state_slot = self.ssm_segment.allocate_working()

    def free_ssm_slot(self, seq: Sequence) -> None:
        if self.ssm_segment is None or seq.ssm_state_slot is None:
            return
        self.ssm_segment.free_working(seq.ssm_state_slot)
        seq.ssm_state_slot = None

    def get_num_free_pages(self):
        return self.segment.get_num_free_pages()

    def get_memory_util(self):
        return self.segment.get_memory_util()

    def get_memory_free(self):
        return self.get_num_free_pages() / self.num_pages


# ---------------------------------------------------------------------------
# Prefix cache
# ---------------------------------------------------------------------------


# 64-bit nonzero seed for the chained prefix hash. Mixing a constant in
# at chain start prevents the empty-prefix case from collapsing to 0
# (which is the sentinel ``page2hash`` uses for "no hash registered").
_PREFIX_HASH_SEED = 0x9E3779B97F4A7C15
_PREFIX_CANARY_LEN = 8


def _hash_source(seq: Sequence) -> List[int]:
    """Pick the token list used for prefix-cache hashing.

    ``hash_token_ids`` (set by the multimodal pipeline) wins over the raw
    ``token_ids`` so two VL prompts with the same ``<|image_pad|>``
    placeholders but distinct images do not collide.
    """
    hi = getattr(seq, "hash_token_ids", None)
    return hi if hi is not None else seq.token_ids


def _maybe_invalidate_seq_hash_cache(seq: Sequence, src: List[int]) -> None:
    """Drop the per-seq incremental hash cache if its source list changed.

    The hash source is normally stable for the lifetime of a request --
    text-only seqs use ``token_ids`` (decode only appends past the cached
    page boundaries) and VL seqs set ``hash_token_ids`` once before the
    first ``pre_allocate_computed_page``. The check below is a cheap
    safety net for the edge case where the MM pipeline rewrites
    ``hash_token_ids`` after some pages have already been hashed.
    """
    ref = seq._hash_source_ref
    if ref is None or ref != id(src):
        seq._page_hashes = []
        seq._canary_cache = None
        seq._hash_source_ref = id(src)


def _ensure_page_hash(seq: Sequence, page_size: int, page_idx: int) -> int:
    """Return the chained hash for the first ``(page_idx+1)*page_size`` tokens.

    Each new page mixes the previous chain hash with the tuple of token
    ids in this page, so extending the chain by one page costs O(page_size)
    instead of O(prefix_len). The chained hash is reproducible across
    requests: any two seqs sharing identical first ``k*page_size`` tokens
    produce identical ``_page_hashes[k-1]``.
    """
    src = _hash_source(seq)
    _maybe_invalidate_seq_hash_cache(seq, src)
    cache = seq._page_hashes
    if page_idx < len(cache):
        return cache[page_idx]
    while len(cache) <= page_idx:
        i = len(cache)
        prev = cache[i - 1] if i > 0 else _PREFIX_HASH_SEED
        page_tokens = tuple(src[i * page_size:(i + 1) * page_size])
        cache.append(hash((prev, page_tokens)))
    return cache[page_idx]


def _ensure_canary(seq: Sequence) -> tuple:
    """Return the first ``_PREFIX_CANARY_LEN`` ids as a tuple, cached on ``seq``.

    Used as a hash-collision sanity check on lookups. Mirrors the original
    ``key[:8]`` canary semantics, which were the first 8 ids of the
    *prefix tuple* (and therefore identical for every page boundary of a
    single seq), but built without rebuilding the full prefix tuple each
    call.
    """
    src = _hash_source(seq)
    _maybe_invalidate_seq_hash_cache(seq, src)
    c = seq._canary_cache
    if c is None:
        c = tuple(src[:_PREFIX_CANARY_LEN])
        seq._canary_cache = c
    return c


class PrefixMemoryManager(MemoryManager):
    """KV-page-granular prefix cache with optional SSM snapshot integration.

    The cache key is the chained per-page hash built lazily on each
    ``Sequence`` via ``_ensure_page_hash``: extending the chain by one page
    is O(page_size) instead of O(prefix_len), which keeps long-context
    prefill from spending most of its CPU in tuple/hash construction.
    Multimodal disambiguation is preserved because the hash chain reads
    from ``hash_token_ids`` (set by the MM pipeline) when present, falling
    back to ``token_ids`` otherwise. When the underlying ``MemoryManager``
    was constructed with an SSM cache config, every cached page also
    carries an optional SSM snapshot slot that holds the conv+temporal
    state captured at that page boundary by the GDN layer. A cache hit
    copies the snapshot back into the requesting sequence's working slot
    before the new forward runs.
    """

    def init(self, reserve_dummy_page: bool = False):
        super().init(segment_cls=PrefixSegment, reserve_dummy_page=reserve_dummy_page)
        self.segment.ssm_segment = self.ssm_segment

        # Cache-hit-rate stats.
        self.num_allocated_pages = 0
        self.num_hit_pages = 0

        # PP>1 only: SSM snapshot restores performed this scheduling iteration,
        # keyed by seq_id -> snapshot-pool slot. Each PP follower owns a
        # *different* slice of the GDN layers on its own GPU, so the restore
        # (snapshot->working ``copy_state``) the driver runs on rank-0's pools
        # must be replayed on every stage. The driver records the restores here
        # and the payload builder ships them; ``consume`` clears the buffer so
        # each is shipped exactly once.
        self._pending_ssm_restores: Dict[int, int] = {}

    def pre_allocate_computed_page(self, seqs: List[Sequence]):
        for seq in seqs:
            assert len(seq.page_table) == 0
            num_page = (len(seq) + self.page_size - 1) // self.page_size
            if not seq.computed_prompt:
                self.num_allocated_pages += num_page
            for i in range(num_page):
                if (i + 1) * self.page_size <= len(seq):
                    page_num = self.segment.has_computed(seq, (i + 1) * self.page_size)
                    if page_num is not None:
                        seq.page_table.append(page_num)
                        seq.computed_token_num += self.page_size
                        self.num_hit_pages += 1
                    else:
                        break
                else:
                    break
        for seq in seqs:
            self._finalize_prefix_cache_hit(seq)

    def _finalize_prefix_cache_hit(self, seq: Sequence) -> None:
        """Post-process a prefix-cache lookup so forward work remains on the prompt.

        After a **full** hit (``computed_token_num == len(seq)``) every prompt
        token is marked computed and the batch builder would schedule zero
        tokens -- ``compute_logits`` then sees an empty hidden tensor. We must
        leave at least one token to forward so the last hidden state is
        available for the first decode sample:

        * **Full-attention**: roll back 1 token. Re-running it is safe (KV
          rewrite is idempotent).
        * **Hybrid SSM (GDN)**: roll back one *page*, restore the recurrent
          state from the previous boundary snapshot, and recompute the tail.
          A 1-token rollback would apply the last token's recurrence twice
          and corrupt SSM state.

        On **partial** hits hybrid models still copy the deepest hit snapshot
        into the working slot; full-attention models need no change.
        """
        if seq.computed_token_num == 0:
            return

        is_hybrid = self.ssm_segment is not None
        full_hit = seq.computed_token_num >= len(seq)

        if is_hybrid:
            if full_hit:
                seq.computed_token_num -= self.page_size
                self.num_hit_pages -= 1
            self._restore_ssm_working_state(seq)
        elif full_hit:
            seq.computed_token_num = len(seq) - 1

    def _restore_ssm_working_state(self, seq: Sequence) -> None:
        """Copy the deepest *filled* SSM snapshot at/below ``computed_token_num``.

        Cached KV pages stay in ``page_table`` and are recomputed in place by
        the upcoming forward (idempotent for full-attention KV). We never
        free/pop pages here -- a sibling request in the same batch may share
        them.
        """
        while seq.computed_token_num > 0:
            boundary_page = seq.page_table[
                seq.computed_token_num // self.page_size - 1
            ]
            snap_slot = self._valid_snapshot_slot(boundary_page)
            if snap_slot is not None:
                self.allocate_ssm_slot(seq)
                self.ssm_segment.copy_state(
                    "snapshot", snap_slot, "working", seq.ssm_state_slot
                )
                # PP>1: record so the same restore is replayed on every PP
                # stage (each owns a different GDN-layer slice). Skip entirely
                # on PP=1 where rank-0's copy above is the whole story (and
                # nothing would ever drain the buffer).
                if get_pp_size() > 1:
                    self._pending_ssm_restores[seq.seq_id] = snap_slot
                return
            seq.computed_token_num -= self.page_size
            self.num_hit_pages -= 1
        # No usable boundary snapshot: scheduler allocates a fresh (zeroed)
        # working slot and the seq recomputes the whole prompt from h_0.

    def register_decode_boundary(self, seq: Sequence, pos: int) -> None:
        """Register the prefix-cache hash for the page completed by the real
        token now at ``seq.token_ids[pos]`` (no-op unless ``pos`` lands on a
        page boundary).

        This is the decode-stage counterpart to the prefill-time cacheable
        ``allocate(seq, n_tokens)``. It is intentionally **decoupled from
        page allocation** (``pre_allocate_page``) and driven instead from the
        scheduler's output-finalization hooks (``process_output`` /
        ``process_output_finalize`` via ``ModelRunner.register_decode_page_hash``)
        so it only ever runs once ``token_ids[pos]`` holds the *real* sampled
        token. Under overlap scheduling the freshly scheduled decode token is a
        negative placeholder until finalized; registering at allocation time
        would hash the placeholder id and poison the cache (see
        ``docs/prefix_cache_overlap_poisoning.md``).
        """
        n = pos + 1
        if n % self.page_size != 0:
            return
        page_idx = n // self.page_size - 1
        # The seq may have been preempted (page_table reset to []) between the
        # forward that filled this page and finalize; only register a live page.
        if 0 <= page_idx < len(seq.page_table):
            self.segment.update(seq, n, seq.page_table[page_idx])

    def pre_allocate_page(self, seqs: List[Sequence]):
        for seq in seqs:
            len_page_table = len(seq.page_table)
            num_page = (
                seq.seq_len + self.page_size - 1
            ) // self.page_size - len_page_table
            for i in range(len_page_table, len_page_table + num_page):
                if (i + 1) * self.page_size <= len(seq):
                    page_num = self.segment.allocate(seq, (i + 1) * self.page_size)
                else:
                    page_num = self.segment.allocate()
                seq.page_table.append(page_num)

    def consume_pending_ssm_restores(self) -> Dict[int, int]:
        """Return and clear this iteration's SSM snapshot restores (PP>1).

        Called by the driver's payload builder so each prefix-cache-hit restore
        is shipped to the PP followers exactly once. Empty (cheap) on PP=1 and
        for every non-hit iteration.
        """
        if not self._pending_ssm_restores:
            return {}
        restores = self._pending_ssm_restores
        self._pending_ssm_restores = {}
        return restores

    def _valid_snapshot_slot(self, page_num: int) -> Optional[int]:
        """Return the snapshot slot for ``page_num`` only if it was actually
        *written* (filled), else ``None``. A reserved-but-unfilled slot holds
        zeros and must never be restored onto a non-empty prefix."""
        if not self.segment.page2ssm_snapshot_valid[page_num]:
            return None
        return self.segment.page2ssm_snapshot[page_num]

    def get_cache_hit_rate(self):
        if self.num_allocated_pages == 0:
            return 0.0
        return round(100 * self.num_hit_pages / self.num_allocated_pages, 2)


class PrefixSegment(Segment):
    """Paged KV segment with hash-keyed prefix cache and optional SSM
    snapshot pointers.

    The cache key for a page is produced by the module-level
    ``_ensure_page_hash(seq, page_size, page_idx)`` which incrementally
    chains a per-page hash on the ``Sequence`` itself; for VL the
    sequence's ``hash_token_ids`` view feeds the chain so identical-text +
    different-image prompts no longer collide.

    Collision safety: ``hash2page`` maps Python's tuple-hash to a page; on
    lookup we additionally compare the canary (first 8 ids of the cached
    prefix) before declaring a hit. Without the canary, the previous
    implementation could silently share KV across two distinct prefixes
    whose ``hash()`` happened to match.

    SSM extension: for every cached page we keep an optional snapshot slot
    in the partner :class:`SSMSegment`. When a sequence allocates a page
    in :meth:`PrefixMemoryManager.pre_allocate_page`, a snapshot slot is
    reserved alongside; the GDN layer fills it after the page boundary is
    crossed during prefill. On a cache hit the snapshot is copied back into
    the requesting sequence's working slot.
    """

    # Set by :class:`PrefixMemoryManager.init`.
    ssm_segment: Optional[SSMSegment] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hash2page: Dict[int, int] = {}
        self.page_ref_num = [0 for _ in range(self.num_pages)]
        self.page2hash: List[int] = [0 for _ in range(self.num_pages)]
        # Canary stored per physical page; tuples are kept short
        # (``_PREFIX_CANARY_LEN``) to avoid blowing up memory at large num_pages.
        self.page2canary: List[Optional[tuple]] = [None for _ in range(self.num_pages)]
        # SSM snapshot slot id per physical page; ``None`` if no snapshot
        # was captured (e.g. when the SSM cache is disabled or the boundary
        # never had a chance to snapshot during prefill).
        self.page2ssm_snapshot: List[Optional[int]] = [None for _ in range(self.num_pages)]
        # Whether the reserved snapshot slot actually holds a *written*
        # recurrent state yet. A slot is reserved at ``allocate`` time for
        # every cacheable page, but the GDN layer only writes the snapshot
        # for the page boundary on which a prefill chunk *ends* (see
        # ``InputData._cal_ssm_metadata``). Interior boundaries crossed inside
        # a single chunk keep their reserved-but-zeroed slot, so a hit there
        # must NOT restore it (restoring zeros == h_0 grafted onto a non-zero
        # prefix -> garbage). ``page2ssm_snapshot_valid`` separates "reserved"
        # from "filled" so the restore/rollback paths only trust real states.
        self.page2ssm_snapshot_valid: List[bool] = [False for _ in range(self.num_pages)]

    # --- public API ---------------------------------------------------------

    def update(self, seq: Sequence, n_tokens: int, page_num: int) -> None:
        """Register a hash for ``page_num`` after its KV was filled in decode."""
        page_idx = n_tokens // self.page_size - 1
        page_hash = _ensure_page_hash(seq, self.page_size, page_idx)
        if page_hash not in self.hash2page:
            self.page2hash[page_num] = page_hash
            self.hash2page[page_hash] = page_num
            self.page2canary[page_num] = _ensure_canary(seq)

    def has_computed(self, seq: Sequence, n_tokens: int) -> Optional[int]:
        """Look up a cached page. Returns the page id or ``None`` on miss.

        Performs a canary equality check so two distinct prefixes that happen
        to share a Python ``hash()`` value never silently alias.
        """
        page_idx = n_tokens // self.page_size - 1
        page_hash = _ensure_page_hash(seq, self.page_size, page_idx)
        page_num = self.hash2page.get(page_hash)
        if page_num is None:
            return None
        if self.page2canary[page_num] != _ensure_canary(seq):
            # Hash collision; treat as miss and let the caller allocate a
            # fresh page. We deliberately do not evict the cached page here
            # because the *other* prefix is the legitimate owner.
            return None
        self.id_allocator.allocate(page_num)
        self.page_ref_num[page_num] += 1
        return page_num

    def allocate(self, seq: Optional[Sequence] = None, n_tokens: Optional[int] = None):
        """Allocate a page; optionally register a prefix hash for it.

        Signature is overloaded:

        * ``allocate()`` — non-cacheable allocation (the trailing partial
          page during prefill or any decode page that hasn't crossed a
          boundary yet). Returns a fresh page id without hash registration.
        * ``allocate(seq, n_tokens)`` — cacheable allocation: the caller
          guarantees the new page contains the prefix ``seq[:n_tokens]``,
          so we hash it and register the mapping.
        """
        page_hash = None
        key_canary: Optional[tuple] = None
        if seq is not None and n_tokens is not None:
            page_idx = n_tokens // self.page_size - 1
            page_hash = _ensure_page_hash(seq, self.page_size, page_idx)
            key_canary = _ensure_canary(seq)

        page_num = self.id_allocator.allocate()
        # Re-mint: drop any prior hash entries that pointed at this physical
        # page when it was last cached.
        if self.page2hash[page_num] != 0 and self.page2hash[page_num] in self.hash2page:
            del self.hash2page[self.page2hash[page_num]]
        # Drop stale SSM snapshot for the previous tenant of this page.
        self._release_snapshot_for(page_num)

        if page_hash is not None:
            self.page2hash[page_num] = page_hash
            self.hash2page[page_hash] = page_num
            self.page2canary[page_num] = key_canary
            # Pre-reserve a snapshot slot so the GDN layer can fill it when
            # it crosses this page boundary. If the snapshot pool is full
            # we silently degrade to "KV-cached but no SSM" (the hit path
            # in PrefixMemoryManager will then roll the hit back if it
            # cannot honor the SSM half).
            if self.ssm_segment is not None:
                self.page2ssm_snapshot[page_num] = self.ssm_segment.allocate_snapshot()
                # Freshly reserved slot holds no written state yet.
                self.page2ssm_snapshot_valid[page_num] = False
        else:
            self.page2hash[page_num] = 0
            self.page2canary[page_num] = None
            self.page2ssm_snapshot[page_num] = None
            self.page2ssm_snapshot_valid[page_num] = False

        self.page_ref_num[page_num] += 1
        return page_num

    def free(self, page_num: int) -> None:
        assert self.page_ref_num[page_num] > 0
        self.page_ref_num[page_num] -= 1
        if self.page_ref_num[page_num] == 0:
            # NOTE: keep ``page2ssm_snapshot[page_num]`` alive even though
            # no one is pinning the page anymore. The cached KV survives
            # the ref-count hitting zero (its ``hash2page`` entry stays
            # registered until the page is re-minted for a *different*
            # prompt by :meth:`allocate`). The SSM snapshot must follow
            # the same lifetime — otherwise a serial re-use of a cached
            # prompt would always lose the snapshot half of the hit and
            # ``_rollback_to_last_ssm_hit`` would drop the KV half too.
            self.id_allocator.free(page_num)

    def _release_snapshot_for(self, page_num: int) -> None:
        if self.ssm_segment is None:
            return
        snap_slot = self.page2ssm_snapshot[page_num]
        if snap_slot is not None:
            self.ssm_segment.free_snapshot(snap_slot)
            self.page2ssm_snapshot[page_num] = None
        self.page2ssm_snapshot_valid[page_num] = False
