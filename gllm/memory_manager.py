from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import torch
import torch.distributed as dist
from logger import logger

from gllm.id_allocator import IDAllocator
from gllm.sequence import Sequence
from gllm.utils import get_dtype_bytes


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
        for layer_id in range(self.cfg.num_layers):
            dst_conv[layer_id][dst_slot].copy_(src_conv[layer_id][src_slot])
            dst_temp[layer_id][dst_slot].copy_(src_temp[layer_id][src_slot])

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
    ):
        """``num_layers`` here is the number of layers that *actually consume
        KV pages*. For text-only / non-hybrid models that's the full decoder
        depth; for Qwen3.5 and other hybrid GDN models it's the count of
        ``full_attention`` layers (the linear-attention layers route their
        recurrent state through :class:`SSMSegment` instead).
        """
        self.num_layers = num_layers
        self.num_pages = num_pages
        self.page_size = page_size
        self.kv_head_num = kv_head_num
        self.kv_head_dim = kv_head_dim

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
        else:
            self.kv_cache = [
                torch.ones((num_pages, page_size, kv_head_dim))
                for _ in range(num_layers)
            ]
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
        self.ssm_cache_config = ssm_cache_config
        self.max_working_ssm_slots = max_working_ssm_slots
        self.max_snapshot_ssm_slots = max_snapshot_ssm_slots
        # Populated by :meth:`init`; ``None`` when the model is not hybrid.
        self.ssm_segment: Optional[SSMSegment] = None

    @property
    def use_ssm_cache(self) -> bool:
        return self.ssm_cache_config is not None

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

        logger.info(
            f"KV cache: {self.num_pages} pages ({self.page_size} tokens/page), "
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
        """Allocate the SSM working+snapshot pools when the model needs them."""
        if self.ssm_cache_config is None:
            return
        cfg = self.ssm_cache_config
        self.ssm_segment = SSMSegment(
            cfg,
            working_pool_size=self.max_working_ssm_slots,
            snapshot_pool_size=self.max_snapshot_ssm_slots,
        )
        per_slot = cfg.per_slot_bytes()
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
            return (
                self.num_layers
                * self.page_size
                * self.kv_head_dim
                * get_dtype_bytes(self.dtype)
            )

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

    def free(self, seq: Sequence):
        for page_num in seq.page_table:
            self.segment.free(page_num)
        self.free_ssm_slot(seq)

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


def _default_cache_key_fn(seq: Sequence, n_tokens: int):
    """Default per-page cache key: tuple of the first ``n_tokens`` token ids.

    The optional ``hash_token_ids`` on the sequence overrides ``token_ids``
    when present. This is how Phase G.2 disambiguates multimodal prompts
    that share placeholder token ids but reference different images: the
    MM pipeline pre-renders ``hash_token_ids`` with content-derived
    ``pad_id``s so the cache key naturally diverges across distinct images.
    """
    src = seq.hash_token_ids if getattr(seq, "hash_token_ids", None) is not None else seq.token_ids
    # Tuple-of-int is hashable; the segment also stores the first few ids
    # verbatim for a collision-safety canary on lookup.
    return tuple(src[:n_tokens])


class PrefixMemoryManager(MemoryManager):
    """KV-page-granular prefix cache with optional SSM snapshot integration.

    The cache key is computed via a pluggable ``key_fn`` (default = token-id
    prefix; overridable to splice multimodal content hashes into the key).
    When the underlying ``MemoryManager`` was constructed with an SSM cache
    config, every cached page also carries an optional SSM snapshot slot
    that holds the conv+temporal state captured at that page boundary by
    the GDN layer. A cache hit copies the snapshot back into the requesting
    sequence's working slot before the new forward runs.
    """

    def __init__(self, *args, key_fn=_default_cache_key_fn, **kwargs):
        super().__init__(*args, **kwargs)
        self._key_fn = key_fn

    def init(self, reserve_dummy_page: bool = False):
        super().init(segment_cls=PrefixSegment, reserve_dummy_page=reserve_dummy_page)
        # Stash the configured key function on the segment so it can be used
        # without round-tripping through the manager every call.
        self.segment.key_fn = self._key_fn
        self.segment.ssm_segment = self.ssm_segment

        # Cache-hit-rate stats.
        self.num_allocated_pages = 0
        self.num_hit_pages = 0

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
        # For hybrid models: restore SSM state from the snapshot of the
        # deepest hit page. Pure-text models skip this branch since
        # ``ssm_segment`` is ``None``.
        self.restore_ssm_snapshot_on_hit(seqs)

    def pre_allocate_page(self, seqs: List[Sequence]):
        for seq in seqs:
            # Update hash of the newly-generated page boundary in decode
            # stage. Same logic as the original implementation, just using
            # the configured key function.
            if seq.computed_prompt and len(seq) % self.page_size == 0:
                self.segment.update(seq, len(seq), seq.page_table[-1])
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

    def restore_ssm_snapshot_on_hit(self, seqs: List[Sequence]):
        if self.ssm_segment is None:
            return
        for seq in seqs:
            if seq.computed_token_num == 0:
                continue
            # Deepest cached page = last entry in ``page_table`` after
            # ``pre_allocate_computed_page`` populated it from hits.
            last_hit_page = seq.page_table[-1]
            snap_slot = self.segment.page2ssm_snapshot[last_hit_page]
            if snap_slot is None:
                # Cached KV exists but no SSM snapshot for that boundary
                # (e.g. the page boundary fell mid-FLA-chunk and was never
                # snapshotted). In that case we cannot honor the SSM-cache
                # hit without redoing the GDN compute; the safe fallback is
                # to drop the KV hit too so attention sees the full prefix
                # and starts from h_0=0. We do this by rolling back the
                # logical state to before the SSM-less hit.
                self._rollback_to_last_ssm_hit(seq)
                continue
            # Make sure the seq has a working slot to receive the snapshot.
            self.allocate_ssm_slot(seq)
            self.ssm_segment.copy_state(
                "snapshot", snap_slot, "working", seq.ssm_state_slot
            )

    def _rollback_to_last_ssm_hit(self, seq: Sequence) -> None:
        """Drop tail KV hits that have no matching SSM snapshot."""
        # Walk back from the deepest hit until we find a page that *does*
        # have a snapshot (or until we hit page 0 = no hit at all). KV
        # pages that we drop here are still ref-counted by the segment, so
        # we have to release them.
        while seq.page_table:
            last_page = seq.page_table[-1]
            if self.segment.page2ssm_snapshot[last_page] is not None:
                # Found a usable boundary; restore SSM from it.
                self.allocate_ssm_slot(seq)
                self.ssm_segment.copy_state(
                    "snapshot",
                    self.segment.page2ssm_snapshot[last_page],
                    "working",
                    seq.ssm_state_slot,
                )
                return
            self.segment.free(seq.page_table.pop())
            seq.computed_token_num -= self.page_size
            self.num_hit_pages -= 1
        # No usable boundary anywhere; computed_token_num is now 0 and the
        # sequence will be processed as a fresh prefill from h_0 = 0.

    def get_cache_hit_rate(self):
        if self.num_allocated_pages == 0:
            return 0.0
        return round(100 * self.num_hit_pages / self.num_allocated_pages, 2)


class PrefixSegment(Segment):
    """Paged KV segment with hash-keyed prefix cache and optional SSM
    snapshot pointers.

    The cache key for a page is produced by ``self.key_fn(seq, n_tokens)``
    (injected by :class:`PrefixMemoryManager` at ``init`` time). For pure
    token-id models that returns the standard prefix tuple; for VL the
    sequence's ``hash_token_ids`` view picks up unique content-derived ids
    spliced into the placeholder positions, so identical-text+different-
    image prompts no longer collide.

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

    # These get set by :class:`PrefixMemoryManager.init`.
    key_fn = staticmethod(_default_cache_key_fn)
    ssm_segment: Optional[SSMSegment] = None

    # Tunable: how many leading ids to keep in the canary. Big enough to
    # make accidental collisions astronomically unlikely while still cheap
    # to compare on every lookup.
    _CANARY_LEN = 8

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hash2page: Dict[int, int] = {}
        self.page_ref_num = [0 for _ in range(self.num_pages)]
        self.page2hash: List[int] = [0 for _ in range(self.num_pages)]
        # Canary stored per physical page; tuples are kept short
        # (``_CANARY_LEN``) to avoid blowing up memory at large num_pages.
        self.page2canary: List[Optional[tuple]] = [None for _ in range(self.num_pages)]
        # SSM snapshot slot id per physical page; ``None`` if no snapshot
        # was captured (e.g. when the SSM cache is disabled or the boundary
        # never had a chance to snapshot during prefill).
        self.page2ssm_snapshot: List[Optional[int]] = [None for _ in range(self.num_pages)]

    # --- helpers ------------------------------------------------------------

    def _canary(self, key: tuple) -> tuple:
        return key[: self._CANARY_LEN]

    def _key_for(self, seq: Sequence, n_tokens: int) -> tuple:
        return self.key_fn(seq, n_tokens)

    # --- public API ---------------------------------------------------------

    def update(self, seq: Sequence, n_tokens: int, page_num: int) -> None:
        """Register a hash for ``page_num`` after its KV was filled in decode."""
        key = self._key_for(seq, n_tokens)
        page_hash = hash(key)
        if page_hash not in self.hash2page:
            self.page2hash[page_num] = page_hash
            self.hash2page[page_hash] = page_num
            self.page2canary[page_num] = self._canary(key)

    def has_computed(self, seq: Sequence, n_tokens: int) -> Optional[int]:
        """Look up a cached page. Returns the page id or ``None`` on miss.

        Performs a canary equality check so two distinct prefixes that happen
        to share a Python ``hash()`` value never silently alias.
        """
        key = self._key_for(seq, n_tokens)
        page_hash = hash(key)
        page_num = self.hash2page.get(page_hash)
        if page_num is None:
            return None
        if self.page2canary[page_num] != self._canary(key):
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
            key = self._key_for(seq, n_tokens)
            page_hash = hash(key)
            key_canary = self._canary(key)

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
        else:
            self.page2hash[page_num] = 0
            self.page2canary[page_num] = None
            self.page2ssm_snapshot[page_num] = None

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
