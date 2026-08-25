from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import torch
import torch.distributed as dist
from logger import logger

from gllm.distributed.parallel_state import get_pp_size
from gllm.runtime.cache_arena import (
    CacheArena,
    CacheLayout,
    CacheTensorLayout,
    RegisteredCache,
)
from gllm.runtime.sequence import GenerationSequence
from gllm.utils import async_tensor_h2d, get_dtype_bytes

# DeepSeek Sparse Attention FP8 MLA cache: the nope latent is quantized in
# 128-wide tiles (one fp32 scale per tile), matching FlashMLA's packed layout.
_DSA_FP8_TILE = 128

# DSA indexer scoring always uses deep_gemm FP8 MQA-logits kernels; decode needs a
# persistent paged FP8 index-K cache in the 132-byte block-contiguous layout
# ``get_paged_mqa_logits_metadata`` / ``fp8_paged_mqa_logits`` expect (per page:
# [page_size*128 fp8 bytes][page_size*4 fp32-scale bytes]).


@dataclass
class SSMCacheConfig:
    """Layout description for the recurrent-state cache used by linear-attention
    (Mamba / Gated DeltaNet) layers.

    ``num_layers`` is the count of *linear-attention* layers on this PP rank,
    *not* the total decoder depth. The full-attention layers continue to use
    the regular paged KV cache (``Segment.k_cache`` / ``v_cache``) and do not
    consume slots here.

    Shapes (per layer, after TP sharding on the head dim):

    * ``conv_state``  : ``(num_slots, conv_dim, conv_kernel - 1)``
    * ``temporal_state``: ``(num_slots, num_v_heads, head_v_dim, head_k_dim)``

    Slot 0 in the working-state arena view is reserved as the CUDA-graph dummy
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

class SSMSegment:
    """GDN/Mamba tensor views and lifecycle over the shared cache arena.

    Working state and prefix snapshots register separate allocation types so
    the arena can treat snapshots as pressure-reclaimable, but both types use
    the same aligned slot grid and the same conv/temporal tensor views. Their
    slot ids are therefore directly usable by the existing GDN kernels. Slot 0
    of the working type is reserved for CUDA-graph padding.

    The segment never allocates storage itself. Both working state and prefix
    snapshots are registered cache layouts over the process-wide arena.
    """

    def __init__(
        self,
        cfg: SSMCacheConfig,
        *,
        state_cache: RegisteredCache,
        snapshot_cache: RegisteredCache,
    ):
        self.cfg = cfg
        if state_cache.arena is not snapshot_cache.arena:
            raise ValueError("SSM working state and snapshots must share one arena")
        if state_cache.layout.tensors != snapshot_cache.layout.tensors:
            raise ValueError("SSM working state and snapshot layouts must match")
        self.cache_arena = state_cache.arena
        self.arena_type = state_cache.name
        self.snapshot_arena_type = snapshot_cache.name
        # Each logical block holds ONE full per-layer
        # GDN recurrent state (conv window + temporal state). A running sequence
        # borrows one block for its rolling state; an MTP verify step transiently
        # borrows ``k`` extra blocks per seq for the per-token checkpoints; a
        # prefix-cached prefix keeps its state in a reclaimable arena slot. The
        # cached state is copied into a fresh working slot on a hit, so GDN's
        # in-place updates never touch the cached copy.
        self.num_blocks = state_cache.num_slots
        conv_shape = cfg.conv_state_shape_per_slot()
        temp_shape = cfg.temporal_state_shape_per_slot()

        # Layout: ``[num_layers, num_blocks, *per_block]`` as a single stacked
        # tensor view. ``conv_state[layer_id]`` still returns that layer's
        # ``[num_blocks, *per_block]`` slice, now with an arena entry stride.
        # Kernels consume the explicit stride. The stacked view lets
        # ``commit_blocks`` copy
        # the checkpoint across ALL layers in one ``index_copy_`` (2 kernel
        # launches total) instead of ``2 * num_layers`` per-layer launches.
        # Registered tensors are slot-major; recurrent kernels use layer-major
        # views. ``movedim`` changes only metadata and retains the stable arena
        # entry stride required by CUDA Graph replay.
        self.conv_state = state_cache.tensor("conv_state").movedim(1, 0)
        self.temporal_state = state_cache.tensor("temporal_state").movedim(1, 0)
        if self.conv_state.shape != (cfg.num_layers, self.num_blocks, *conv_shape):
            raise ValueError(self.conv_state.shape)
        if self.temporal_state.shape != (
            cfg.num_layers,
            self.num_blocks,
            *temp_shape,
        ):
            raise ValueError(self.temporal_state.shape)

        dummy = self.cache_arena.allocator.allocate(self.arena_type, slot=0)
        if dummy != [0]:
            raise RuntimeError(f"cache arena did not reserve SSM frame 0: {dummy}")
        self.conv_state[:, 0].zero_()
        self.temporal_state[:, 0].zero_()

        # Dummy slot that padded rows / unused pointers can refer to without
        # aliasing any real state.
        self.dummy_working_slot: int = 0

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

    # --- block lifecycle ------------------------------------------------
    #
    # A "block" holds one full per-layer GDN recurrent state. Sequences borrow
    # one block for their rolling state; MTP verify borrows extra transient
    # blocks for per-token checkpoints.

    def allocate_block(self) -> Optional[int]:
        blocks = self.cache_arena.allocator.allocate(self.arena_type, 1)
        if blocks is None:
            return None
        block = blocks[0]
        self._zero_block(block)
        return block

    def _zero_block(self, block: int) -> None:
        def _zero():
            self.conv_state[:, block].zero_()
            self.temporal_state[:, block].zero_()

        if self.restore_stream is not None:
            with torch.cuda.stream(self.restore_stream):
                _zero()
        else:
            _zero()

    def free_block(self, block: int) -> None:
        if block is None or block == self.dummy_working_slot:
            return
        # Zero before returning so the next borrower starts from h_0 = 0
        # without needing an explicit "reset state" pass through every layer.
        # Stacked layout -> one ``zero_`` per state covers all layers.
        self._zero_block(block)
        self.cache_arena.allocator.free(self.arena_type, [block])

    def num_free_blocks(self) -> int:
        return self.cache_arena.allocator.num_available_slots(self.arena_type)

    def allocate_block_table(self, n: int) -> Optional[list]:
        """Borrow ``n`` blocks for a sequence's SSM state block table.

        Speculative decode gives each sequence a fixed ``1+k`` block table:
        column 0 holds the rolling/committed state and columns 1..k hold verify
        checkpoints. Returns a list of ``n`` block ids, or ``None`` if
        the arena cannot satisfy the whole request. Allocation is atomic.
        """
        blocks = self.cache_arena.allocator.allocate(self.arena_type, n)
        if blocks is None:
            return None
        for block in blocks:
            self._zero_block(block)
        return blocks

    def free_block_table(self, blocks) -> None:
        """Return a sequence's whole SSM block table to the arena."""
        if not blocks:
            return
        real_blocks = [
            int(blk) for blk in blocks
            if blk is not None and int(blk) != self.dummy_working_slot
        ]
        for blk in real_blocks:
            self._zero_block(blk)
        self.cache_arena.allocator.free(self.arena_type, real_blocks)

    # --- prefix-cache cached-state blocks ------------------------------

    def allocate_snapshot(self) -> Optional[int]:
        # Snapshot allocations are best effort. The allocator may use any free
        # aligned extent, but does not evict one snapshot merely to create a
        # different snapshot of the same type.
        blocks = self.cache_arena.allocator.allocate(self.snapshot_arena_type, 1)
        if blocks is None:
            return None
        block = blocks[0]
        self._zero_block(block)
        return block

    def free_snapshot(self, slot: int) -> None:
        if slot is None or slot == self.dummy_working_slot:
            return
        self._zero_block(slot)
        self.cache_arena.allocator.free(self.snapshot_arena_type, [slot])

    def num_free_snapshot(self) -> int:
        return self.num_free_blocks()

    # --- transfer -------------------------------------------------------

    def copy_state(
        self,
        src_kind: str,
        src_slot: int,
        dst_kind: str,
        dst_slot: int,
    ) -> None:
        """Copy a full multi-layer recurrent state between two arena entries.

        ``src_kind``/``dst_kind`` ("working"/"snapshot") are semantic labels
        for the copy direction; both index the same strided tensor view.

        * Prefill capture: ``copy_state("working", req_block, "snapshot",
          cached_block)`` after the GDN layer crosses a cacheable page boundary.
        * Prefix-cache hit restore: ``copy_state("snapshot", cached_block,
          "working", req_block)`` before the new request's first forward, giving
          it a private mutable copy (copy-on-write; the cached block stays
          read-only for other future hits).
        """
        src_conv, src_temp = self._pool(src_kind)
        dst_conv, dst_temp = self._pool(dst_kind)
        if src_conv is None or dst_conv is None:
            return

        def _do_copies():
            # Stacked ``[num_layers, num_blocks, *]`` -> copy every layer's slot
            # in one op per state (2 launches) instead of ``2 * num_layers``.
            dst_conv[:, dst_slot].copy_(src_conv[:, src_slot])
            dst_temp[:, dst_slot].copy_(src_temp[:, src_slot])

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
        # Working state and snapshots use the same logical grid over the same
        # physical arena. The kind only documents capture vs restore; the two
        # entries always have distinct live ownership.
        if kind in ("working", "snapshot"):
            return self.conv_state, self.temporal_state
        raise ValueError(f"unknown SSM state kind: {kind!r}")

    # --- MTP verify checkpoint commit ----------------------------------
    #
    # An MTP verify forward runs the GDN recurrent kernel over [x1, d1..dk] and
    # checkpoints the state after each token into transient arena entries (one
    # entry per verify step, per sequence).
    # The verify forward does NOT write the sequence's rolling block (it passes
    # ``disable_state_update``). After the accept step knows each seq committed
    # ``1+na`` tokens, we copy the step-``na`` checkpoint block's contents into
    # the sequence's rolling block -- the exact post-commit recurrent state,
    # with no rollback and no recompute forward. The transient blocks are then
    # returned to the arena. One rolling block remains the source of
    # truth so ordinary one-token decode is unchanged; the selected checkpoint
    # is committed there before the transient blocks are released.

    def commit_blocks(self, commit) -> None:
        """Copy chosen checkpoint blocks into rolling blocks (batched).

        ``commit`` is a list of ``(dst_block, src_block)`` pairs. For each, the
        full per-layer state at ``src_block`` is copied into ``dst_block``.
        With the stacked ``[num_layers, num_blocks, *]`` layout the copy is one
        ``index_copy_`` over the block dim (dim=1) across ALL layers at once --
        2 kernel launches total (conv + temporal) instead of ``2 * num_layers``.
        """
        if not commit:
            return
        dev = self.conv_state.device
        dst = torch.as_tensor([c[0] for c in commit], dtype=torch.long, device=dev)
        src = torch.as_tensor([c[1] for c in commit], dtype=torch.long, device=dev)
        self.conv_state.index_copy_(1, dst, self.conv_state.index_select(1, src))
        self.temporal_state.index_copy_(
            1, dst, self.temporal_state.index_select(1, src)
        )


class Segment:
    def __init__(
        self,
        num_layers: int,
        page_size: int,
        kv_head_num: int,
        kv_head_dim: int,
        use_mla: bool,
        cache: RegisteredCache,
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
        self.num_pages = cache.num_slots
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
        if not use_mla:
            keys = cache.tensor("key")
            values = cache.tensor("value")
            expected = (
                self.num_pages,
                num_layers,
                page_size,
                kv_head_num,
                kv_head_dim,
            )
            if keys.shape != expected or values.shape != expected:
                raise ValueError((keys.shape, values.shape, expected))
            self.k_cache = [keys[:, layer] for layer in range(num_layers)]
            self.v_cache = [values[:, layer] for layer in range(num_layers)]
        else:
            latent = cache.tensor("mla")
            if self.mla_cache_fp8:
                self.mla_fp8_dim = latent.shape[-1]
            self.kv_cache = [latent[:, layer] for layer in range(num_layers)]
        # DeepSeek Sparse Attention: parallel indexer key cache (bf16, one
        # single-head index_head_dim vector per token per layer). Only
        # allocated when index_head_dim > 0.
        if index_head_dim > 0:
            index = cache.tensor("index_key")
            self.index_k_cache = [index[:, layer] for layer in range(num_layers)]
            # DSA FP8 indexer scoring: a parallel paged FP8 index-K cache in the
            # 132-byte block-contiguous layout the deep_gemm paged-MQA-logits
            # kernel reads (per page: [page_size*index_head_dim fp8][page_size*
            # (index_head_dim/128)*4 scale]). ``index_head_dim`` (128) => 128 fp8
            # + 4 scale = 132 bytes/token.
            index_fp8 = cache.tensor("index_key_fp8")
            self.index_fp8_bytes = index_fp8.shape[-1] // page_size
            self.index_k_fp8_cache = [
                index_fp8[:, layer] for layer in range(num_layers)
            ]
        else:
            self.index_k_cache = None
            self.index_k_fp8_cache = None
        self.id_allocator = cache.slot_allocator()

    def allocate(self):
        pagenum = self.id_allocator.allocate()
        return pagenum

    def free(self, page_num: int):
        self.id_allocator.free(page_num)

    def free_many(self, page_nums) -> None:
        self.id_allocator.free_many(int(page) for page in page_nums)

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
        max_running_seqs: int = 256,
        index_head_dim: int = 0,
        qk_rope_head_dim: int = 0,
        mla_cache_fp8: bool = False,
        mtp_k: int = 0,
        ssm_snapshot_stride_tokens: int = 256,
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
                cache. ``None`` registers only the attention cache in the arena.
            ssm_snapshot_stride_tokens: token granularity of recurrent-state
                prefix caching, rounded down to whole KV pages (see
                ``PrefixSegment.ssm_snapshot_stride``). Smaller = finer restore
                points but more reclaimable arena entries per prompt.
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
        # Draft-chain length (mtp_k); a running seq borrows up to this many
        # transient checkpoint blocks during an MTP verify step (0 = MTP off).
        self.mtp_k = mtp_k
        # Recurrent-state prefix-cache granularity, in TOKENS. Converted to
        # whole pages and installed on the segment by
        # ``PrefixMemoryManager.init``; ignored without prefix caching.
        self.ssm_snapshot_stride_tokens = ssm_snapshot_stride_tokens
        # Populated by :meth:`init`; ``None`` when the model is not hybrid.
        self.ssm_segment: Optional[SSMSegment] = None
        self.cache_arena: Optional[CacheArena] = None
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
        """No SSM prefix caching without prefix metadata (base manager)."""
        return {}

    def init(self, segment_cls=Segment, reserve_dummy_page: bool = False):
        """Allocate one arena and register every persistent model cache."""
        kv_layout = self._kv_cache_layout()
        free_mem, _ = torch.cuda.mem_get_info()
        kv_page_bytes = kv_layout.entry_bytes
        arena_budget = int(free_mem * self.gpu_memory_util)
        num_physical_pages = arena_budget // kv_page_bytes
        if dist.is_initialized():
            gathered = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered, num_physical_pages)
            num_physical_pages = min(gathered)
        if num_physical_pages <= 0:
            raise RuntimeError("not enough GPU memory for one cache arena page")

        device = torch.device("cuda", torch.cuda.current_device())
        backing = torch.empty(
            num_physical_pages * kv_page_bytes,
            dtype=torch.uint8,
            device=device,
        )
        arena = CacheArena(backing, physical_page_bytes=kv_page_bytes)
        kv_cache = arena.register_cache(kv_layout)
        self.cache_arena = arena
        self.num_pages = kv_cache.num_slots
        self.segment = segment_cls(
            self.num_layers,
            self.page_size,
            self.kv_head_num,
            self.kv_head_dim,
            self.use_mla,
            kv_cache,
            index_head_dim=self.index_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            mla_cache_fp8=self.mla_cache_fp8,
        )

        if self.ssm_cache_config is not None:
            cfg = self.ssm_cache_config
            state_cache = arena.register_cache(self._ssm_cache_layout("ssm_state"))
            snapshot_cache = arena.register_cache(
                self._ssm_cache_layout("ssm_snapshot")
            )
            per_seq_blocks = 1 + self.mtp_k
            # Slot 0 is the graph-padding state and is never allocatable. This
            # checks viability only; it does not reserve a separate SSM pool.
            usable_ssm_slots = max(0, state_cache.num_slots - 1)
            if usable_ssm_slots < per_seq_blocks:
                need = (per_seq_blocks + 1) * state_cache.entry_bytes
                raise RuntimeError(
                    f"cache arena needs >= {need / (1 << 30):.1f} GB to run "
                    f"one request with MTP k={self.mtp_k}, but its total budget "
                    f"is {backing.numel() / (1 << 30):.1f} GB. Raise "
                    "--gpu-memory-util or --tp."
                )
            self.ssm_segment = SSMSegment(
                cfg,
                state_cache=state_cache,
                snapshot_cache=snapshot_cache,
            )
        else:
            usable_ssm_slots = None

        self.dummy_page = self.segment.allocate() if reserve_dummy_page else None

        cache_names = ", ".join(arena.allocator.cache_type_names)
        logger.info(
            "Cache arena: %.2f GB, physical page %.2f KB; registered caches: "
            "%s; KV=%d pages (%d tokens/page, %.2f KB/token)",
            backing.numel() / (1 << 30),
            kv_page_bytes / 1024,
            cache_names,
            self.num_pages,
            self.page_size,
            kv_page_bytes / (1024 * self.page_size),
        )
        if usable_ssm_slots is not None:
            logger.info(
                "Shared SSM capacity: %d slots, MTP state demand=%d slots/request",
                usable_ssm_slots,
                1 + self.mtp_k,
            )

        self.kv_cache_dtype = "auto"
        self.k_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
        self.v_scale = self.k_scale

    def _kv_cache_layout(self) -> CacheLayout:
        """Describe all tensor banks that share one attention-cache page id."""
        tensors = []
        if not self.use_mla:
            shape = (
                self.num_layers,
                self.page_size,
                self.kv_head_num,
                self.kv_head_dim,
            )
            tensors.extend(
                (
                    CacheTensorLayout("key", self.dtype, shape),
                    CacheTensorLayout("value", self.dtype, shape),
                )
            )
        else:
            if self.mla_cache_fp8:
                qk_rope = self.qk_rope_head_dim
                kv_lora = self.kv_head_dim - qk_rope
                if kv_lora % _DSA_FP8_TILE:
                    raise ValueError(
                        f"kv_lora_rank {kv_lora} must be divisible by "
                        f"{_DSA_FP8_TILE} for the DSA FP8 MLA cache"
                    )
                mla_dim = (
                    kv_lora
                    + (kv_lora // _DSA_FP8_TILE) * 4
                    + qk_rope * get_dtype_bytes(self.dtype)
                )
                tensors.append(
                    CacheTensorLayout(
                        "mla",
                        torch.float8_e4m3fn,
                        (self.num_layers, self.page_size, 1, mla_dim),
                    )
                )
            else:
                tensors.append(
                    CacheTensorLayout(
                        "mla",
                        self.dtype,
                        (self.num_layers, self.page_size, self.kv_head_dim),
                    )
                )

        if self.index_head_dim > 0:
            if self.index_head_dim % _DSA_FP8_TILE:
                raise ValueError(
                    f"index_head_dim {self.index_head_dim} must be divisible by "
                    f"{_DSA_FP8_TILE}"
                )
            index_fp8_bytes = self.index_head_dim + (
                self.index_head_dim // _DSA_FP8_TILE
            ) * 4
            tensors.extend(
                (
                    CacheTensorLayout(
                        "index_key",
                        self.dtype,
                        (self.num_layers, self.page_size, self.index_head_dim),
                    ),
                    CacheTensorLayout(
                        "index_key_fp8",
                        torch.uint8,
                        (
                            self.num_layers,
                            self.page_size * index_fp8_bytes,
                        ),
                    ),
                )
            )
        return CacheLayout("kv_cache", tuple(tensors))

    def _ssm_cache_layout(self, name: str) -> CacheLayout:
        cfg = self.ssm_cache_config
        if cfg is None:
            raise RuntimeError("cannot register SSM cache without its config")
        return CacheLayout(
            name,
            (
                CacheTensorLayout(
                    "conv_state",
                    cfg.conv_state_dtype,
                    (cfg.num_layers, *cfg.conv_state_shape_per_slot()),
                ),
                CacheTensorLayout(
                    "temporal_state",
                    cfg.dtype,
                    (cfg.num_layers, *cfg.temporal_state_shape_per_slot()),
                ),
            ),
            prefer_high=True,
        )

    def get_sizeof_KV_per_page(self):  # Bytes
        return self._kv_cache_layout().entry_bytes

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

    def store_index_k_fp8(
        self,
        layer_idx: int,
        index_k: torch.Tensor,
        slot_mapping_tensor: torch.Tensor,
        use_ue8m0: bool = False,
    ):
        """Quantize + write the indexer key into the paged FP8 index cache.

        Companion to :meth:`store_index_k` for the DSA FP8 scoring path: writes
        ``index_k`` ``[num_tokens, index_head_dim]`` into the 132-byte
        block-contiguous paged FP8 index cache that ``fp8_paged_mqa_logits``
        reads. ``use_ue8m0`` rounds the per-token scale to a power of two (set
        by the caller from the checkpoint's ``scale_fmt``).
        """
        from gllm import _custom_ops as ops

        cache = self.segment.index_k_fp8_cache[layer_idx]
        ops.store_index_k_fp8(
            index_k,
            cache,
            slot_mapping_tensor,
            self.segment.page_size,
            self.segment.index_head_dim,
            use_ue8m0=use_ue8m0,
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

    def pre_allocate_page(self, seqs: List[GenerationSequence], cacheable: bool = True):
        # Base manager has no prefix cache; ``cacheable`` is accepted for a
        # uniform signature with ``PrefixMemoryManager`` and ignored.
        for seq in seqs:
            num_page = (seq.seq_len + self.page_size - 1) // self.page_size - len(
                seq.page_table
            )
            for _ in range(num_page):
                seq.page_table.append(self.segment.allocate())

    def pre_allocate_page_for_lengths(
        self, seqs: List[GenerationSequence], seq_lens: List[int]
    ) -> None:
        """Grow page tables to explicit lengths without mutating token lists.

        MTP reserves its whole speculative window before draft/verify. Those
        tokens are not committed and must not enter prefix-cache hashing, so
        allocation only needs the target lengths and can avoid constructing an
        O(context) temporary ``token_ids`` list for every sequence.
        """
        if len(seqs) != len(seq_lens):
            raise ValueError((len(seqs), len(seq_lens)))
        for seq, seq_len in zip(seqs, seq_lens):
            num_page = (int(seq_len) + self.page_size - 1) // self.page_size - len(
                seq.page_table
            )
            for _ in range(num_page):
                seq.page_table.append(self.segment.allocate())

    def register_decode_boundary(self, seq: GenerationSequence, pos: int) -> None:
        """No-op without a prefix cache; overridden by ``PrefixMemoryManager``."""
        return

    def free(self, seq: GenerationSequence):
        self.segment.free_many(seq.page_table)
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
        new_rows = torch.ones((extra, self.vocab_size), dtype=self.dtype, device="cuda")
        self._rep_pool = torch.cat([self._rep_pool, new_rows], dim=0)
        self._rep_free_slots.extend(range(old_rows, old_rows + extra))

    def free_rep_slot(self, seq: GenerationSequence) -> None:
        if seq.rep_slot is None:
            return
        # Lazy reset: the row is re-filled with ones when the slot is handed
        # to the next seq (see ``build_repetition_penalty_mask``), so we only
        # need to return the id and clear the per-seq bookkeeping here.
        if self._rep_free_slots is not None:
            self._rep_free_slots.append(seq.rep_slot)
        seq.rep_slot = None
        seq.rep_filled = 0

    def build_repetition_penalty_mask(self, seqs: List[GenerationSequence]):
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
            (
                seq.rep_slot
                if (
                    getattr(seq, "repetition_penalty", 1.0) != 1.0
                    and seq.rep_slot is not None
                )
                else 0
            )
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

    def allocate_ssm_slot(self, seq: GenerationSequence) -> bool:
        if self.ssm_segment is None:
            return True
        if self.mtp_k > 0:
            # MTP on: give the sequence a fixed 1+k block table (column 0 is
            # rolling state; the remaining columns are verify checkpoints).
            if seq.ssm_block_table is not None:
                return True
            bt = self.ssm_segment.allocate_block_table(1 + self.mtp_k)
            if bt is None:
                return False  # arena exhausted; scheduler gates admission
            seq.ssm_block_table = bt
            # Column 0 is also the sequence's committed-state slot, shared by
            # ordinary decode and prefix-cache snapshot restore.
            seq.ssm_state_slot = bt[0]
            seq.ssm_num_accepted = 1
            return True
        else:
            if seq.ssm_state_slot is not None:
                return True
            slot = self.ssm_segment.allocate_block()
            if slot is None:
                return False
            seq.ssm_state_slot = slot
            return True

    def free_ssm_slot(self, seq: GenerationSequence) -> None:
        if self.ssm_segment is None:
            return
        if seq.ssm_block_table is not None:
            self.ssm_segment.free_block_table(seq.ssm_block_table)
            seq.ssm_block_table = None
            seq.ssm_state_slot = None
            seq.ssm_num_accepted = 1
            return
        if seq.ssm_state_slot is None:
            return
        self.ssm_segment.free_block(seq.ssm_state_slot)
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


def _hash_source(seq: GenerationSequence) -> List[int]:
    """Pick the token list used for prefix-cache hashing.

    ``hash_token_ids`` (set by the multimodal pipeline) wins over the raw
    ``token_ids`` so two VL prompts with the same ``<|image_pad|>``
    placeholders but distinct images do not collide.
    """
    hi = getattr(seq, "hash_token_ids", None)
    return hi if hi is not None else seq.token_ids


def _maybe_invalidate_seq_hash_cache(seq: GenerationSequence, src: List[int]) -> None:
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


def _ensure_page_hash(seq: GenerationSequence, page_size: int, page_idx: int) -> int:
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
        page_tokens = tuple(src[i * page_size : (i + 1) * page_size])
        cache.append(hash((prev, page_tokens)))
    return cache[page_idx]


def _ensure_canary(seq: GenerationSequence) -> tuple:
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
    ``GenerationSequence`` via ``_ensure_page_hash``: extending the chain by one page
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
        self.cache_arena.allocator.set_evictor(
            "kv_cache", self.segment.evict_arena_slot
        )
        if self.ssm_segment is not None:
            self.cache_arena.allocator.set_reclaimer(
                "ssm_snapshot", self.segment.reclaim_one_ssm_snapshot
            )
        # Recurrent-state caching granularity for this run. Rounded DOWN to
        # whole pages (only page boundaries can carry a snapshot) with a floor
        # of one page. A request below ``page_size`` therefore degrades to
        # per-page snapshots, which increases state-copy work and arena churn.
        stride_tokens = int(self.ssm_snapshot_stride_tokens)
        if stride_tokens < self.page_size:
            logger.warning(
                "ssm_snapshot_stride_tokens=%d is below page_size=%d; clamping to "
                "one page. Recurrent-state caching may materialize a reclaimable "
                "snapshot every %d prompt tokens.",
                stride_tokens,
                self.page_size,
                self.page_size,
            )
        self.segment.ssm_snapshot_stride = max(1, stride_tokens // self.page_size)
        if self.ssm_segment is not None:
            logger.info(
                "SSM snapshot stride: %d tokens (%d pages)",
                self.segment.ssm_snapshot_stride * self.page_size,
                self.segment.ssm_snapshot_stride,
            )

        # Cache-hit-rate stats.
        self.num_allocated_pages = 0
        self.num_hit_pages = 0

        # PP>1 only: SSM snapshot restores performed this scheduling iteration,
        # keyed by seq_id -> snapshot arena slot. Each PP follower owns a
        # *different* slice of the GDN layers on its own GPU, so the restore
        # (snapshot->working ``copy_state``) the driver runs on rank-0's views
        # must be replayed on every stage. The driver records the restores here
        # and the payload builder ships them; ``consume`` clears the buffer so
        # each is shipped exactly once.
        self._pending_ssm_restores: Dict[int, int] = {}

    def pre_allocate_computed_page(self, seqs: List[GenerationSequence]):
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

    def _finalize_prefix_cache_hit(self, seq: GenerationSequence) -> None:
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
        # KNOWN RESIDUAL (MTP): a draft head sharing these pages stores a
        # SHIFTED entry -- position ``p`` holds
        # ``(target_hidden[p], embed(token[p+1]))``.  At the deepest cached
        # position ``C-1`` that next token is this request's first token
        # OUTSIDE the shared prefix, so exactly ONE reused head entry was
        # written for someone else's continuation (every shallower entry is
        # genuinely reusable).  Handing a token back would repair it, but on a
        # hybrid model ``computed_token_num`` is pinned to an SSM *snapshot*
        # boundary: giving back one page makes ``_restore_ssm_working_state``
        # walk to the previous snapshot and discard a whole
        # ``ssm_snapshot_stride_tokens`` window.  Measured on Qwen3.8-27B with
        # a prefix-repetition workload that is -36% output throughput to buy
        # +0.01 acceptance length, so the stale entry is deliberately kept.
        if is_hybrid:
            if full_hit:
                seq.computed_token_num -= self.page_size
                self.num_hit_pages -= 1
            self._restore_ssm_working_state(seq)
        elif full_hit:
            seq.computed_token_num = len(seq) - 1

    def _restore_ssm_working_state(self, seq: GenerationSequence) -> None:
        """Copy the deepest *filled* SSM snapshot at/below ``computed_token_num``.

        Cached KV pages stay in ``page_table`` and are recomputed in place by
        the upcoming forward (idempotent for full-attention KV). We never
        free/pop pages here -- a sibling request in the same batch may share
        them.
        """
        while seq.computed_token_num > 0:
            boundary_page = seq.page_table[seq.computed_token_num // self.page_size - 1]
            snap_slot = self._valid_snapshot_slot(boundary_page)
            if snap_slot is not None:
                if not self.allocate_ssm_slot(seq):
                    # The prefix remains cached, but without a private mutable
                    # state slot this request must wait for arena capacity.
                    hit_pages = seq.computed_token_num // self.page_size
                    seq.computed_token_num = 0
                    self.num_hit_pages = max(0, self.num_hit_pages - hit_pages)
                    return
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

    def register_decode_boundary(self, seq: GenerationSequence, pos: int) -> None:
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

    def pre_allocate_page(self, seqs: List[GenerationSequence], cacheable: bool = True):
        """Grow each seq's page table to cover its current ``seq_len``.

        ``cacheable`` (default True) registers a prefix-cache hash for any page
        that closes a complete ``page_size`` boundary, so a later seq sharing
        that prefix can hit it. MTP draft/verify must pass ``cacheable=False``:
        those calls run over SPECULATIVE token_ids (unverified drafts) that are
        mostly rejected, so hashing them would poison ``hash2page`` with entries
        keyed on tokens that never get committed -- a sibling seq could then hit
        a stale mapping and corrupt the page ref-count (double-free). The real
        hash is registered later over committed tokens via
        ``register_decode_boundary`` (from the scheduler's finalize hook).
        """
        for seq in seqs:
            seq_cacheable = cacheable and not getattr(seq, "_mtp_async_pending", False)
            len_page_table = len(seq.page_table)
            num_page = (
                seq.seq_len + self.page_size - 1
            ) // self.page_size - len_page_table
            for i in range(len_page_table, len_page_table + num_page):
                if seq_cacheable and (i + 1) * self.page_size <= len(seq):
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
        if page_num in self.segment._ssm_snapshot_lru:
            self.segment._ssm_snapshot_lru.move_to_end(page_num)
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
    chains a per-page hash on the ``GenerationSequence`` itself; for VL the
    sequence's ``hash_token_ids`` view feeds the chain so identical-text +
    different-image prompts no longer collide.

    Collision safety: ``hash2page`` maps Python's tuple-hash to a page; on
    lookup we additionally compare the canary (first 8 ids of the cached
    prefix) before declaring a hit. Without the canary, the previous
    implementation could silently share KV across two distinct prefixes
    whose ``hash()`` happened to match.

    SSM extension: every cached page may reference a snapshot entry in the
    partner :class:`SSMSegment`. The entry is borrowed lazily only when a
    prefill forward reaches an eligible page boundary, and remains reclaimable
    under arena pressure. On a cache hit the snapshot is copied back into the
    requesting sequence's working entry.
    """

    # Set by :class:`PrefixMemoryManager.init`.
    ssm_segment: Optional[SSMSegment] = None

    # Recurrent-state caching granularity, in PAGES: only page boundaries at a
    # multiple of this stride can hold a cached state. Installed by
    # :meth:`PrefixMemoryManager.init` from ``--ssm-snapshot-stride-tokens``;
    # the value here is only a floor for a segment built outside that path.
    #
    # Only boundaries that a prefill chunk actually reaches are materialized;
    # reserving entries at allocation time would leave most of them unwritten.
    # A coarse stride reduces state-copy traffic and metadata. It is not a
    # capacity partition: every materialized snapshot freely borrows an arena
    # extent and remains pressure-reclaimable.
    ssm_snapshot_stride: int = 1

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
        self.page2ssm_snapshot: List[Optional[int]] = [
            None for _ in range(self.num_pages)
        ]
        # Whether the snapshot slot actually holds a *written* recurrent state.
        # A slot only ever gets written for the boundary a prefill chunk *ends*
        # on (see ``InputData._cal_ssm_metadata``); this flag separates
        # "reserved" from "filled" so the restore path never grafts a zeroed
        # state (== h_0) onto a non-empty prefix.
        self.page2ssm_snapshot_valid: List[bool] = [
            False for _ in range(self.num_pages)
        ]
        self._ssm_snapshot_lru: "OrderedDict[int, None]" = OrderedDict()

    # --- public API ---------------------------------------------------------

    def evict_arena_slot(self, page_num: int) -> None:
        """Invalidate an unpinned KV entry before another arena type reuses it."""
        if self.page_ref_num[page_num] != 0:
            raise RuntimeError(
                f"arena attempted to evict pinned KV page {page_num} "
                f"(refs={self.page_ref_num[page_num]})"
            )
        page_hash = self.page2hash[page_num]
        if page_hash and self.hash2page.get(page_hash) == page_num:
            del self.hash2page[page_hash]
        self._release_snapshot_for(page_num)
        self.page2hash[page_num] = 0
        self.page2canary[page_num] = None
        self.page2ssm_snapshot[page_num] = None
        self.page2ssm_snapshot_valid[page_num] = False

    def reclaim_one_ssm_snapshot(self) -> bool:
        """Release the least-recently-used snapshot under arena pressure."""
        while self._ssm_snapshot_lru:
            page_num, _ = self._ssm_snapshot_lru.popitem(last=False)
            if self.page2ssm_snapshot[page_num] is None:
                continue
            self._release_snapshot_for(page_num)
            return True
        return False

    def update(self, seq: GenerationSequence, n_tokens: int, page_num: int) -> None:
        """Register a hash for ``page_num`` after its KV was filled in decode."""
        page_idx = n_tokens // self.page_size - 1
        page_hash = _ensure_page_hash(seq, self.page_size, page_idx)
        if page_hash not in self.hash2page:
            self.page2hash[page_num] = page_hash
            self.hash2page[page_hash] = page_num
            self.page2canary[page_num] = _ensure_canary(seq)

    def has_computed(self, seq: GenerationSequence, n_tokens: int) -> Optional[int]:
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

    def allocate(
        self, seq: Optional[GenerationSequence] = None, n_tokens: Optional[int] = None
    ):
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
            # NO state block is reserved here. Reservation is lazy and coarse --
            # see ``reserve_ssm_snapshot`` and ``ssm_snapshot_stride``.
            self.page2ssm_snapshot[page_num] = None
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
            # the same lifetime — otherwise serial reuse of a cached
            # prompt would always lose the snapshot half of the hit and
            # ``_restore_ssm_working_state`` would drop the KV half too.
            self.id_allocator.free(page_num)

    def free_many(self, page_nums) -> None:
        """Drop a request's KV references and batch-release newly unpinned pages."""
        released = []
        for value in page_nums:
            page_num = int(value)
            assert self.page_ref_num[page_num] > 0
            self.page_ref_num[page_num] -= 1
            if self.page_ref_num[page_num] == 0:
                released.append(page_num)
        if not released:
            return
        self.id_allocator.free_many(released)

    def reserve_ssm_snapshot(self, page_num: int, n_tokens: int) -> Optional[int]:
        """Lazily borrow a cached-state entry for ``page_num``, or ``None``.

        Called from the *write* path (``InputData._cal_ssm_metadata``) for the
        boundary a prefill chunk just landed on, so an entry is only ever taken
        for a boundary that will actually hold a state. Returns ``None`` when:

        * the boundary is not on the coarse ``ssm_snapshot_stride`` grid,
        * the page is not cacheable (nothing could ever hit it), or
        * no arena extent is currently available. Snapshots are reclaimable;
          live KV/working-state pressure can evict them later without a fixed
          watermark or reserved sub-pool.
        """
        if self.ssm_segment is None:
            return None
        if n_tokens <= 0 or n_tokens % (self.ssm_snapshot_stride * self.page_size):
            return None
        if self.page2hash[page_num] == 0:
            return None
        slot = self.page2ssm_snapshot[page_num]
        if slot is not None:
            return slot
        slot = self.ssm_segment.allocate_snapshot()
        if slot is None:
            return None
        self.page2ssm_snapshot[page_num] = slot
        self.page2ssm_snapshot_valid[page_num] = False
        self._ssm_snapshot_lru[page_num] = None
        return slot

    def _release_snapshot_for(self, page_num: int) -> None:
        if self.ssm_segment is None:
            return
        snap_slot = self.page2ssm_snapshot[page_num]
        if snap_slot is not None:
            self.ssm_segment.free_snapshot(snap_slot)
            self.page2ssm_snapshot[page_num] = None
            self._ssm_snapshot_lru.pop(page_num, None)
        self.page2ssm_snapshot_valid[page_num] = False
