from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import os

import torch
import torch.distributed as dist
from logger import logger

from collections import deque

from gllm.distributed.parallel_state import get_pp_size
from gllm.runtime.id_allocator import IDAllocator
from gllm.runtime.sequence import GenerationSequence
from gllm.utils import async_tensor_h2d, get_dtype_bytes

# DeepSeek Sparse Attention FP8 MLA cache: the nope latent is quantized in
# 128-wide tiles (one fp32 scale per tile), matching FlashMLA's packed layout.
_DSA_FP8_TILE = 128

# DSA indexer scoring backend (mirrors gllm/models/deepseek_v32.py). Default on:
# the indexer scores via deep_gemm FP8 MQA-logits kernels; decode needs a
# persistent paged FP8 index-K cache in the 132-byte block-contiguous layout
# ``get_paged_mqa_logits_metadata`` / ``fp8_paged_mqa_logits`` expect (per page:
# [page_size*128 fp8 bytes][page_size*4 fp32-scale bytes]). Set
# ``GLLM_DSA_FP8_SCORE=0`` to fall back to fp32 einsum scoring (no FP8 cache).
_DSA_FP8_SCORE = os.environ.get("GLLM_DSA_FP8_SCORE", "1") == "1"


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
        num_blocks: int,
    ):
        self.cfg = cfg
        # Shared recurrent-state block pool. Each block holds ONE full per-layer
        # GDN recurrent state (conv window + temporal state). A running sequence
        # borrows one block for its rolling state; an MTP verify step transiently
        # borrows ``k`` extra blocks per seq for the per-token checkpoints; a
        # prefix-cached prefix keeps its state in a ref-counted block borrowed
        # from THIS SAME pool (there is no separate snapshot pool anymore -- the
        # cached-state block lives here and is copied into a fresh working block
        # on a cache hit, so GDN's in-place updates never touch the cached copy).
        # The pool size is derived from the memory budget, NOT from
        # ``maxd``; max concurrency is *bounded by* ``num_blocks``.
        # +1 keeps block 0 reserved as the CUDA-graph dummy block.
        self.num_blocks = num_blocks + 1
        # Back-compat alias: some call sites / logs still read working_pool_size.
        self.working_pool_size = self.num_blocks

        conv_shape = cfg.conv_state_shape_per_slot()
        temp_shape = cfg.temporal_state_shape_per_slot()
        device = torch.device("cuda", torch.cuda.current_device())

        # Layout: ``[num_layers, num_blocks, *per_block]`` as a single stacked
        # tensor (not a Python list of per-layer tensors). ``conv_state[layer_id]``
        # still returns that layer's ``[num_blocks, *per_block]`` slice -- a
        # contiguous view -- so every per-layer call site (kernels, ``copy_state``,
        # ``zero_``) is unchanged. The stacked layout lets ``commit_blocks`` copy
        # the checkpoint across ALL layers in one ``index_copy_`` (2 kernel
        # launches total) instead of ``2 * num_layers`` per-layer launches.
        # ``torch.zeros`` (not ``empty``) because the SSM kernels read from
        # block 0 / freshly-allocated blocks before the first write and require a
        # clean initial state (h_0 = 0).
        self.conv_state = torch.zeros(
            (cfg.num_layers, self.num_blocks, *conv_shape),
            dtype=cfg.conv_state_dtype,
            device=device,
        )
        self.temporal_state = torch.zeros(
            (cfg.num_layers, self.num_blocks, *temp_shape),
            dtype=cfg.dtype,
            device=device,
        )

        # Block 0 reserved as the CUDA-graph dummy.
        self.working_alloc = IDAllocator(1, self.num_blocks - 1)

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

    # --- block pool -----------------------------------------------------
    #
    # A "block" holds one full per-layer GDN recurrent state. Sequences borrow
    # one block for their rolling state; MTP verify borrows extra transient
    # blocks for per-token checkpoints. ``allocate_working`` / ``free_working``
    # are kept as aliases so pre-existing call sites keep working.

    def allocate_block(self) -> int:
        return self.working_alloc.allocate()

    def free_block(self, block: int) -> None:
        if block is None or block == self.dummy_working_slot:
            return
        # Zero before returning so the next borrower starts from h_0 = 0
        # without needing an explicit "reset state" pass through every layer.
        # Stacked layout -> one ``zero_`` per state covers all layers.
        self.conv_state[:, block].zero_()
        self.temporal_state[:, block].zero_()
        self.working_alloc.free(block)

    def num_free_blocks(self) -> int:
        return self.working_alloc.get_num_free_ids()

    def allocate_block_table(self, n: int) -> Optional[list]:
        """Borrow ``n`` blocks for a sequence's SSM state block table.

        Speculative decode gives each sequence a fixed ``1+k`` block table:
        column 0 holds the rolling/committed state and columns 1..k hold verify
        checkpoints. Returns a list of ``n`` block ids, or ``None`` if
        the pool cannot satisfy the whole request (caller must not partially
        allocate -- the scheduler gates admission on ``num_free_blocks``).
        """
        if self.working_alloc.get_num_free_ids() < n:
            return None
        return [self.working_alloc.allocate() for _ in range(n)]

    def free_block_table(self, blocks) -> None:
        """Return a sequence's whole SSM block table to the pool (zeroing each)."""
        if not blocks:
            return
        for blk in blocks:
            self.free_block(blk)

    # Back-compat aliases (one working slot == one borrowed block).
    def allocate_working(self) -> int:
        return self.allocate_block()

    def free_working(self, slot: int) -> None:
        self.free_block(slot)

    def num_free_working(self) -> int:
        return self.num_free_blocks()

    # --- prefix-cache cached-state blocks ------------------------------
    #
    # A prefix-cached prefix keeps its recurrent state in a block borrowed from
    # the SAME main pool (no separate snapshot pool). ``allocate_snapshot`` /
    # ``free_snapshot`` are thin aliases over the block allocator so the
    # PrefixSegment lifecycle code (which reserves a cached-state block per
    # cacheable page and frees it on re-mint) reads naturally. Returns None when
    # the pool is exhausted -> the caller degrades to "KV-cached but no SSM".

    def allocate_snapshot(self) -> Optional[int]:
        if self.working_alloc.get_num_free_ids() == 0:
            return None
        return self.working_alloc.allocate()

    def free_snapshot(self, slot: int) -> None:
        if slot is None or slot == self.dummy_working_slot:
            return
        self.free_block(slot)

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
        """Copy a full multi-layer recurrent state between two blocks of the
        shared pool. ``src_kind``/``dst_kind`` ("working"/"snapshot") are only
        semantic labels for the copy direction; both index the same pool.

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
        # Both "working" and "snapshot" now live in the SAME block pool -- the
        # kind is just a semantic label for the copy direction (capture vs
        # restore). Cached-state ("snapshot") blocks and live rolling
        # ("working") blocks are distinct block ids in ``conv_state`` /
        # ``temporal_state``; the copy is always between different block ids.
        if kind in ("working", "snapshot"):
            return self.conv_state, self.temporal_state
        raise ValueError(f"unknown ssm pool kind: {kind!r}")

    # --- MTP verify checkpoint commit ----------------------------------
    #
    # An MTP verify forward runs the GDN recurrent kernel over [x1, d1..dk] and
    # checkpoints the state after each token into a set of transient blocks
    # borrowed from this same pool (one block per verify step, per sequence).
    # The verify forward does NOT write the sequence's rolling block (it passes
    # ``disable_state_update``). After the accept step knows each seq committed
    # ``1+na`` tokens, we copy the step-``na`` checkpoint block's contents into
    # the sequence's rolling block -- the exact post-commit recurrent state,
    # with no rollback and no recompute forward. The transient blocks are then
    # freed back to the shared pool. One rolling block remains the source of
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
        self.conv_state.index_copy_(
            1, dst, self.conv_state.index_select(1, src)
        )
        self.temporal_state.index_copy_(
            1, dst, self.temporal_state.index_select(1, src)
        )


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
        device = torch.device("cuda", torch.cuda.current_device())
        # Packed FP8 layout size: kv_lora_rank(=kv_head_dim - qk_rope) FP8 bytes
        # + (kv_lora_rank/128) fp32 scale bytes + qk_rope_head_dim bf16 bytes.
        # For MLA, kv_head_dim = kv_lora_rank + qk_rope_head_dim.

        if not use_mla:
            # We don't need zero initialization here
            self.k_cache = [
                torch.ones(
                    (num_pages, page_size, kv_head_num, kv_head_dim), device=device
                )
                for _ in range(num_layers)
            ]
            self.v_cache = [
                torch.ones(
                    (num_pages, page_size, kv_head_num, kv_head_dim), device=device
                )
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
                    device=device,
                )
                for _ in range(num_layers)
            ]
        else:
            self.kv_cache = [
                torch.ones((num_pages, page_size, kv_head_dim), device=device)
                for _ in range(num_layers)
            ]
        # DeepSeek Sparse Attention: parallel indexer key cache (bf16, one
        # single-head index_head_dim vector per token per layer). Only
        # allocated when index_head_dim > 0.
        if index_head_dim > 0:
            self.index_k_cache = [
                torch.zeros(
                    (num_pages, page_size, index_head_dim), device=device
                )
                for _ in range(num_layers)
            ]
            # DSA FP8 indexer scoring: a parallel paged FP8 index-K cache in the
            # 132-byte block-contiguous layout the deep_gemm paged-MQA-logits
            # kernel reads (per page: [page_size*index_head_dim fp8][page_size*
            # (index_head_dim/128)*4 scale]). ``index_head_dim`` (128) => 128 fp8
            # + 4 scale = 132 bytes/token. Only allocated when FP8 scoring is on.
            if _DSA_FP8_SCORE:
                assert index_head_dim % _DSA_FP8_TILE == 0
                n_sf = index_head_dim // _DSA_FP8_TILE  # scales per token (=1)
                self.index_fp8_bytes = index_head_dim + n_sf * 4  # 132
                self.index_k_fp8_cache = [
                    torch.zeros(
                        (num_pages, page_size * self.index_fp8_bytes),
                        dtype=torch.uint8,
                        device=device,
                    )
                    for _ in range(num_layers)
                ]
            else:
                self.index_k_fp8_cache = None
        else:
            self.index_k_cache = None
            self.index_k_fp8_cache = None
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
            ssm_snapshot_stride_tokens: token granularity of recurrent-state
                prefix caching, rounded down to whole KV pages (see
                ``PrefixSegment.ssm_snapshot_stride``). Smaller = finer restore
                points but more snapshot blocks per prompt; the pool is shared
                with the working state, so too small starves admission.
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
        # Draft-chain length (mtp_k); a running seq borrows up to this many
        # transient checkpoint blocks during an MTP verify step (0 = MTP off).
        self.mtp_k = mtp_k
        # Recurrent-state prefix-cache granularity, in TOKENS. Converted to
        # whole pages and installed on the segment by
        # ``PrefixMemoryManager.init``; ignored without prefix caching.
        self.ssm_snapshot_stride_tokens = ssm_snapshot_stride_tokens
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
        self.k_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
        self.v_scale = self.k_scale

    def _init_ssm_segment_if_needed(self) -> None:
        """Allocate the SSM block pool + snapshot pool when the model needs them.

        For hybrid GDN/Mamba models each pool block holds the full per-layer
        recurrent state. We size one shared block pool from the memory budget,
        decoupled from ``maxd``. Every consumer borrows from it:

        * a running seq borrows 1 rolling block; an MTP verify step borrows a
          few transient checkpoint blocks; a prefix-cached prefix keeps its
          state in a ref-counted block borrowed from this same pool.
        * ``num_ssm_blocks = min(budget/per_block, working_cap + cache_cap)``
          where ``working_cap = maxd * (1 + mtp_k)`` (max blocks live seqs can
          borrow at once) and ``cache_cap`` (== requested prefix-cache blocks)
          is best-effort headroom for cached-prefix reuse. Floored at ``maxd``
          (>=1 rolling block per running seq) or we raise a clear error.
        """
        if self.ssm_cache_config is None:
            return
        cfg = self.ssm_cache_config
        per_block = cfg.per_slot_bytes()

        free_mem, _ = torch.cuda.mem_get_info()
        budget = int(free_mem * self.gpu_memory_util * self.ssm_pool_budget_frac)

        maxd = self.max_working_ssm_slots
        # Blocks a single running seq may hold at once: 1 rolling + mtp_k
        # transient checkpoint blocks during an MTP verify (0 extra when MTP off).
        per_seq_blocks = 1 + self.mtp_k
        working_cap = maxd * per_seq_blocks
        # Best-effort headroom for prefix-cached-prefix state blocks (borrowed
        # from the same pool, ref-counted alongside their KV page).
        cache_cap = max(self.max_snapshot_ssm_slots, 0)
        block_cap = working_cap + cache_cap
        # Budget-derived block count (like KV pages = free_mem*util / kv_page).
        affordable_blocks = int(budget // per_block)
        num_ssm_blocks = min(block_cap, affordable_blocks)
        # Floor: at least one rolling block per concurrently-running seq, else
        # the scheduler could admit a seq with no state block. If even that
        # doesn't fit, fail early with an actionable message.
        if num_ssm_blocks < maxd:
            need = maxd * per_block
            raise RuntimeError(
                f"SSM block pool needs >= {need / (1 << 30):.1f} GB for {maxd} "
                f"concurrent sequences ({maxd} blocks x "
                f"{per_block / (1 << 20):.1f} MB) but only "
                f"{budget / (1 << 30):.1f} GB SSM budget is available. Lower "
                f"--maxd (currently {maxd}) or raise --tp to shrink per-rank state."
            )

        # Keep every TP rank's pool layout identical (state is sharded, not
        # replicated, but the block *count* must match across ranks); free
        # memory can differ slightly per rank, so agree on the minimum.
        if dist.is_initialized():
            gathered = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered, num_ssm_blocks)
            num_ssm_blocks = min(gathered)

        # Prefix-cache headroom actually available after the mandatory working
        # capacity (informational; the pool is shared so cache borrows compete
        # with live seqs at runtime and degrade gracefully when tight).
        cache_headroom = max(0, num_ssm_blocks - working_cap)
        if cache_headroom < cache_cap:
            logger.warning(
                "SSM prefix-cache headroom %d -> %d blocks to fit the memory "
                "budget (%.1f GB free, %.0f%% util, %.0f%% SSM share); prefix-cache "
                "state reuse is %s. Lower --maxd or raise --tp for the full pool.",
                cache_cap,
                cache_headroom,
                free_mem / (1 << 30),
                self.gpu_memory_util * 100,
                self.ssm_pool_budget_frac * 100,
                "reduced" if cache_headroom > 0 else "disabled",
            )

        self.ssm_segment = SSMSegment(
            cfg,
            num_blocks=num_ssm_blocks,
        )
        total = per_block * self.ssm_segment.num_blocks
        logger.info(
            "SSM cache: %d state blocks (max decode concurrency ~%d, prefix-cache "
            "headroom ~%d blocks), %.2f KB/block, %.2f GB total (linear-attn "
            "layers: %d, temporal dtype: %s, conv dtype: %s)",
            self.ssm_segment.num_blocks,
            num_ssm_blocks // per_seq_blocks if per_seq_blocks else num_ssm_blocks,
            cache_headroom,
            per_block / 1024,
            total / (1 << 30),
            cfg.num_layers,
            cfg.dtype,
            cfg.conv_state_dtype,
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
        reads. Only valid when ``segment.index_k_fp8_cache`` is allocated
        (``GLLM_DSA_FP8_SCORE=1``). ``use_ue8m0`` rounds the per-token scale to a
        power of two (set by the caller from the checkpoint's ``scale_fmt``).
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
            num_page = (
                (int(seq_len) + self.page_size - 1) // self.page_size
                - len(seq.page_table)
            )
            for _ in range(num_page):
                seq.page_table.append(self.segment.allocate())

    def register_decode_boundary(self, seq: GenerationSequence, pos: int) -> None:
        """No-op without a prefix cache; overridden by ``PrefixMemoryManager``."""
        return

    def free(self, seq: GenerationSequence):
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

    def allocate_ssm_slot(self, seq: GenerationSequence) -> None:
        if self.ssm_segment is None:
            return
        if self.mtp_k > 0:
            # MTP on: give the sequence a fixed 1+k block table (column 0 is
            # rolling state; the remaining columns are verify checkpoints).
            if seq.ssm_block_table is not None:
                return
            bt = self.ssm_segment.allocate_block_table(1 + self.mtp_k)
            if bt is None:
                return  # pool exhausted; scheduler gates admission on this
            seq.ssm_block_table = bt
            # Mirror column 0 into the scalar slot so any legacy single-slot
            # reader (e.g. prefix-cache snapshot restore) still works.
            seq.ssm_state_slot = bt[0]
            seq.ssm_num_accepted = 1
        else:
            if seq.ssm_state_slot is not None:
                return
            seq.ssm_state_slot = self.ssm_segment.allocate_working()

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
        page_tokens = tuple(src[i * page_size:(i + 1) * page_size])
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
        # Watermark for lazy cached-state reservation: the cache may only use
        # blocks *beyond* the full concurrency budget (``maxd * (1 + mtp_k)``),
        # which is exactly how the pool was sized (``block_cap = working_cap +
        # cache_cap`` in ``_init_ssm_segment_if_needed``). Without this the
        # cache eats the working budget, the scheduler's admission gate (needs
        # ``1 + mtp_k`` free blocks per new sequence) starves, and the running
        # batch collapses to one sequence with a full wait queue.
        self.segment.ssm_reserve_floor = (
            self.max_working_ssm_slots * (1 + self.mtp_k)
            if self.ssm_segment is not None
            else 0
        )
        # Recurrent-state caching granularity for this run. Rounded DOWN to
        # whole pages (only page boundaries can carry a snapshot) with a floor
        # of one page; a request below ``page_size`` therefore degrades to
        # per-page snapshots, which is the configuration that drained the block
        # pool (see ``PrefixSegment.ssm_snapshot_stride``), so say so out loud.
        stride_tokens = int(self.ssm_snapshot_stride_tokens)
        if stride_tokens < self.page_size:
            logger.warning(
                "ssm_snapshot_stride_tokens=%d is below page_size=%d; clamping to "
                "one page. Per-page recurrent-state caching reserves a state block "
                "per %d tokens of prompt and can starve sequence admission.",
                stride_tokens, self.page_size, self.page_size,
            )
        self.segment.ssm_snapshot_stride = max(
            1, stride_tokens // self.page_size
        )
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
        # keyed by seq_id -> snapshot-pool slot. Each PP follower owns a
        # *different* slice of the GDN layers on its own GPU, so the restore
        # (snapshot->working ``copy_state``) the driver runs on rank-0's pools
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
            seq_cacheable = cacheable and not getattr(
                seq, "_mtp_async_pending", False
            )
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

    SSM extension: for every cached page we keep an optional snapshot slot
    in the partner :class:`SSMSegment`. When a sequence allocates a page
    in :meth:`PrefixMemoryManager.pre_allocate_page`, a snapshot slot is
    reserved alongside; the GDN layer fills it after the page boundary is
    crossed during prefill. On a cache hit the snapshot is copied back into
    the requesting sequence's working slot.
    """

    # Set by :class:`PrefixMemoryManager.init`.
    ssm_segment: Optional[SSMSegment] = None

    # Recurrent-state caching granularity, in PAGES: only page boundaries at a
    # multiple of this stride can hold a cached state. Installed by
    # :meth:`PrefixMemoryManager.init` from ``--ssm-snapshot-stride-tokens``;
    # the value here is only a floor for a segment built outside that path.
    #
    # This used to be every cacheable page: ``allocate`` reserved one state
    # block per 16-token page, so a single 2.5k-token prompt reserved ~156
    # blocks out of a ~1800-block pool. Ten such requests drained it, and
    # since new admissions need ``1 + mtp_k`` blocks from the same pool the
    # scheduler then stalled with #run=1 and 120+ queued (observed on
    # MMLU-Pro 5-shot). Worse, those reservations were nearly all dead
    # weight: a state is only ever *written* at a chunk end, so the interior
    # boundaries kept a reserved-but-zeroed block forever and every hit was
    # rejected on the SSM half -> 0% cache hit rate. Coarse + lazy (see
    # ``reserve_ssm_snapshot``) fixes both: ~10 blocks per prompt, and only
    # for boundaries a chunk actually lands on. Restoring from a coarse
    # boundary means recomputing at most ``stride`` pages of tail, trading a
    # bounded amount of recompute for a much smaller snapshot pool.
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
        self.page2ssm_snapshot: List[Optional[int]] = [None for _ in range(self.num_pages)]
        # Whether the snapshot slot actually holds a *written* recurrent state.
        # A slot only ever gets written for the boundary a prefill chunk *ends*
        # on (see ``InputData._cal_ssm_metadata``); this flag separates
        # "reserved" from "filled" so the restore path never grafts a zeroed
        # state (== h_0) onto a non-empty prefix.
        self.page2ssm_snapshot_valid: List[bool] = [False for _ in range(self.num_pages)]

    # --- public API ---------------------------------------------------------

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

    def allocate(self, seq: Optional[GenerationSequence] = None, n_tokens: Optional[int] = None):
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
            # the same lifetime — otherwise a serial re-use of a cached
            # prompt would always lose the snapshot half of the hit and
            # ``_rollback_to_last_ssm_hit`` would drop the KV half too.
            self.id_allocator.free(page_num)

    def reserve_ssm_snapshot(self, page_num: int, n_tokens: int) -> Optional[int]:
        """Lazily reserve the cached-state block for ``page_num``, or ``None``.

        Called from the *write* path (``InputData._cal_ssm_metadata``) for the
        boundary a prefill chunk just landed on, so a block is only ever taken
        for a boundary that will actually hold a state. Returns ``None`` when:

        * the boundary is not on the coarse ``ssm_snapshot_stride`` grid,
        * the page is not cacheable (nothing could ever hit it), or
        * the pool is at its watermark -- cached states must never eat the
          blocks that live sequences need for their rolling state, otherwise
          the scheduler's admission gate (which needs ``1 + mtp_k`` free blocks
          per new sequence) starves and the batch collapses to one sequence.
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
        if self.ssm_segment.num_free_blocks() <= self.ssm_reserve_floor:
            return None
        slot = self.ssm_segment.allocate_snapshot()
        if slot is None:
            return None
        self.page2ssm_snapshot[page_num] = slot
        self.page2ssm_snapshot_valid[page_num] = False
        return slot

    def _release_snapshot_for(self, page_num: int) -> None:
        if self.ssm_segment is None:
            return
        snap_slot = self.page2ssm_snapshot[page_num]
        if snap_slot is not None:
            self.ssm_segment.free_snapshot(snap_slot)
            self.page2ssm_snapshot[page_num] = None
        self.page2ssm_snapshot_valid[page_num] = False
