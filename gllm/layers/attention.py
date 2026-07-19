from typing import Optional

import torch
from logger import logger

from gllm import _custom_ops as ops
from gllm.input_data import InputData, MLACommonMetadata, MLACommonPrefillMetadata
from gllm.layers.linear import ColumnParallelLinear, LinearBase
from gllm.layers.ops.merge_attn_states import merge_attn_states
from gllm.layers.ops.triton_decode_attention import decode_attention_fwd
from sgl_kernel.flash_attn import flash_attn_with_kvcache, flash_attn_varlen_func

# FA3 MLA decode (sgl_kernel flash_attn with qv=) — same path as SGLang ``fa3``.
try:
    from sgl_kernel.flash_attn import is_fa3_supported

    _FA3_AVAILABLE = bool(is_fa3_supported())
    _FA3_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover - depends on hardware / build
    is_fa3_supported = None  # type: ignore[misc, assignment]
    _FA3_AVAILABLE = False
    _FA3_IMPORT_ERROR = _e

# FlashMLA (DeepSeek SM90 MLA decode kernel) is optional: the extension only
# loads on Hopper with a recent CUDA driver. Import lazily so the default
# Triton decode backend keeps working when it is unavailable.
try:
    from sgl_kernel.flash_mla import flash_mla_with_kvcache, get_mla_metadata
    from sgl_kernel.flash_mla import flash_mla_sparse_fwd

    _FLASHMLA_AVAILABLE = True
    _FLASHMLA_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover - depends on hardware / build
    flash_mla_with_kvcache = None
    get_mla_metadata = None
    flash_mla_sparse_fwd = None
    _FLASHMLA_AVAILABLE = False
    _FLASHMLA_IMPORT_ERROR = _e

# FlashMLA only supports a KV page/block size of 64.
_FLASHMLA_PAGE_SIZE = 64

# Log the resolved MLA decode backend once per worker at model load.
_mla_decode_backend_startup_logged = False


def _flash_attn_paged_varlen(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    cache_seqlens: torch.Tensor,
    page_table: torch.Tensor,
    softmax_scale: float,
    causal: bool = True,
    return_softmax_lse: bool = False,
):
    """
    Flash attention with paged KV cache and variable-length sequences.

    Uses flash_attn_with_kvcache which supports both cu_seqlens_q (varlen) and
    page_table (paged KV cache) simultaneously.

    Args:
        q: [total_tokens, num_heads, head_dim]
        k_cache: [num_blocks, block_size, num_kv_heads, head_dim]
        v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
        cu_seqlens_q: [batch_size + 1] cumulative query sequence lengths
        max_seqlen_q: maximum query length
        cache_seqlens: [batch_size] actual sequence lengths in KV cache
        page_table: [batch_size, max_blocks_per_seq] block indices
        softmax_scale: attention scaling factor
        causal: whether to use causal masking
        return_softmax_lse: whether to return logsumexp
    """
    out = flash_attn_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        softmax_scale=softmax_scale,
        causal=causal,
        return_softmax_lse=return_softmax_lse,
    )
    return out


class FlashAttention:

    def __init__(
        self,
        layer_id: int,
        scale: float,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
    ):
        self.scale = scale
        self.layer_id = layer_id
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, input_data: InputData
    ):
        # profile run: the KV segment is only built in ``MemoryManager.init``
        # (after the profiling forward). Guard on ``is None`` rather than
        # ``hasattr`` so the check still fires now that ``segment`` is declared
        # as a ``None`` attribute in ``MemoryManager.__init__``.
        if getattr(input_data.memory_manager, "segment", None) is None:
            return q

        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_key_value_heads, self.head_dim)
        v = v.view(-1, self.num_key_value_heads, self.head_dim)

        input_data.memory_manager.batch_store(
            self.layer_id, k, v, input_data.get_slot_mapping()
        )

        k_cache = input_data.memory_manager.segment.k_cache[self.layer_id]
        v_cache = input_data.memory_manager.segment.v_cache[self.layer_id]

        out = _flash_attn_paged_varlen(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            cu_seqlens_q=input_data.get_query_start_loc(),
            max_seqlen_q=input_data.max_query_len,
            cache_seqlens=input_data.get_seq_lens(),
            page_table=input_data.get_block_table(),
            softmax_scale=self.scale,
            causal=True,
        )
        return out.view(-1, out.shape[-2] * out.shape[-1])


class MLAAttention:
    def __init__(
        self,
        layer_id: int,
        scale: float,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        # MLA Specific Arguments
        q_lora_rank: Optional[int],
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        qk_head_dim: int,
        v_head_dim: int,
        kv_b_proj: ColumnParallelLinear,
        # Decode backend is selected by the upper layer (ModelRunner) and
        # threaded down through the model config; see ``_resolve_decode_backend``.
        decode_backend: str = "triton",
        page_size: Optional[int] = None,
    ):
        self.scale = scale
        self.layer_id = layer_id
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.kv_b_proj = kv_b_proj

        self._pad_v = True

        self._k_scale = torch.tensor(1.0, dtype=torch.float32)

        self.W_UV = None
        self.W_UK_T = None

        self.kv_cache_dtype = "auto"

        self.decode_backend = self._resolve_decode_backend(decode_backend, page_size)
        self._log_startup_decode_backend(
            decode_backend, page_size, self.decode_backend
        )

    @staticmethod
    def _log_startup_decode_backend(
        requested: str, page_size: Optional[int], resolved: str
    ) -> None:
        global _mla_decode_backend_startup_logged
        if _mla_decode_backend_startup_logged:
            return
        _mla_decode_backend_startup_logged = True
        req = (requested or "fa3").lower()
        ps = page_size if page_size is not None else "default"
        if resolved == req:
            logger.info(
                "MLA decode attention backend: %s (page_size=%s)",
                resolved,
                ps,
            )
        else:
            logger.info(
                "MLA decode attention backend: %s (requested %r, page_size=%s)",
                resolved,
                req,
                ps,
            )

    @staticmethod
    def _flashmla_available(page_size: Optional[int]) -> Optional[str]:
        if not _FLASHMLA_AVAILABLE:
            return f"sgl_kernel.flash_mla unavailable ({_FLASHMLA_IMPORT_ERROR})"
        if page_size is not None and page_size != _FLASHMLA_PAGE_SIZE:
            return f"page_size={page_size} != required {_FLASHMLA_PAGE_SIZE}"
        return None

    def _resolve_decode_backend(
        self, requested: str, page_size: Optional[int]
    ) -> str:
        """Pick the decode backend with hardware-aware fallbacks.

        The choice is made by the upper layer (default ``fa3``), but the final
        availability check happens here in the worker where ``sgl_kernel`` is
        loaded. ``triton`` is always valid; ``fa3`` needs Hopper FA3;
        ``flashmla`` needs the FlashMLA extension and ``page_size == 64``.
        """
        requested = (requested or "fa3").lower()
        if requested not in ("triton", "flashmla", "fa3"):
            raise ValueError(
                "mla_decode_backend must be 'fa3', 'flashmla', or 'triton', "
                f"got {requested!r}."
            )
        if requested == "triton":
            return "triton"

        if requested == "fa3":
            if _FA3_AVAILABLE:
                return "fa3"
            reason = f"sgl_kernel FA3 unavailable ({_FA3_IMPORT_ERROR})"
            flashmla_reason = self._flashmla_available(page_size)
            if flashmla_reason is None:
                logger.warning(
                    "MLA decode backend 'fa3' is unavailable (%s); "
                    "falling back to 'flashmla'.",
                    reason,
                )
                return "flashmla"
            logger.warning(
                "MLA decode backend 'fa3' is unavailable (%s); "
                "falling back to 'triton'.",
                reason,
            )
            return "triton"

        flashmla_reason = self._flashmla_available(page_size)
        if flashmla_reason is not None:
            logger.warning(
                "MLA decode backend 'flashmla' is unavailable (%s); "
                "falling back to 'triton'.",
                flashmla_reason,
            )
            return "triton"
        return "flashmla"

    def process_weights(self):
        def get_and_maybe_dequant_weights(layer: LinearBase):
            eye = torch.eye(layer.input_size_per_partition)
            dequant_weights = layer.quant_method(eye, layer.weight, bias=None)
            del eye
            return dequant_weights.T

        kv_b_proj_weight = get_and_maybe_dequant_weights(self.kv_b_proj).T
        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )

        W_UK, W_UV = kv_b_proj_weight.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        # Convert from (L, N, V) to (N, L, V)
        self.W_UV = W_UV.transpose(0, 1)
        # Convert from (L, N, P) to (N, P, L)
        self.W_UK_T = W_UK.permute(1, 2, 0)

    def _flash_attn_varlen_diff_headdims(
        self, q, k, v, softmax_scale, return_softmax_lse, **kwargs
    ):
        maybe_padded_v = v
        if self._pad_v:
            maybe_padded_v = torch.nn.functional.pad(
                v, [0, q.shape[-1] - v.shape[-1]], value=0
            )

        attn_out = flash_attn_varlen_func(
            q=q,
            k=k,
            v=maybe_padded_v,
            return_softmax_lse=return_softmax_lse,
            softmax_scale=softmax_scale,
            **kwargs,
        )

        # Unpack the output if there are multiple results
        # sgl_kernel returns (output, softmax_lse) if return_softmax_lse=True
        rest = None
        if isinstance(attn_out, tuple):
            attn_out, *rest = attn_out

        # Remain consistent with old interface where there
        # is only one output tensor if `return_softmax_lse` is False.
        if return_softmax_lse:
            assert rest is not None
            # sgl_kernel.flash_attn_varlen_func returns softmax_lse with shape
            # (num_heads, total_seq_len), but sgl_kernel.merge_state_v2
            # (used by merge_attn_states) expects (total_seq_len, num_heads).
            softmax_lse = rest[0].transpose(0, 1).contiguous()
            return attn_out, softmax_lse
        return attn_out

    def _run_prefill_context_chunk(
        self, prefill: MLACommonPrefillMetadata, chunk_idx: int, q, k, v
    ):
        assert prefill.chunked_context is not None
        return self._flash_attn_varlen_diff_headdims(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=prefill.query_start_loc,
            cu_seqlens_k=prefill.chunked_context.cu_seq_lens[chunk_idx],
            max_seqlen_q=prefill.max_query_len,
            max_seqlen_k=prefill.chunked_context.max_seq_lens[chunk_idx],
            softmax_scale=self.scale,
            causal=False,  # Context is unmasked
            return_softmax_lse=True,
        )

    def _v_up_proj(self, x: torch.Tensor, out: torch.Tensor):
        # Convert from (B, N, L) to (N, B, L)
        x = x.view(-1, self.num_heads, self.kv_lora_rank).transpose(0, 1)
        # Convert from (B, N * V) to (N, B, V)
        out = out.view(-1, self.num_heads, self.v_head_dim).transpose(0, 1)

        # Multiply (N, B, L) x (N, L, V) -> (N, B, V)
        if self.W_UV is None:
            self.process_weights()
        torch.bmm(x, self.W_UV, out=out)  # Reuse "out" to make it "hot"

        # Convert from (N, B, V) to (B, N * V)
        out_new = out.transpose(0, 1).reshape(-1, self.num_heads * self.v_head_dim)

        # Adjust output buffer shape back to the original (B, N * V)
        N, B, V = out.shape
        out.resize_((B, N * V))
        out.copy_(out_new)  # Copy result

    def _compute_prefill_context(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        k_scale: torch.Tensor,
    ):
        assert attn_metadata.prefill is not None
        prefill_metadata = attn_metadata.prefill
        assert prefill_metadata.chunked_context is not None

        output = None
        iters = len(prefill_metadata.chunked_context.seq_tot)
        workspace = prefill_metadata.chunked_context.workspace

        for i in range(iters):
            toks = prefill_metadata.chunked_context.seq_tot[i]

            if kv_c_and_k_pe_cache.dtype == torch.float8_e4m3fn:
                # DeepSeek Sparse Attention: the paged latent cache is stored in
                # FlashMLA's 656-byte FP8-packed layout, so read it back through
                # the dequantizing gather (per-tile fp8 -> bf16 + rope verbatim).
                # The scales are embedded in the cache, so no external k_scale.
                ops.gather_and_dequant_mla_fp8(
                    src_cache=kv_c_and_k_pe_cache,
                    dst=workspace,
                    block_table=prefill_metadata.block_table,
                    cu_seq_lens=prefill_metadata.chunked_context.cu_seq_lens[i],
                    batch_size=attn_metadata.num_prefills,
                    kv_lora_rank=self.kv_lora_rank,
                    qk_rope_head_dim=self.qk_rope_head_dim,
                    seq_starts=prefill_metadata.chunked_context.starts[i],
                )
            else:
                ops.gather_and_maybe_dequant_cache(
                    src_cache=kv_c_and_k_pe_cache,
                    dst=workspace,
                    block_table=prefill_metadata.block_table,
                    cu_seq_lens=prefill_metadata.chunked_context.cu_seq_lens[i],
                    batch_size=attn_metadata.num_prefills,
                    kv_cache_dtype=self.kv_cache_dtype,
                    scale=k_scale,
                    seq_starts=prefill_metadata.chunked_context.starts[i],
                )

            kv_c_normed = workspace[:toks][..., : self.kv_lora_rank]
            k_pe = workspace[:toks][..., self.kv_lora_rank :].unsqueeze(1)

            kv_nope = self.kv_b_proj(kv_c_normed).view(
                -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

            k = torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))), dim=-1)

            attn_output, attn_softmax_lse = self._run_prefill_context_chunk(
                prefill=prefill_metadata,
                chunk_idx=i,
                q=q,
                k=k,
                v=v,
            )

            if output is None:
                output = attn_output
                output_lse = attn_softmax_lse
            else:
                output_tmp = torch.empty_like(output)
                output_lse_tmp = torch.empty_like(output_lse)
                merge_attn_states(
                    output=output_tmp,
                    output_lse=output_lse_tmp,
                    prefix_output=output,
                    prefix_lse=output_lse,
                    suffix_output=attn_output,
                    suffix_lse=attn_softmax_lse,
                )
                output = output_tmp
                output_lse = output_lse_tmp

        return output, output_lse

    def _run_prefill_new_tokens(
        self, prefill: MLACommonPrefillMetadata, q, k, v, return_softmax_lse
    ):
        return self._flash_attn_varlen_diff_headdims(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=prefill.query_start_loc,
            cu_seqlens_k=prefill.query_start_loc,
            max_seqlen_q=prefill.max_query_len,
            max_seqlen_k=prefill.max_query_len,
            softmax_scale=self.scale,
            causal=True,
            return_softmax_lse=return_softmax_lse,
        )

    def _forward_prefill_sparse(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        topk_indices: torch.Tensor,
        k_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Sparse MLA prefill via ``flash_mla_sparse_fwd`` (DeepSeek-V3.2 DSA).

        ``topk_indices`` are per-prefill-query absolute physical KV slots
        (``[num_prefill_tokens, index_topk]`` int32, -1 padded) already made
        causal by the indexer. Mirrors the absorbed decode path: project the
        query nope part into the latent space with ``W_UK_T``, feed the sparse
        kernel the latent KV cache indexed by the top-k slots, then ``W_UV`` the
        result back to the value space.

        ``flash_mla_sparse_fwd(q, kv, indices, sm_scale, d_v)`` expects a
        contiguous ``kv`` of shape ``[num_slots, 1, kv_lora + qk_rope]`` and
        ``indices`` of shape ``[num_q, 1, topk]`` indexing ``kv``'s first dim; -1
        entries are skipped. Physical slots index the flat cache directly, so the
        flat latent cache IS that ``kv`` (bf16); the FP8-packed cache is
        dequantized to the same layout first.
        """
        if flash_mla_sparse_fwd is None:
            raise RuntimeError(
                "flash_mla_sparse_fwd unavailable; cannot run DSA sparse prefill."
            )
        if self.W_UK_T is None or self.W_UV is None:
            self.process_weights()

        num_q = q.shape[0]
        # Absorb q_nope -> latent: (B,N,P) -> (N,B,P) x (N,P,L) -> (B,N,L).
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_nope = q_nope.transpose(0, 1)  # (N, B, P)
        N, B, P = q_nope.shape
        _, _, L = self.W_UK_T.shape
        ql_nope = q_nope.new_empty((N, B, L))
        torch.bmm(q_nope, self.W_UK_T, out=ql_nope)  # (N, B, L)
        ql_nope = ql_nope.transpose(0, 1)  # (B, N, L)
        # Sparse-kernel query: latent nope ++ rope -> (num_q, num_heads, L + R).
        q_sparse = torch.cat([ql_nope, q_pe], dim=-1).contiguous()

        # Build the contiguous latent KV the kernel indexes into.
        dim = self.kv_lora_rank + self.qk_rope_head_dim
        if kv_c_and_k_pe_cache.dtype == torch.float8_e4m3fn:
            # FP8-packed (656B) cache: sparse prefill on the FP8 layout needs a
            # dequantizing gather to the bf16 latent layout the kernel indexes.
            # Not wired yet -- validate the bf16 cache path first (the default
            # ``mla_cache_dtype``); FP8 sparse prefill is a follow-up.
            raise NotImplementedError(
                "DSA sparse prefill on the FP8-packed MLA cache is not yet "
                "implemented; run with the default bf16 MLA cache "
                "(--mla-cache-dtype bf16)."
            )
        kv = kv_c_and_k_pe_cache.reshape(-1, 1, dim)

        indices = topk_indices.to(torch.int32)
        if indices.dim() == 2:
            indices = indices.unsqueeze(1)  # (num_q, 1, topk)

        # FlashMLA's sparse prefill kernel requires the query head count to be a
        # multiple of 64 (SM90/Hopper) or 128 (SM100+/Blackwell). Under TP the
        # per-rank head count is smaller (e.g. 128/8=16), so zero-pad the head
        # axis up to the required multiple and trim the output back (matches
        # SGLang's NSA backend).
        required = 128 if torch.cuda.get_device_capability()[0] >= 10 else 64
        num_heads = q_sparse.shape[1]
        pad = num_heads % required != 0
        if pad:
            assert required % num_heads == 0, (
                f"num_heads {num_heads} cannot be zero-padded to {required}; "
                "TP size may be too large for this model."
            )
            q_in = q_sparse.new_zeros((num_q, required, q_sparse.shape[-1]))
            q_in[:, :num_heads, :] = q_sparse
        else:
            q_in = q_sparse

        o, _, _ = flash_mla_sparse_fwd(
            q=q_in,
            kv=kv,
            indices=indices,
            sm_scale=self.scale,
            d_v=self.kv_lora_rank,
        )
        if pad:
            o = o[:, :num_heads, :]
        # o: (num_q, num_heads, kv_lora_rank) -> v_up -> (num_q, num_heads*v_head_dim).
        out = q.new_empty((num_q, self.num_heads * self.v_head_dim))
        self._v_up_proj(o.reshape(num_q, self.num_heads, self.kv_lora_rank), out=out)
        return out

    def _forward_prefill(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        k_scale: torch.Tensor,
        topk_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert attn_metadata.prefill is not None

        # DeepSeek Sparse Attention (V3.2): when the indexer supplies per-prefill
        # -query top-k physical KV slots, run the sparse prefill. The top-k set
        # already spans each query's full causal history (cached prefix + earlier
        # new tokens, all written to the KV cache before this call), so the sparse
        # path replaces BOTH the dense new-token (suffix) and context (prefix)
        # pieces below -- no separate merge needed.
        if topk_indices is not None:
            return self._forward_prefill_sparse(
                q, kv_c_and_k_pe_cache, attn_metadata, topk_indices, k_scale
            )

        has_context = attn_metadata.prefill.chunked_context is not None
        kv_nope = self.kv_b_proj(kv_c_normed).view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k = torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))), dim=-1)

        output = self._run_prefill_new_tokens(
            prefill=attn_metadata.prefill,
            q=q,
            k=k,
            v=v,
            return_softmax_lse=has_context,
        )

        if has_context:
            suffix_output, suffix_lse = output
            context_output, context_lse = self._compute_prefill_context(
                q, kv_c_and_k_pe_cache, attn_metadata, k_scale
            )

            output = torch.empty_like(suffix_output)
            merge_attn_states(
                output=output,
                prefix_output=context_output,
                prefix_lse=context_lse,
                suffix_output=suffix_output,
                suffix_lse=suffix_lse,
            )

        # unpad if necessary
        if self._pad_v:
            output = output[..., : v.shape[-1]]

        return output.flatten(start_dim=-2)

    def _forward_decode(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        topk_indices: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if topk_indices is not None:
            # DeepSeek Sparse Attention decode. Two viable kernels on SM90:
            #   * FP8-packed KV cache -> FlashMLA sparse ``fwd_kvcache_mla``
            #     (takes ``indices`` + ``is_fp8_kvcache=True``); paged bf16 sparse
            #     is rejected by that kernel ("Sparse BF16 MLA not supported").
            #   * bf16 KV cache -> FA3 ``flash_attn_with_kvcache`` fed the top-k
            #     physical slots as a ``page_size=1`` page table (SGLang's NSA
            #     ``fa3`` decode path). FA3 has no ``indices`` arg, but a
            #     per-query page table of length ``index_topk`` with
            #     ``cache_seqlens`` clamped to the selected count is exactly
            #     equivalent, and FA3 reads bf16 natively.
            if kv_c_and_k_pe_cache.dtype == torch.float8_e4m3fn:
                return self._forward_decode_flashmla(
                    q, kv_c_and_k_pe_cache, attn_metadata, topk_indices=topk_indices
                )
            return self._forward_decode_fa3(
                q, kv_c_and_k_pe_cache, attn_metadata, topk_indices=topk_indices
            )
        if self.decode_backend == "fa3":
            return self._forward_decode_fa3(q, kv_c_and_k_pe_cache, attn_metadata)
        if self.decode_backend == "flashmla":
            return self._forward_decode_flashmla(
                q, kv_c_and_k_pe_cache, attn_metadata
            )
        return self._forward_decode_triton(q, kv_c_and_k_pe_cache, attn_metadata)

    def _forward_decode_fa3(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        topk_indices: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Absorbed MLA decode via FA3 (SGLang ``FlashAttentionBackend`` path).

        When ``topk_indices`` is supplied (DeepSeek Sparse Attention), the
        indexer's per-query top-k **physical KV slots** are fed to FA3 as a
        ``page_size == 1`` page table with ``cache_seqlens`` clamped to the
        selected count -- exactly SGLang's NSA ``fa3`` decode path. FA3 has no
        sparse ``indices`` argument, but a length-``index_topk`` per-query page
        table of physical slots is equivalent and reads bf16 KV natively (unlike
        FlashMLA's paged sparse kernel, which rejects bf16 on SM90).
        """
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError(
                "FP8 KV cache is not yet supported by the FA3 MLA decode backend."
            )
        if type(q) is not tuple:
            raise ValueError(
                "FA3 MLA decode expects absorbed (q_nope, q_pe) tuple."
            )

        q_nope, q_rope = q
        decode_meta = attn_metadata.decode

        sparse = topk_indices is not None
        if sparse:
            # Sparse: view the cache flat over physical slots (page_size == 1)
            # so each page-table entry is one selected slot. ``topk_indices``
            # are absolute physical slots front-packed with -1 padding at the
            # tail; ``cache_seqlens`` = number of valid (>= 0) selected slots
            # per query, so FA3 never walks into the -1 padding.
            flat = kv_c_and_k_pe_cache.reshape(-1, kv_c_and_k_pe_cache.shape[-1])
            c_kv_cache = flat[..., : self.kv_lora_rank].view(-1, 1, 1, self.kv_lora_rank)
            k_rope_cache = flat[..., self.kv_lora_rank :].view(
                -1, 1, 1, flat.shape[-1] - self.kv_lora_rank
            )
            if topk_indices.dim() == 3:
                topk_indices = topk_indices.squeeze(1)
            page_table = topk_indices.to(torch.int32)
            cache_seqlens = (page_table >= 0).sum(dim=1).to(torch.int32)
        else:
            # Paged latent cache: rope on k, compressed latent on v (MQA head 1).
            c_kv_cache = kv_c_and_k_pe_cache[..., : self.kv_lora_rank].unsqueeze(2)
            k_rope_cache = kv_c_and_k_pe_cache[..., self.kv_lora_rank :].unsqueeze(2)
            page_table = decode_meta.block_table
            cache_seqlens = decode_meta.seq_lens

        cu_seqlens_q = decode_meta.query_start_loc
        max_seqlen_q = max(1, decode_meta.max_query_len)

        result = flash_attn_with_kvcache(
            q=q_rope,
            k_cache=k_rope_cache,
            v_cache=c_kv_cache,
            qv=q_nope,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            softmax_scale=self.scale,
            causal=True,
            return_softmax_lse=True,
            num_splits=0,
        )

        if isinstance(result, tuple):
            o, softmax_lse, *_ = result
        else:
            o = result
            softmax_lse = None

        if o.dim() == 2:
            o = o.view(-1, self.num_heads, self.kv_lora_rank)
        elif o.dim() == 3 and o.shape[1] != self.num_heads:
            o = o.view(-1, self.num_heads, self.kv_lora_rank)

        if softmax_lse is not None:
            if softmax_lse.dim() == 3:
                softmax_lse = softmax_lse.squeeze(-1)
            if softmax_lse.shape[0] == self.num_heads:
                softmax_lse = softmax_lse.transpose(0, 1).contiguous()
        else:
            softmax_lse = torch.zeros(
                o.shape[0],
                self.num_heads,
                dtype=o.dtype,
                device=o.device,
            )

        return o, softmax_lse

    def _forward_decode_flashmla(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        topk_indices: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError(
                "FP8 KV cache is not yet supported by the gLLM FlashMLA "
                "decode backend."
            )

        if type(q) is tuple:
            q = torch.cat(q, dim=-1)
        assert isinstance(q, torch.Tensor)

        # FlashMLA expects q as (batch, q_len, num_heads, head_dim). Decode
        # always has q_len == 1 here (one query token per sequence).
        q = q.unsqueeze(1)
        q_len = q.shape[1]

        decode_meta = attn_metadata.decode

        # FlashMLA only supports a KV page/block size of 64.
        page_size = kv_c_and_k_pe_cache.shape[1]
        assert page_size == _FLASHMLA_PAGE_SIZE, (
            "FlashMLA decode backend requires page_size == "
            f"{_FLASHMLA_PAGE_SIZE}, got {page_size}. Launch with "
            f"--page-size {_FLASHMLA_PAGE_SIZE}."
        )

        # DeepSeek Sparse Attention (V3.2): when the indexer supplies per-query
        # top-k indices (already transformed to absolute physical KV slots), run
        # the *sparse* FlashMLA decode. On SM90 (Hopper) this requires the FP8
        # packed KV cache -- ``kv_c_and_k_pe_cache`` is then float8 with the
        # 656-byte packed last dim. Metadata + kernel both take
        # ``is_fp8_kvcache=True``; ``causal=False`` because the top-k set is
        # already causal (the indexer masked future tokens before selecting).
        sparse = topk_indices is not None
        if sparse:
            is_fp8 = kv_c_and_k_pe_cache.dtype == torch.float8_e4m3fn
            topk = topk_indices.shape[-1]
            # indices: (batch, q_len, topk) int32; -1 = invalid/skip.
            if topk_indices.dim() == 2:
                topk_indices = topk_indices.unsqueeze(1)
            # FP8 cache is already [pages, page_sz, 1, packed_dim]; the bf16
            # latent cache is [pages, page_sz, dim] and needs a head dim of 1.
            k_cache = (
                kv_c_and_k_pe_cache
                if kv_c_and_k_pe_cache.dim() == 4
                else kv_c_and_k_pe_cache.unsqueeze(-2)
            )
            flashmla_meta = get_mla_metadata(
                decode_meta.seq_lens,
                q_len * self.num_heads,
                1,
                num_heads_q=self.num_heads,
                is_fp8_kvcache=is_fp8,
                topk=topk,
            )
            tile_scheduler_metadata, num_splits = flashmla_meta
            o, lse = flash_mla_with_kvcache(
                q=q,
                k_cache=k_cache,
                block_table=decode_meta.block_table,
                cache_seqlens=decode_meta.seq_lens,
                head_dim_v=self.kv_lora_rank,
                tile_scheduler_metadata=tile_scheduler_metadata,
                num_splits=num_splits,
                softmax_scale=self.scale,
                causal=False,
                is_fp8_kvcache=is_fp8,
                indices=topk_indices,
            )
            return o.squeeze(1), lse.squeeze(-1)

        # The tile-scheduler metadata only depends on cache_seqlens and the
        # head count, so it is identical for every MLA layer within a single
        # forward step. Compute it once and cache it on the shared decode
        # metadata object to avoid a redundant kernel launch per layer.
        flashmla_meta = decode_meta.flashmla_meta
        if flashmla_meta is None:
            # num_q_tokens_per_head_k = q_len * num_heads_q // num_heads_k,
            # with num_heads_k == 1 (MQA over the latent).
            num_q_tokens_per_head_k = q_len * self.num_heads
            flashmla_meta = get_mla_metadata(
                decode_meta.seq_lens,
                num_q_tokens_per_head_k,
                1,  # num_heads_k: MQA over the compressed latent
            )
            decode_meta.flashmla_meta = flashmla_meta
        tile_scheduler_metadata, num_splits = flashmla_meta

        o, lse = flash_mla_with_kvcache(
            q=q,
            k_cache=kv_c_and_k_pe_cache.unsqueeze(-2),  # add head dim of 1
            block_table=decode_meta.block_table,
            cache_seqlens=decode_meta.seq_lens,
            head_dim_v=self.kv_lora_rank,
            tile_scheduler_metadata=tile_scheduler_metadata,
            num_splits=num_splits,
            softmax_scale=self.scale,
            causal=True,
        )

        # o:   (B, q_len=1, num_heads, kv_lora_rank) -> (B, num_heads, L)
        # lse: (B, num_heads, q_len=1)               -> (B, num_heads)
        return o.squeeze(1), lse.squeeze(-1)

    def _forward_decode_triton(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError("FP8 Triton MLA not yet supported")

        if type(q) is tuple:
            q = torch.cat(q, dim=-1)

        assert isinstance(q, torch.Tensor)
        B = q.shape[0]
        q_num_heads = q.shape[1]
        o = torch.zeros(
            B, q_num_heads, self.kv_lora_rank, dtype=q.dtype, device=q.device
        )
        lse = torch.zeros(B, q_num_heads, dtype=q.dtype, device=q.device)

        # For batch invariance, use only 1 split to ensure deterministic reduction
        num_kv_splits = 4

        # TODO(lucas) Allocate ahead of time
        attn_logits = torch.empty(
            (
                B,
                q_num_heads,
                num_kv_splits,
                # NOTE(lucas) idk why the +1 is here but sglang has it so we
                # just mirror that
                self.kv_lora_rank + 1,
            ),
            dtype=torch.float32,
            device=q.device,
        )

        # Add a head dim of 1
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.unsqueeze(2)
        kv_c_cache = kv_c_and_k_pe_cache[..., : self.kv_lora_rank]
        PAGE_SIZE = kv_c_and_k_pe_cache.size(1)

        # Run MQA
        decode_attention_fwd(
            q,
            kv_c_and_k_pe_cache,
            kv_c_cache,
            o,
            lse,
            attn_metadata.decode.block_table,
            attn_metadata.decode.seq_lens,
            attn_logits,
            num_kv_splits,
            self.scale,
            PAGE_SIZE,
        )

        return o, lse

    def forward(
        self,
        q: torch.Tensor,
        k_c_normed: torch.Tensor,  # key in unified attn
        k_pe: torch.Tensor,  # value in unified attn
        input_data: InputData,
        output: torch.Tensor,
        decode_topk_indices: Optional[torch.Tensor] = None,
        prefill_topk_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """MLA attention.

        ``decode_topk_indices`` / ``prefill_topk_indices`` (DeepSeek Sparse
        Attention / V3.2) carry the indexer's per-query top-k token selection
        for the decode / prefill token ranges respectively. When ``None`` the
        dense (full) attention is used, which is exactly equivalent for
        sequences no longer than ``index_topk`` (top-k then selects all keys).
        """
        assert output is not None, "Output tensor must be provided."

        # profile run (see FlashAttention.forward): guard on ``is None`` so the
        # check survives ``segment`` being a declared ``None`` attribute.
        if getattr(input_data.memory_manager, "segment", None) is None:
            self.process_weights()
            return output

        kv_cache = input_data.memory_manager.segment.kv_cache[self.layer_id]

        attn_metadata = input_data.metadata
        if attn_metadata is None:
            # The zero fill is required when used with DP + EP
            # to ensure all ranks within a DP group compute the
            # same expert outputs.
            return output.fill_(0)

        num_actual_toks = attn_metadata.num_actual_tokens

        # Inputs and outputs may be padded for CUDA graphs
        output_padded = output
        output = output[:num_actual_toks, ...]
        q = q[:num_actual_toks, ...]
        k_c_normed = k_c_normed[:num_actual_toks, ...]
        k_pe = k_pe[:num_actual_toks, ...]

        assert (
            attn_metadata.num_decodes is not None
            and attn_metadata.num_prefills is not None
            and attn_metadata.num_decode_tokens is not None
        )

        has_decode = attn_metadata.num_decodes > 0
        has_prefill = attn_metadata.num_prefills > 0
        num_decode_tokens = attn_metadata.num_decode_tokens

        decode_q = q[:num_decode_tokens]

        prefill_q = q[num_decode_tokens:]
        prefill_k_pe = k_pe[num_decode_tokens:]
        prefill_k_c_normed = k_c_normed[num_decode_tokens:]

        # write the latent and rope to kv cache
        if kv_cache.numel() > 0:
            if getattr(input_data.memory_manager, "mla_cache_fp8", False):
                # DeepSeek Sparse Attention: native FP8-packed MLA latent cache
                # (nope FP8 + per-tile scales + rope bf16). Written directly so
                # the FP8 sparse decode kernel can read it without any runtime
                # dequant. Prefill reads it back via dequant (see below).
                ops.concat_and_cache_mla_fp8(
                    k_c_normed,
                    k_pe.squeeze(1),
                    kv_cache,
                    attn_metadata.slot_mapping.flatten(),
                )
            else:
                ops.concat_and_cache_mla(
                    k_c_normed,
                    k_pe.squeeze(1),
                    kv_cache,
                    attn_metadata.slot_mapping.flatten(),
                    kv_cache_dtype=self.kv_cache_dtype,
                    scale=self._k_scale,
                )

        if has_prefill:
            output[num_decode_tokens:] = self._forward_prefill(
                prefill_q,
                prefill_k_c_normed,
                prefill_k_pe,
                kv_cache,
                attn_metadata,
                self._k_scale,
                topk_indices=prefill_topk_indices,
            )

        if has_decode:
            assert attn_metadata.decode is not None
            decode_q_nope, decode_q_pe = decode_q.split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
            )
            # Convert from (B, N, P) to (N, B, P)
            decode_q_nope = decode_q_nope.transpose(0, 1)

            N, B, P = decode_q_nope.shape
            if self.W_UK_T is None:
                self.process_weights()
            _, _, L = self.W_UK_T.shape

            decode_ql_nope = decode_q_nope.new_empty((N, B, L))

            # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
            torch.bmm(decode_q_nope, self.W_UK_T, out=decode_ql_nope)

            # Convert from (N, B, L) to (B, N, L)
            decode_ql_nope = decode_ql_nope.transpose(0, 1)

            decode_q = (decode_ql_nope, decode_q_pe)

            # call decode attn (sparse when the indexer supplied top-k indices)
            attn_out, lse = self._forward_decode(
                decode_q, kv_cache, attn_metadata, topk_indices=decode_topk_indices
            )

            # v_up projection
            self._v_up_proj(attn_out, out=output[:num_decode_tokens])

        return output_padded
