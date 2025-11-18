import torch

from typing import Optional

from gllm.vllm_flash_attn import flash_attn_varlen_func
from gllm.layers.linear import ColumnParallelLinear
from gllm.layers.ops.triton_decode_attention import decode_attention_fwd
from gllm.layers.ops.merge_attn_states import merge_attn_states
from gllm import _custom_ops as ops
from gllm.input_data import (InputData, MLACommonPrefillMetadata,
                             MLACommonMetadata)
from gllm.utils import get_flash_attn_version


class FlashAttention():

    def __init__(self,
                 layer_id: int,
                 scale: float,
                 num_heads: int,
                 num_key_value_heads: int,
                 head_dim: int):
        self.scale = scale
        self.layer_id = layer_id
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.fa_version = get_flash_attn_version()

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                input_data: InputData):

        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_key_value_heads, self.head_dim)
        v = v.view(-1, self.num_key_value_heads, self.head_dim)

        input_data.memory_manager.batch_store(
            self.layer_id, k, v, input_data.slot_mapping_tensor)

        k_cache = input_data.memory_manager.segment.k_cache[self.layer_id]
        v_cache = input_data.memory_manager.segment.v_cache[self.layer_id]

        out = flash_attn_varlen_func(q,
                                     k_cache,
                                     v_cache,
                                     cu_seqlens_q=input_data.query_start_loc,
                                     max_seqlen_q=input_data.max_query_len,
                                     seqused_k=input_data.seq_lens,
                                     max_seqlen_k=input_data.max_seq_len,
                                     softmax_scale=self.scale,
                                     causal=True,
                                     block_table=input_data.block_table,
                                     fa_version=self.fa_version)
        return out.view(-1, out.shape[-2]*out.shape[-1])
    

class MLAAttention():
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
    ):
        self.scale = scale
        self.layer_id = layer_id
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.fa_version = get_flash_attn_version()

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.kv_b_proj = kv_b_proj

        self._pad_v = True

        self._k_scale = torch.tensor(1.0, dtype=torch.float32)

        kv_b_proj_weight = self.kv_b_proj.weight.T
        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )

        W_UK, W_UV = kv_b_proj_weight.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Convert from (L, N, V) to (N, L, V)
        self.W_UV = W_UV.transpose(0, 1)
        # Convert from (L, N, P) to (N, P, L)
        self.W_UK_T = W_UK.permute(1, 2, 0)
        
        self.kv_cache_dtype = 'auto'

    def _flash_attn_varlen_diff_headdims(self, q, k, v, softmax_scale,
                                         return_softmax_lse, **kwargs):
        maybe_padded_v = v
        if self._pad_v:
            maybe_padded_v = torch.nn.functional.pad(
                v, [0, q.shape[-1] - v.shape[-1]], value=0)

        attn_out = flash_attn_varlen_func(
            q=q,
            k=k,
            v=maybe_padded_v,
            return_softmax_lse=return_softmax_lse,
            softmax_scale=softmax_scale,
            fa_version=self.fa_version,
            **kwargs,
        )

        # Unpack the output if there is multiple results,
        # triton always returns (output, softmax_lse),
        # vllm_flash_attn returns (output, softmax_lse) when
        #  `return_softmax_lse = True`
        # flash_attn (RoCM) returns (output, softmax_lse, ...) when
        #  `return_attn_probs = True`
        rest = None
        if isinstance(attn_out, tuple):
            attn_out, *rest = attn_out

        # Remain consistent with old `flash_attn_varlen_func` where there
        # is only one output tensor if `return_softmax_lse` is False.
        if return_softmax_lse:
            assert rest is not None
            return attn_out, rest[0]
        return attn_out
    
    def _run_prefill_context_chunk(self, prefill: MLACommonPrefillMetadata,
                                      chunk_idx: int, q, k, v):
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

            kv_c_normed = workspace[:toks]\
                [..., :self.kv_lora_rank]
            k_pe = workspace[:toks]\
                [..., self.kv_lora_rank:].unsqueeze(1)

            kv_nope = self.kv_b_proj(kv_c_normed).view( \
                -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = kv_nope\
                .split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

            k = torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))),
                          dim=-1)

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
    
    def _run_prefill_new_tokens(self, prefill: MLACommonPrefillMetadata, q,
                                   k, v, return_softmax_lse):
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

    def _forward_prefill(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        k_scale: torch.Tensor,
    ) -> torch.Tensor:
        assert attn_metadata.prefill is not None

        has_context = attn_metadata.prefill.chunked_context is not None
        kv_nope = self.kv_b_proj(kv_c_normed).view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
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
                q, kv_c_and_k_pe_cache, attn_metadata, k_scale)

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
            output = output[..., :v.shape[-1]]

        return output.flatten(start_dim=-2)

    def _forward_decode(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
    ) -> torch.Tensor:
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
        kv_cache: torch.Tensor,
        input_data: InputData,
        output: torch.Tensor,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."

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
                prefill_q, prefill_k_c_normed, prefill_k_pe, kv_cache,
                attn_metadata, self._k_scale)

        if has_decode:
            assert attn_metadata.decode is not None
            decode_q_nope, decode_q_pe = decode_q.split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
            # Convert from (B, N, P) to (N, B, P)
            decode_q_nope = decode_q_nope.transpose(0, 1)
            
            N, B, P = decode_q_nope.shape
            _, _, L = self.W_UK_T.shape
            
            decode_ql_nope = decode_q_nope.new_empty((N, B, L))
            
            # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
            torch.bmm(decode_q_nope, self.W_UK_T, out=decode_ql_nope)
            
            # Convert from (N, B, L) to (B, N, L)
            decode_ql_nope = decode_ql_nope.transpose(0, 1)
            
            decode_q = (decode_ql_nope, decode_q_pe)
            
            # call decode attn
            attn_out, lse = self._forward_decode(
                decode_q, kv_cache, attn_metadata
            )

            # v_up projection
            self._v_up_proj(attn_out, out=output[:num_decode_tokens])

        return output_padded