"""Native-checkpoint-precision DeepSeek-V4 model components."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from gllm.distributed.parallel_state import get_tp_rank
from gllm.layers.attention.deepseek_v4.cache import DeepseekV4AttentionCache
from gllm.layers.attention.deepseek_v4.layer import DeepseekV4Attention
from gllm.layers.attention.deepseek_v4.ops import serving_max_length
from gllm.layers.deepseek_v4_mhc import mhc_head, mhc_post, mhc_pre
from gllm.layers.layernorm import RMSNorm
from gllm.layers.quantization.fp8 import block_fp8_scale_to_float32
from gllm.layers.moe.deepseek_v4 import DeepseekV4MoE
from gllm.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from gllm.models.weight_loader import (
    LoadContext,
    WeightRule,
    contains,
    run_weight_loader,
)
from gllm.models.weight_utils import (
    copy_single_proj_dim0,
    copy_single_proj_dim1,
    get_tensor_from_dict,
)
from gllm.runtime.memory_manager import (
    DeepseekV4KVCacheConfig,
    DeepseekV4StateCacheConfig,
)
from gllm.runtime.piecewise_cuda_graph import piecewise_dynamic_tensor


class DeepseekV4DecoderLayer(nn.Module):
    """One V4 mHC block in the checkpoint's exact operation order."""

    supports_piecewise_cuda_graph = True

    def __init__(
        self,
        layer_id: int,
        config: Any,
        *,
        attention_cls=DeepseekV4Attention,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps

        self.attn = attention_cls(layer_id, config)
        self.ffn = DeepseekV4MoE(layer_id, config)
        self.attn_norm = RMSNorm(
            self.hidden_size, self.norm_eps, params_dtype=torch.bfloat16
        )
        self.ffn_norm = RMSNorm(
            self.hidden_size, self.norm_eps, params_dtype=torch.bfloat16
        )

        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size
        self.hc_attn_fn = nn.Parameter(
            torch.empty(mix_hc, hc_dim, dtype=torch.float32, device="cuda")
        )
        self.hc_ffn_fn = nn.Parameter(
            torch.empty(mix_hc, hc_dim, dtype=torch.float32, device="cuda")
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32, device="cuda")
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32, device="cuda")
        )
        self.hc_attn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32, device="cuda")
        )
        self.hc_ffn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32, device="cuda")
        )

    def _hc_pre(
        self,
        hidden_states: torch.Tensor,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return mhc_pre(
            hidden_states,
            fn,
            scale,
            base,
            norm_eps=self.norm_eps,
            hc_mult=self.hc_mult,
            sinkhorn_iters=self.hc_sinkhorn_iters,
            hc_eps=self.hc_eps,
        )

    def _attention_input(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        layer_input, post, comb = self._hc_pre(
            hidden_states,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
        )
        return self.attn_norm(layer_input), residual, post, comb

    def _ffn(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        layer_input, post, comb = self._hc_pre(
            hidden_states,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
        )
        output = self.ffn(self.ffn_norm(layer_input), input_ids)
        return mhc_post(output, residual, post, comb)

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        cache: DeepseekV4AttentionCache | None = None,
    ) -> tuple[torch.Tensor, DeepseekV4AttentionCache]:
        layer_input, residual, post, comb = self._attention_input(hidden_states)
        attention_output, cache = self.attn.forward_prefill_with_cache(
            layer_input, cache
        )
        hidden_states = mhc_post(attention_output, residual, post, comb)
        return self._ffn(hidden_states, input_ids), cache

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        position: int,
        cache: DeepseekV4AttentionCache,
    ) -> torch.Tensor:
        layer_input, residual, post, comb = self._attention_input(hidden_states)
        attention_output = self.attn.forward_decode(
            layer_input,
            position=position,
            cache=cache,
        )
        hidden_states = mhc_post(attention_output, residual, post, comb)
        return self._ffn(hidden_states, input_ids)

    def forward_paged(
        self,
        input_data,
        hidden_states: torch.Tensor,
        *,
        local_layer_id: int,
    ) -> torch.Tensor:
        """Packed serving path backed by request-owned/paged cache arenas."""
        layer_input, residual, post, comb = self._attention_input(hidden_states)
        attention_output, residual, post, comb = piecewise_dynamic_tensor(
            lambda x: self.attn.forward_paged(
                input_data,
                x,
                local_layer_id=local_layer_id,
            ),
            layer_input,
            residual,
            post,
            comb,
        )
        hidden_states = mhc_post(attention_output, residual, post, comb)
        return self._ffn(hidden_states, input_data.get_tokens())



class DeepseekV4ModelBase(nn.Module):
    """Embedding, decoder stack, mHC head fold and checkpoint loading.

    Split from :class:`DeepseekV4Model` only so the serving ``forward`` and the
    parameter/weight plumbing stay separately readable; both are production
    code.  The token-at-a-time oracles live in
    :mod:`gllm.models.deepseek_v4_reference`.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.embed = VocabParallelEmbedding(
            config.vocab_size,
            self.hidden_size,
            params_dtype=torch.bfloat16,
        )
        self.layers = nn.ModuleList(
            [
                DeepseekV4DecoderLayer(layer_id, config)
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            self.hidden_size, self.norm_eps, params_dtype=torch.bfloat16
        )
        # The official implementation intentionally promotes the LM-head
        # checkpoint to FP32 and computes logits in FP32.
        self.head = ParallelLMHead(
            config.vocab_size,
            self.hidden_size,
            params_dtype=torch.float32,
        )
        hc_dim = self.hc_mult * self.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(
                self.hc_mult,
                hc_dim,
                dtype=torch.float32,
                device="cuda",
            )
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32, device="cuda")
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32, device="cuda")
        )

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed(input_ids)
        return hidden_states.unsqueeze(-2).expand(
            *hidden_states.shape[:-1], self.hc_mult, self.hidden_size
        ).contiguous()

    def _head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = mhc_head(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            norm_eps=self.norm_eps,
            hc_mult=self.hc_mult,
            hc_eps=self.hc_eps,
        )
        return self.head(self.norm(hidden_states).float())





class DeepseekV4Model(DeepseekV4ModelBase):
    """Serving model: one packed forward over the paged cache arenas."""

    def forward(self, input_data, hidden_states=None, residual=None):
        if hidden_states is None:
            hidden_states = self._embed(input_data.get_tokens())
        elif hidden_states.ndim == 2:
            # PiecewiseGraphRunner supplies precomputed token embeddings. The
            # first static segment expands them into the mHC residual streams,
            # matching ``_embed`` without another embedding lookup.
            hidden_states = hidden_states.unsqueeze(-2).expand(
                *hidden_states.shape[:-1], self.hc_mult, self.hidden_size
            ).contiguous()
        elif hidden_states.ndim != 3 or hidden_states.shape[-2] != self.hc_mult:
            raise ValueError(
                "DeepSeek-V4 hidden states must be [tokens, hidden] embeddings "
                "or [tokens, hc_mult, hidden] residual streams"
            )
        for local_layer_id, layer in enumerate(self.layers):
            hidden_states = layer.forward_paged(
                input_data,
                hidden_states,
                local_layer_id=local_layer_id,
            )
        hidden_states = mhc_head(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            norm_eps=self.norm_eps,
            hc_mult=self.hc_mult,
            hc_eps=self.hc_eps,
        )
        return self.norm(hidden_states)



# ---------------------------------------------------------------------------
# Weight loading
#
# V4 goes through the repository's declarative loader like every other model:
# one rule table keyed on the parameter path, driven by a single pass over
# ``named_parameters()``.  That is what supplies PP layer remapping, the shared
# per-expert thread pool, per-parameter progress and the checkpoint-key
# fallbacks -- none of which a hand-rolled recursive loader gets for free.
#
# Two naming gaps between the module tree and the checkpoint are closed by
# :func:`_v4_src_key` before matching:
#
#   * attention projections live under ``attn.projections.*`` in the module
#     tree but directly under ``attn.*`` in the checkpoint;
#   * gLLM's block-FP8 linears call their scale ``weight_scale_inv`` while the
#     checkpoint calls it ``scale``.
# ---------------------------------------------------------------------------


def _v4_src_key(key: str) -> str:
    """Map a parameter path to its checkpoint key."""
    if key.startswith("model."):
        key = key[len("model.") :]
    key = key.replace(".projections.", ".")
    key = key.replace(".compressor.norm_weight", ".compressor.norm.weight")
    if key.endswith(".weight_scale_inv"):
        key = key[: -len(".weight_scale_inv")] + ".scale"
    return key


def _src(ctx: LoadContext, key: str) -> torch.Tensor:
    return get_tensor_from_dict(ctx.weights, key)


def _h_replicated(ctx: LoadContext, key: str, param: torch.Tensor) -> None:
    """Every rank holds the whole tensor (Q/KV down-projections, norms, router)."""
    param.copy_(_src(ctx, key))


def _h_fp32(ctx: LoadContext, key: str, param: torch.Tensor) -> None:
    """Promote to FP32, which the checkpoint's own reference also does."""
    param.copy_(_src(ctx, key).float())


def _h_column(ctx: LoadContext, key: str, param: torch.Tensor) -> None:
    copy_single_proj_dim0(param, _src(ctx, key))


def _h_row(ctx: LoadContext, key: str, param: torch.Tensor) -> None:
    copy_single_proj_dim1(param, _src(ctx, key))


def _h_column_scale(ctx: LoadContext, key: str, param: torch.Tensor) -> None:
    copy_single_proj_dim0(param, block_fp8_scale_to_float32(_src(ctx, key)))


def _h_row_scale(ctx: LoadContext, key: str, param: torch.Tensor) -> None:
    copy_single_proj_dim1(param, block_fp8_scale_to_float32(_src(ctx, key)))


def _h_replicated_scale(ctx: LoadContext, key: str, param: torch.Tensor) -> None:
    param.copy_(block_fp8_scale_to_float32(_src(ctx, key)))


def _h_vocab_shard(ctx: LoadContext, key: str, param: torch.Tensor) -> None:
    """Vocab-parallel embedding / LM head: copy this rank's vocab window.

    The padded tail past ``num_org_elements`` stays zero so a padded id can
    never alias a real embedding row.
    """
    shard = ctx.extra["vocab_shards"][key]
    param.zero_()
    weight = _src(ctx, key)
    param[: shard.num_org_elements].copy_(
        weight[shard.org_vocab_start_index : shard.org_vocab_end_index].to(
            param.dtype
        )
    )


def _h_attn_sink(ctx: LoadContext, key: str, param: torch.Tensor) -> None:
    """Per-head sink logits follow the attention head sharding."""
    heads = param.shape[0]
    rank = get_tp_rank()
    param.copy_(_src(ctx, key)[rank * heads : (rank + 1) * heads].float())


def _h_router_bias(ctx: LoadContext, key: str, param: torch.Tensor) -> None:
    param.copy_(
        _src(ctx, key.replace("e_score_correction_bias", "gate.bias")).float()
    )


def _h_hash_table(ctx: LoadContext, key: str, param: torch.Tensor) -> None:
    param.copy_(_src(ctx, key.replace("tid2eid", "gate.tid2eid")))


def _h_shared_w13(ctx: LoadContext, key: str, param: torch.Tensor) -> None:
    """Shared expert gate (``w1``) ++ up (``w3``), stacked column-parallel."""
    base, field = key.rsplit(".gate_up_proj.", 1)
    suffix = "scale" if field == "scale" else "weight"
    half = param.shape[0] // 2
    rank = get_tp_rank()
    for offset, name in ((0, "w1"), (half, "w3")):
        source = _src(ctx, f"{base}.{name}.{suffix}")
        if suffix == "scale":
            source = block_fp8_scale_to_float32(source)
        param[offset : offset + half].copy_(
            source[rank * half : (rank + 1) * half]
        )


def _h_shared_w2(ctx: LoadContext, key: str, param: torch.Tensor) -> None:
    """Shared expert down projection (``w2``), row-parallel."""
    base, field = key.rsplit(".down_proj.", 1)
    suffix = "scale" if field == "scale" else "weight"
    source = _src(ctx, f"{base}.w2.{suffix}")
    if suffix == "scale":
        source = block_fp8_scale_to_float32(source)
    copy_single_proj_dim1(param, source)


def _h_mxfp4_experts(ctx: LoadContext, key: str, param: torch.Tensor) -> None:
    """One stacked routed-expert tensor, filled from per-expert checkpoint keys.

    ``ctx.pool`` is the loader-wide expert thread pool, so a V4 layer's hundreds
    of per-expert reads overlap instead of running one at a time.
    """
    prefix, field = key.rsplit(".", 1)
    ctx.extra["experts"][prefix].load_stacked_param(
        field, param, ctx.weights, prefix, pool=ctx.pool
    )


class DeepseekV4ForCausalLM(nn.Module):
    """gLLM serving wrapper for native-precision DeepSeek-V4 checkpoints."""

    supports_full_cuda_graph = True

    def __init__(self, config: Any) -> None:
        super().__init__()
        from gllm.distributed.parallel_state import get_pp_size, is_dp_attn

        if get_pp_size() != 1:
            raise NotImplementedError(
                "DeepSeek-V4 initial support requires pipeline parallel size 1"
            )
        if is_dp_attn():
            raise NotImplementedError(
                "DeepSeek-V4 initial support does not yet implement DP-attention/EP routing"
            )
        self.config = config
        self.max_model_len = serving_max_length(config)
        self.num_kv_heads = 1
        self.head_dim = config.head_dim
        self.model = DeepseekV4Model(config)
        self.lm_head = self.model.head
        self.num_layers = len(self.model.layers)
        self.start_layer = 0
        self.end_layer = self.num_layers
        self.ret_residual = False
        self.mtp = None
        self.dspark = None
        if (
            getattr(config, "mtp_enabled", False)
            and getattr(config, "dspark_block_size", 0) > 0
        ):
            from gllm.models.deepseek_v4_dspark import DeepseekV4DSpark

            self.dspark = DeepseekV4DSpark(
                config,
                embed=self.model.embed,
                lm_head=self.lm_head,
            )
        self.dsv4_kv_cache_config = DeepseekV4KVCacheConfig.from_model_config(
            config
        )
        self.dsv4_state_cache_config = (
            DeepseekV4StateCacheConfig.from_model_config(config)
        )

    def forward(self, input_data, hidden_states=None, residual=None):
        return self.model(input_data, hidden_states, residual)

    def compute_logits(self, input_data, hidden_states: torch.Tensor):
        indices = input_data.get_query_start_loc()[1:] - 1
        return self.logits_from_hidden(hidden_states[indices])

    def logits_from_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states.float())

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed(input_ids)

    def weight_rules(self):
        """Ordered rule table; the first match per parameter wins.

        Read it as a description of how each family of V4 tensors is sharded:
        replicated, column-parallel (output dim), row-parallel (input dim),
        vocab-parallel, per-head, or per-expert.
        """
        return [
            # -- vocab-parallel ------------------------------------------
            WeightRule(contains("embed.weight", "head.weight"), _h_vocab_shard, "vocab"),
            # -- mHC mixing coefficients and norms: replicated, FP32 ------
            WeightRule(contains("hc_"), _h_fp32, "mhc"),
            WeightRule(contains("attn_sink"), _h_attn_sink, "attn_sink"),
            # -- routed-expert stacks (must precede the generic linears) --
            WeightRule(contains("ffn.experts."), _h_mxfp4_experts, "routed_experts"),
            # -- shared expert: gate++up stacked, down row-parallel -------
            WeightRule(contains("shared_experts.gate_up_proj"), _h_shared_w13, "shared_w13"),
            WeightRule(contains("shared_experts.down_proj"), _h_shared_w2, "shared_w2"),
            # -- router --------------------------------------------------
            WeightRule(contains("e_score_correction_bias"), _h_router_bias, "router_bias"),
            WeightRule(contains("tid2eid"), _h_hash_table, "hash_routing"),
            # -- learned compressors ------------------------------------
            # Listed before the attention projections: a compressor has its own
            # ``wkv`` and would otherwise be captured by the rule below.
            # Replicated, and promoted to FP32 as the reference does.
            WeightRule(contains("compressor."), _h_fp32, "compressor"),
            # -- attention / indexer projections -------------------------
            # ``wq_a`` and ``wkv`` are the low-rank down-projections: every
            # rank needs the whole latent, so they stay replicated.
            WeightRule(contains("wq_a.weight", "wkv.weight"), _h_replicated, "replicated_w"),
            WeightRule(contains("wq_b.scale", "wo_a.scale"), _h_column_scale, "column_scale"),
            WeightRule(contains("wq_b.weight", "wo_a.weight", "weights_proj.weight"), _h_column, "column_w"),
            WeightRule(contains("wo_b.scale"), _h_row_scale, "row_scale"),
            WeightRule(contains("wo_b.weight"), _h_row, "row_w"),
            # -- any remaining block-FP8 scale belongs to a replicated
            #    linear (``wq_a``, ``wkv``, DSpark's ``main_proj``). It must
            #    still go through the E8M0 conversion, so this has to precede
            #    the catch-all below.
            WeightRule(lambda key: key.endswith(".scale"), _h_replicated_scale, "replicated_scale"),
            # -- everything else (RMSNorm weights, router matrix) --------
            WeightRule(lambda _: True, _h_replicated, "default"),
        ]

    def _make_load_context(self, weights) -> LoadContext:
        """Bind the two lookups a handler cannot derive from a key alone.

        Both are keyed by the *checkpoint* key the handler is called with, not
        by parameter identity: ``run_weight_loader`` hands handlers ``v.data``,
        which is a fresh tensor object on every access. Keying by the resolved
        key also keeps this correct under pipeline parallelism, where the key
        carries the global layer id while ``self.model.layers`` is local.
        """
        vocab_shards = {
            "embed.weight": self.model.embed.shard_indices,
            "head.weight": self.model.head.shard_indices,
        }
        experts = {
            f"layers.{self.start_layer + local}.ffn.experts": layer.ffn.experts
            for local, layer in enumerate(self.model.layers)
        }
        return LoadContext(
            weights=weights,
            num_experts=self.config.n_routed_experts,
            extra={"vocab_shards": vocab_shards, "experts": experts},
        )

    def load_weights(self, weights, mp_load_progress=None) -> None:
        # DSpark's parameters live under ``mtp.*`` in the checkpoint, a
        # namespace the rule table above knows nothing about. Detach the head
        # for the base pass and load it separately, as DeepSeek-V3.2 does.
        dspark = self.dspark
        self.dspark = None
        try:
            run_weight_loader(
                self,
                weights,
                self.weight_rules(),
                mp_load_progress,
                pp_idx_offset=2,
                start_layer=self.start_layer,
                ctx=self._make_load_context(weights),
                src_key_fn=_v4_src_key,
            )
        finally:
            self.dspark = dspark

        for layer in self.model.layers:
            layer.ffn.experts.process_weights_after_loading()
        if self.dspark is not None:
            self.dspark.load_weights(weights, self, mp_load_progress)
            self.dspark.process_weights_after_loading()


__all__ = [
    "DeepseekV4DecoderLayer",
    "DeepseekV4ForCausalLM",
    "DeepseekV4Model",
    "DeepseekV4ModelBase",
]
