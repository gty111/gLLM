"""Qwen3.8-Flash-Next text, vision and MTP inference.

Reuses gLLM's TP/EP GDN and MoE. Raw index keys live in ordinary paged KV
storage, so prefix reuse / eviction follows the same ownership as K/V. The
query path caches compressed index keys and uses paged GPU selection and
attention for decode, prefill and verify. PLE histories share
the SSM request slot, prefix snapshot and speculative rollback lifecycle.
"""

import copy

import torch
from torch import nn

from gllm.distributed.parallel_state import get_pp_size, get_tp_size, is_dp_attn
from gllm.layers.attention.paged_sparse import (
    paged_sparse_attention,
    select_paged_query,
    update_compressed_index,
)
from gllm.layers.hyper_connection import GatedResidual
from gllm.layers.layernorm import GemmaRMSNorm
from gllm.layers.linear import QKVParallelLinear, RowParallelLinear
from gllm.layers.ops.qsa import prepare_qsa
from gllm.layers.ops.state_snapshot import copy_slot_snapshots
from gllm.layers.position_learning import (
    NGramHash,
    PLELayer,
    packed_history_tail,
    packed_token_histories,
)
from gllm.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from gllm.models.qwen2_moe import Qwen2MoeSparseMoeBlock
from gllm.models.qwen3_5 import Qwen3_5GatedDeltaNet, _build_rope
from gllm.models.qwen3_5_moe import Qwen3_5MoeLLMForCausalLM
from gllm.models.qwen3_vl import Qwen3VLForConditionalGeneration
from gllm.models.weight_utils import ShardedEmbeddingLookup, get_tensor_from_dict
from gllm.runtime.piecewise_cuda_graph import PiecewiseRuntime, piecewise_dynamic_tensor


def parse_text_config(raw):
    """Normalize native Transformers config and derive cache/stream geometry."""
    from transformers import Qwen4ExpTextConfig

    text = dict(raw.get("text_config", raw))
    if text.get("model_type") != "qwen4_exp_text":
        raise ValueError("Expected a qwen4_exp_text config")
    aliases = {"full_attention", "qwen_sparse_attention", "indexed_attention"}
    text["layer_types"] = [
        "indexed_attention" if kind in aliases else kind for kind in text["layer_types"]
    ]
    # Validate serving-specific geometry before native config construction.
    # Transformers may otherwise reject it with a less specific message.
    from types import SimpleNamespace

    geometry = SimpleNamespace(**text)
    geometry.ngram_heads = (geometry.ngram_size - 1) * geometry.heads_per_ngram
    _validate_text_config(geometry)
    config = Qwen4ExpTextConfig.from_dict(text)
    config.hyper_hidden_size = config.hidden_size * config.hc_count
    config.qsa_layer_ids = tuple(
        i for i, kind in enumerate(config.layer_types) if kind == "indexed_attention"
    )
    config.gdn_layer_ids = tuple(
        i for i, kind in enumerate(config.layer_types) if kind == "linear_attention"
    )
    config.ple_layer_indices = tuple(i - 1 for i in config.ple_layer_ids)
    config.ngram_heads = (config.ngram_size - 1) * config.heads_per_ngram
    config.ple_head_dim = config.ple_embed_dim // config.ngram_heads
    config.ple_history_length = (config.ple_conv_kernel_size - 1) * config.ngram_size
    return config


def _validate_text_config(config):
    positive_fields = (
        "hidden_size",
        "num_hidden_layers",
        "hc_count",
        "hc_lowrank",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "indexer_n_heads",
        "indexer_kv_heads",
        "indexer_head_dim",
        "indexer_budget",
        "indexer_compress_ratio",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_conv_kernel_dim",
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "vocab_size",
        "ple_embed_dim",
        "ple_conv_kernel_size",
        "ngram_size",
        "heads_per_ngram",
        "ngram_vocab_size_base",
        "split_ngram_parts",
        "make_ngram_vocab_size_divisible_by",
    )
    for name in positive_fields:
        if getattr(config, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if config.shared_expert_intermediate_size < 0:
        raise ValueError("shared_expert_intermediate_size must be non-negative")
    if not 0 <= config.eos_token_id < config.vocab_size:
        raise ValueError("eos_token_id must be within vocab_size")
    if config.rms_norm_eps <= 0:
        raise ValueError("rms_norm_eps must be positive")
    if len(config.layer_types) != config.num_hidden_layers:
        raise ValueError("layer_types must have one entry per decoder layer")
    if set(config.layer_types) - {"linear_attention", "indexed_attention"}:
        raise ValueError("Unknown Qwen4Exp layer type")
    if len(set(config.ple_layer_ids)) != len(config.ple_layer_ids) or any(
        i < 1 or i > config.num_hidden_layers for i in config.ple_layer_ids
    ):
        raise ValueError("ple_layer_ids must be unique one-based decoder IDs")
    if config.num_attention_heads % config.num_key_value_heads:
        raise ValueError("Attention heads must be divisible by KV heads")
    if config.indexer_kv_heads != 1:
        raise ValueError("Qwen4Exp QSA reference requires one indexer KV head")
    if config.indexer_budget % config.indexer_compress_ratio:
        raise ValueError("indexer_budget must be divisible by compress ratio")
    if config.linear_num_value_heads % config.linear_num_key_heads:
        raise ValueError("GDN value heads must be divisible by key heads")
    if config.num_experts_per_tok > config.num_experts:
        raise ValueError("num_experts_per_tok exceeds num_experts")
    if config.ngram_size < 2 or config.ple_embed_dim % config.ngram_heads:
        raise ValueError("ple_embed_dim must be divisible by total ngram heads")


def validate_tp_config(config, size: int) -> None:
    if size <= 0:
        raise ValueError("TP size must be positive")
    for name in (
        "num_attention_heads",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
    ):
        if getattr(config, name) % size:
            raise ValueError(f"{name} must be divisible by TP size {size}")
    kv = config.num_key_value_heads
    if kv % size and size % kv:
        raise ValueError("KV heads must divide or be divided by TP size")


def _ple_source_slots(input_data):
    slots = input_data.get_recurrent_state_slot_per_seq().long().clone()
    count = int(getattr(input_data, "num_mtp_verify_rows", 0))
    if count:
        table = input_data.get_ssm_block_table_2d()[:count]
        columns = input_data.get_ssm_num_accepted()[:count].long() - 1
        slots[:count] = table.gather(1, columns[:, None]).squeeze(1).long()
    return slots


def _ple_context(input_data, config):
    slots = _ple_source_slots(input_data)
    length = config.ngram_size - 1
    context = input_data.get_tokens().new_full(
        (slots.shape[0], length), config.eos_token_id
    )
    segment = input_data.memory_manager.ssm_segment
    if segment is not None:
        context = torch.where(
            input_data.get_has_initial_state_per_seq()[:, None],
            segment.auxiliary_states["ple_tokens"][slots],
            context,
        )
    return slots, context


def _ple_verify_checkpoints(input_data, state, inputs, history):
    count = int(getattr(input_data, "num_mtp_verify_rows", 0))
    if count:
        table = input_data.get_ssm_block_table_2d()[:count]
        tokens = count * table.shape[1]
        checkpoints = packed_token_histories(
            inputs, input_data.get_query_start_loc(), history
        )
        state.index_copy_(0, table.flatten().long(), checkpoints[:tokens])


class QSAIndexer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.indexer_n_heads
        self.head_dim = config.indexer_head_dim
        self.index_qk_proj = nn.Linear(
            config.hidden_size,
            (self.heads + 1) * self.head_dim,
            bias=False,
            device="cuda",
        )
        self.q_layernorm = GemmaRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_layernorm = GemmaRMSNorm(self.head_dim, config.rms_norm_eps)

    def project(self, hidden):
        projected = self.index_qk_proj(hidden)
        q, k = projected.split([self.heads * self.head_dim, self.head_dim], -1)
        return self.q_layernorm(q.reshape(-1, self.heads, self.head_dim)), k


class Qwen4ExpAttention(nn.Module):
    def __init__(self, config, kv_layer_id):
        super().__init__()
        self.kv_layer_id = kv_layer_id
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.num_heads = self.total_num_heads // get_tp_size()
        self.num_kv_heads = max(1, self.total_num_kv_heads // get_tp_size())
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.attn_output_gate = True
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads * 2,
            self.total_num_kv_heads,
            bias=False,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, config.rms_norm_eps)
        self.rotary_emb = _build_rope(
            self.head_dim,
            min(
                config.max_position_embeddings,
                getattr(config, "model_max_length", config.max_position_embeddings),
            ),
            config,
        )
        self.indexer = QSAIndexer(config)
        self.budget = config.indexer_budget
        self.ratio = config.indexer_compress_ratio
        self.max_context = min(
            config.max_position_embeddings,
            getattr(config, "model_max_length", config.max_position_embeddings),
        )
        rotary = self.rotary_emb
        if getattr(rotary, "mrope_interleaved", False):
            axes = rotary.interleaved_axis[: rotary.rotary_dim // 2]
        elif getattr(rotary, "mrope_section", None):
            axes = torch.tensor(
                [
                    axis
                    for axis, size in enumerate(rotary.mrope_section)
                    for _ in range(size)
                ],
                dtype=torch.int32,
                device=rotary.cos_sin_cache.device,
            )
        else:
            axes = torch.zeros(
                rotary.rotary_dim // 2,
                dtype=torch.int32,
                device=rotary.cos_sin_cache.device,
            )
        self.register_buffer("_rotary_axes", axes, persistent=False)

    def forward(self, input_data, hidden):
        output = self._forward_core(
            input_data, self.qkv_proj(hidden), self.indexer.index_qk_proj(hidden)
        )
        return self.o_proj(output)

    def forward_piecewise(self, input_data, hidden, *passthrough):
        qkv = self.qkv_proj(hidden)
        index_qk = self.indexer.index_qk_proj(hidden)
        output, *bridged = piecewise_dynamic_tensor(
            lambda projected, index: self._forward_core(input_data, projected, index),
            qkv,
            *passthrough,
            inputs=(index_qk,),
            output_shape=(self.q_size,),
        )
        return self.o_proj(output), *bridged

    def _forward_core(self, input_data, qkv, index_qk):
        gate = qkv[:, : 2 * self.q_size].reshape(-1, self.num_heads, 2 * self.head_dim)[
            ..., self.head_dim :
        ]
        positions = input_data.get_position()
        if getattr(self.rotary_emb, "mrope_section", None):
            positions = input_data.mrope_positions[:, : qkv.shape[0]]
        segment = input_data.memory_manager.segment
        if segment is None:
            # Startup profiles activations before allocating the cache arena.
            return qkv.new_zeros((qkv.shape[0], self.q_size))
        key_cache = segment.k_cache[self.kv_layer_id]
        value_cache = segment.v_cache[self.kv_layer_id]
        index_cache = segment.index_k_cache[self.kv_layer_id]
        q, iq, requests = prepare_qsa(
            qkv,
            index_qk,
            self.q_norm.weight,
            self.k_norm.weight,
            self.indexer.q_layernorm.weight,
            positions,
            self._rotary_axes,
            input_data.get_slot_mapping(),
            input_data.get_query_start_loc(),
            key_cache,
            value_cache,
            index_cache,
            self.rotary_emb.cos_sin_cache,
            heads=self.num_heads,
            kv_heads=self.num_kv_heads,
            index_heads=self.indexer.heads,
            head_dim=self.head_dim,
            index_dim=self.indexer.head_dim,
            rotary_dim=self.rotary_emb.rotary_dim,
            eps=self.q_norm.variance_epsilon,
            page_size=segment.page_size,
        )
        logical_positions = input_data.positions[: q.shape[0]]
        table = input_data.get_block_table()
        update_compressed_index(
            index_cache,
            table,
            logical_positions,
            requests,
            self.indexer.k_layernorm.weight,
            page_size=segment.page_size,
            ratio=self.ratio,
            rotary_dim=self.rotary_emb.rotary_dim,
            eps=self.indexer.k_layernorm.variance_epsilon,
        )
        selected = select_paged_query(
            iq,
            index_cache,
            table,
            logical_positions,
            requests,
            max_context=self.max_context,
            search_context=(
                None
                if torch.cuda.is_current_stream_capturing()
                else input_data.block_table_cpu.shape[1] * segment.page_size
            ),
            page_size=segment.page_size,
            ratio=self.ratio,
            budget=self.budget,
        )
        output = paged_sparse_attention(
            q,
            key_cache,
            value_cache,
            table,
            selected,
            requests,
            page_size=segment.page_size,
            scale=self.scaling,
        )
        return (output * gate.sigmoid()).reshape(-1, self.q_size)


class Qwen4ExpDecoderLayer(nn.Module):
    supports_piecewise_cuda_graph = True

    def __init__(
        self,
        config,
        parsed,
        layer_id,
        *,
        kv_layer_id=None,
        ssm_layer_id=None,
        ple_dense_id=None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.self_attn = self.linear_attn = None
        if parsed.layer_types[layer_id] == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_id, ssm_layer_id)
            activation = getattr(config, "output_gate_type", None) or config.hidden_act
            if activation not in {"silu", "swish", "sigmoid"}:
                raise NotImplementedError(f"Unsupported GDN output gate: {activation}")
            self.linear_attn.norm.activation = activation
        else:
            self.self_attn = Qwen4ExpAttention(config, kv_layer_id)
        self.mlp = Qwen2MoeSparseMoeBlock(config)
        for name in ("attn_hyper_connection", "mlp_hyper_connection"):
            setattr(
                self,
                name,
                GatedResidual(
                    parsed.hidden_size,
                    parsed.hc_count,
                    parsed.hc_lowrank,
                    parsed.rms_norm_eps,
                    device="cuda",
                ),
            )
        self.ple = None
        self.ple_dense_id = ple_dense_id
        self.ple_lookup = None
        if ple_dense_id is not None:
            self.ple = PLELayer(parsed, device="cuda")
            self.ple.ple_embedding = NGramHash(parsed, ple_dense_id, device="cuda")

    def _ple_block(self, input_data, streams):
        if self.ple is not None:
            _, ngram_context = _ple_context(input_data, self.ple.ple_embedding)
            if self.ple_lookup is None:
                raise RuntimeError("PLE checkpoint rows have not been loaded")
            ids = self.ple.ple_embedding.forward_packed(
                input_data.get_tokens(), input_data.get_query_start_loc(), ngram_context
            )
            embeddings = self.ple_lookup(
                ids, device=streams.device, dtype=streams.dtype
            ).flatten(-2)
            segment = input_data.memory_manager.ssm_segment
            states = (
                None if segment is None else segment.auxiliary_states["ple_history"]
            )
            slots = _ple_source_slots(input_data)
            if states is None:
                history = streams.new_zeros(
                    slots.shape[0],
                    self.ple.dilation * (self.ple.conv1d.weight.shape[-1] - 1),
                    streams.shape[-1],
                )
            else:
                history = states[slots, self.ple_dense_id]
                history = (
                    history * input_data.get_has_initial_state_per_seq()[:, None, None]
                )
            initial_history = history
            streams, history, normalized = self.ple.forward_packed(
                streams,
                embeddings,
                input_data.get_query_start_loc(),
                history,
                return_normalized=True,
            )
            if states is not None:
                states[:, self.ple_dense_id].index_copy_(0, slots, history)
                _ple_verify_checkpoints(
                    input_data,
                    states[:, self.ple_dense_id],
                    normalized,
                    initial_history,
                )
        return streams

    def forward(self, input_data, streams):
        if self.ple is not None:
            streams = piecewise_dynamic_tensor(
                lambda x: self._ple_block(input_data, x), streams
            )
        hidden, residual = self.attn_hyper_connection.mix(streams)
        attention = self.self_attn if self.self_attn is not None else self.linear_attn
        if PiecewiseRuntime.current() is not None:
            hidden, streams, injection = attention.forward_piecewise(
                input_data, hidden, *residual
            )
        else:
            hidden = attention(input_data, hidden)
            streams, injection = residual
        hidden, residual = self.mlp_hyper_connection.combine_and_mix(
            hidden, (streams, injection)
        )
        return self.mlp_hyper_connection.combine(self.mlp(hidden), residual)


class Qwen4ExpModel(nn.Module):
    def __init__(self, config, parsed):
        super().__init__()
        self.config = config
        self.parsed = parsed
        self.start_layer, self.end_layer = 0, parsed.num_hidden_layers
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        ple_indices = sorted(parsed.ple_layer_indices)
        self.layers = nn.ModuleList()
        kv = ssm = 0
        self.ssm_layer_global_ids = []
        for index, kind in enumerate(parsed.layer_types):
            layer = Qwen4ExpDecoderLayer(
                config,
                parsed,
                index,
                kv_layer_id=kv if kind == "indexed_attention" else None,
                ssm_layer_id=ssm if kind == "linear_attention" else None,
                ple_dense_id=ple_indices.index(index) if index in ple_indices else None,
            )
            self.layers.append(layer)
            if kind == "linear_attention":
                ssm += 1
                self.ssm_layer_global_ids.append(index)
            else:
                kv += 1
        self.num_kv_layers, self.num_ssm_layers = kv, ssm
        from gllm.layers.ops.fla.prefill import share_prefill_input_workspace

        share_prefill_input_workspace(self.layers)
        self.hyper_connection_mixer = GatedResidual(
            parsed.hidden_size,
            parsed.hc_count,
            parsed.hc_lowrank,
            parsed.rms_norm_eps,
            use_combine=False,
            device="cuda",
        )

    def embed_input_ids(self, ids):
        return self.embed_tokens(ids)

    def forward(
        self, input_data, hidden_states=None, residual=None, deepstack_input_embeds=None
    ):
        if deepstack_input_embeds:
            raise ValueError("Flash-Next does not use deepstack visual embeddings")
        if residual is not None:
            raise ValueError("Flash-Next owns its residual streams")
        if hidden_states is None:
            hidden_states = self.embed_input_ids(input_data.get_tokens())
        streams = (
            hidden_states.unsqueeze(1).expand(-1, self.parsed.hc_count, -1).flatten(1)
        )
        for layer in self.layers:
            streams = layer(input_data, streams)
        streams = piecewise_dynamic_tensor(
            lambda x: self._finish_states(input_data, x), streams
        )
        if getattr(self.config, "mtp_enabled", False):
            return streams
        hidden, _ = self.hyper_connection_mixer.mix(streams)
        return hidden

    def _finish_states(self, input_data, streams):
        if not self.parsed.ple_layer_ids:
            return streams
        segment = input_data.memory_manager.ssm_segment
        slots, context = _ple_context(input_data, self.parsed)
        # Snapshot all PLE histories alongside the GDN states. Their slot is
        # shared, so prefix restore/free/preemption need no separate allocator.
        if segment is not None:
            token_history = packed_history_tail(
                input_data.get_tokens(), input_data.get_query_start_loc(), context
            )
            segment.auxiliary_states["ple_tokens"].index_copy_(0, slots, token_history)
            _ple_verify_checkpoints(
                input_data,
                segment.auxiliary_states["ple_tokens"],
                input_data.get_tokens(),
                context,
            )
        indices = input_data.get_ssm_snapshot_copy_indices()
        if segment is not None and input_data.max_query_len == 1:
            targets = input_data.get_ssm_snapshot_write_slot_per_seq()
            for state in segment.auxiliary_states.values():
                copy_slot_snapshots(state, slots, targets)
            input_data.mark_ssm_snapshot_writes_enqueued()
        elif segment is not None and indices is not None:
            src, dst = indices
            for state in segment.auxiliary_states.values():
                copy_slot_snapshots(state, src, dst)
            input_data.mark_ssm_snapshot_writes_enqueued()
        return streams


class Qwen4ExpForCausalLM(Qwen3_5MoeLLMForCausalLM):
    """TP/EP text decoder, with optional HC-stream MTP state."""

    supports_full_cuda_graph = True
    supports_piecewise_cuda_graph = True
    supports_mtp_verify_cuda_graph = True

    def __init__(self, config):
        nn.Module.__init__(self)
        if get_pp_size() != 1 or is_dp_attn():
            raise NotImplementedError("Flash-Next requires PP=1 and DP=1")
        text = getattr(config, "text_config", config)
        text.mtp_enabled = getattr(
            config, "mtp_enabled", getattr(text, "mtp_enabled", False)
        )
        quant = getattr(text, "quantization_config", None)
        if quant:
            if quant.get("quant_method") != "fp8" or quant.get("weight_block_size") != [
                128,
                128,
            ]:
                raise NotImplementedError(
                    "Flash-Next currently supports BF16 or block-128 FP8 checkpoints"
                )
            # Official FP8 checkpoints quantize routed experts and PLE rows.
            # Attention, GDN and shared experts stay BF16.
            ignored = quant.get("modules_to_not_convert", [])
            for index, kind in enumerate(text.layer_types):
                projections = (
                    [
                        "linear_attn." + name
                        for name in (
                            "in_proj_qkv",
                            "in_proj_z",
                            "in_proj_a",
                            "in_proj_b",
                            "out_proj",
                        )
                    ]
                    if kind == "linear_attention"
                    else [
                        "self_attn." + name
                        for name in (
                            "q_proj",
                            "k_proj",
                            "v_proj",
                            "o_proj",
                            "indexer.index_qk_proj",
                        )
                    ]
                )
                if text.shared_expert_intermediate_size:
                    projections.extend(
                        "mlp.shared_expert." + name
                        for name in ("gate_proj", "up_proj", "down_proj")
                    )
                for name in projections:
                    name = f"model.language_model.layers.{index}.{name}"
                    if not any(
                        name == skip or name.startswith(skip + ".") for skip in ignored
                    ):
                        raise NotImplementedError(
                            f"Flash-Next FP8 dense projection is unsupported: {name}"
                        )
            text = copy.deepcopy(text)
            text.moe_quantization_config = quant
            text.quantization_config = None
        if torch.get_default_dtype() != torch.bfloat16:
            raise ValueError("Flash-Next requires BF16 activations")
        if getattr(config, "skip_language", False):
            raise NotImplementedError(
                "Flash-Next visual-only execution is not implemented"
            )
        parsed = parse_text_config(text.to_dict())
        validate_tp_config(parsed, get_tp_size())
        self.config = text
        self.parsed_config = parsed
        self.model = Qwen4ExpModel(text, parsed)
        self.num_layers = parsed.num_hidden_layers
        self.start_layer, self.end_layer = 0, self.num_layers
        self.max_model_len = text.max_position_embeddings
        self.num_ssm_layers = self.model.num_ssm_layers
        self.num_kv_heads = parsed.num_key_value_heads
        self.head_dim = parsed.head_dim
        attn = next(
            layer.self_attn
            for layer in self.model.layers
            if layer.self_attn is not None
        )
        self.index_head_dim = (
            (2 * parsed.indexer_head_dim + attn.rotary_emb.rotary_dim + 127) // 128
        ) * 128
        self.requires_mrope_buffer = bool(
            getattr(attn.rotary_emb, "mrope_section", None)
        )
        self.ret_residual = False
        self.mtp = None
        self.ssm_cache_config = self._build_ssm_cache_config(text)
        if parsed.ple_layer_ids:
            self.ssm_cache_config.auxiliary_shapes["ple_tokens"] = (
                parsed.ngram_size - 1,
            )
            self.ssm_cache_config.auxiliary_dtypes["ple_tokens"] = torch.int64
            self.ssm_cache_config.auxiliary_shapes["ple_history"] = (
                len(parsed.ple_layer_ids),
                parsed.ple_history_length,
                parsed.hyper_hidden_size,
            )
        self.lm_head = ParallelLMHead(parsed.vocab_size, parsed.hidden_size)
        self.output_hidden_size = (
            parsed.hyper_hidden_size
            if getattr(config, "mtp_enabled", False)
            else parsed.hidden_size
        )
        if getattr(config, "mtp_enabled", False):
            self.mtp = Qwen4ExpMTP(
                text, parsed, self.model, self.lm_head, self.model.num_kv_layers
            )
        self.checkpoint_path = getattr(config, "checkpoint_path", None)

    def logits_from_hidden(self, hidden_states):
        if hidden_states.shape[-1] == self.parsed_config.hyper_hidden_size:
            hidden_states, _ = self.model.hyper_connection_mixer.mix(hidden_states)
        return self.lm_head(hidden_states)

    def load_weights(self, weights, mp_load_progress=None):
        super().load_weights(weights, mp_load_progress)
        for module in self.modules():
            if isinstance(module, GatedResidual):
                module.process_weights_after_loading()
        for layer in self.model.layers:
            if layer.ple is None:
                continue
            prefix = f"model.layers.{layer.layer_id}.ple.ple_embedding"
            for name, buffer in layer.ple.ple_embedding.named_buffers():
                source = get_tensor_from_dict(weights, f"{prefix}.{name}")
                if source.shape != buffer.shape or source.dtype != torch.int64:
                    raise ValueError(f"Invalid PLE hash buffer: {prefix}.{name}")
                buffer.copy_(source)
            if self.checkpoint_path is None:
                raise ValueError(
                    "Flash-Next needs a local indexed checkpoint for PLE lookup"
                )
            prefix = f"model.language_model.layers.{layer.layer_id}.ple.ple_embedding.ngram_embedding"
            source = ShardedEmbeddingLookup(
                self.checkpoint_path,
                [
                    f"{prefix}.shard_{part}.weight"
                    for part in range(self.parsed_config.split_ngram_parts)
                ],
            )
            try:
                layer.ple_lookup = source.load_parallel()
            finally:
                source.close()


class Qwen4ExpMTP(nn.Module):
    """MTP carries all HC branches between target and successive draft steps."""

    def __init__(self, config, parsed, parent, lm_head, kv_layer_id):
        super().__init__()
        self._parent, self._head = [parent], [lm_head]
        self.hidden_size, self.hc_count = parsed.hidden_size, parsed.hc_count
        self.pre_fc_norm_embedding = GemmaRMSNorm(
            parsed.hidden_size, parsed.rms_norm_eps
        )
        self.pre_fc_norm_hidden = GemmaRMSNorm(
            parsed.hyper_hidden_size, parsed.rms_norm_eps
        )
        self.fc_embedding = nn.Linear(
            parsed.hidden_size, parsed.hidden_size, bias=False, device="cuda"
        )
        self.fc_hidden = nn.Linear(
            parsed.hidden_size, parsed.hidden_size, bias=False, device="cuda"
        )
        draft = copy.deepcopy(parsed)
        draft.layer_types = ["indexed_attention"]
        draft.ple_layer_ids = []
        self.mtp_block = Qwen4ExpDecoderLayer(config, draft, 0, kv_layer_id=kv_layer_id)
        self.hyper_connection_mixer = GatedResidual(
            parsed.hidden_size,
            parsed.hc_count,
            parsed.hc_lowrank,
            parsed.rms_norm_eps,
            use_combine=False,
            device="cuda",
        )

    @staticmethod
    def _src_key(name):
        if name.startswith("mtp_block."):
            return "mtp.layers.0." + name.removeprefix("mtp_block.")
        return "mtp." + name

    def forward(self, input_data, prev_hidden, input_ids, *, visual_inputs=None):
        embedded = self._parent[0].embed_input_ids(input_ids)
        if visual_inputs is not None:
            indices, visual = visual_inputs
            embedded.index_copy_(0, indices, visual.to(embedded.dtype))
        embedded = self.fc_embedding(self.pre_fc_norm_embedding(embedded))
        streams = self.pre_fc_norm_hidden(prev_hidden).unflatten(
            -1, (self.hc_count, self.hidden_size)
        )
        streams = (self.fc_hidden(streams) + embedded[:, None]).flatten(-2)
        return self.mtp_block(input_data, streams)

    def logits_from_hidden(self, hidden_states):
        hidden_states, _ = self.hyper_connection_mixer.mix(hidden_states)
        return self._head[0](hidden_states)


class Qwen4ExpForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """Native Flash-Next vision tower and hyper-connection text decoder."""

    supports_full_cuda_graph = True
    supports_piecewise_cuda_graph = True
    supports_mtp_verify_cuda_graph = True

    def __init__(self, config):
        config.text_config.checkpoint_path = getattr(config, "checkpoint_path", None)
        super().__init__(config, language_model_type=Qwen4ExpForCausalLM)
        lm = self.language_model
        self.ssm_cache_config = None if lm is None else lm.ssm_cache_config
        self.num_kv_layers = 0 if lm is None else lm.num_kv_layers
        self.num_ssm_layers = 0 if lm is None else lm.num_ssm_layers
        self.index_head_dim = 0 if lm is None else lm.index_head_dim
        self.requires_mrope_buffer = False if lm is None else lm.requires_mrope_buffer
        self.output_hidden_size = (
            config.text_config.hidden_size if lm is None else lm.output_hidden_size
        )

    @property
    def mtp(self):
        return getattr(self.language_model, "mtp", None)
