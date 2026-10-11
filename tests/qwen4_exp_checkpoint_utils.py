"""Header-only BF16 Flash-Next checkpoint audit and bounded PLE row lookup.

Auditing reads the safetensors headers, not the 360 GB tensor payload. The
schema deliberately covers the text backbone; vision and MTP are reported
separately and are not claimed as implemented by this initial adapter.
"""

import itertools
import json
import math
import struct
from pathlib import Path

from gllm.layers.position_learning import ngram_vocab_layout
from gllm.models.qwen4_exp import parse_text_config


def text_tensor_shapes(config):
    """Required native checkpoint names/shapes, before any TP or fusion."""
    c = config
    h, hyper = c.hidden_size, c.hyper_hidden_size
    shapes = {
        "model.language_model.embed_tokens.weight": (c.vocab_size, h),
        "lm_head.weight": (c.vocab_size, h),
    }

    def hc(prefix, inject=True):
        shapes[f"{prefix}.hc_norm.weight"] = (hyper,)
        shapes[f"{prefix}.input_mix_weight_down.weight"] = (c.hc_lowrank, hyper)
        shapes[f"{prefix}.input_mix_weight_up.weight"] = (hyper, c.hc_lowrank)
        if inject:
            shapes[f"{prefix}.block_inject_weight.weight"] = (c.hc_count, hyper)

    hc("model.language_model.hyper_connection_mixer", inject=False)
    for layer, kind in enumerate(c.layer_types):
        p = f"model.language_model.layers.{layer}"
        hc(f"{p}.attn_hyper_connection")
        hc(f"{p}.mlp_hyper_connection")
        shapes[f"{p}.mlp.gate.weight"] = (c.num_experts, h)
        # The official BF16 checkpoint stacks experts and does not append
        # .weight to these two tensor names. This is not the older per-expert
        # Qwen3.5 layout.
        shapes[f"{p}.mlp.experts.gate_up_proj"] = (
            c.num_experts,
            2 * c.moe_intermediate_size,
            h,
        )
        shapes[f"{p}.mlp.experts.down_proj"] = (
            c.num_experts,
            h,
            c.moe_intermediate_size,
        )
        if c.shared_expert_intermediate_size:
            shapes[f"{p}.mlp.shared_expert.gate_proj.weight"] = (
                c.shared_expert_intermediate_size,
                h,
            )
            shapes[f"{p}.mlp.shared_expert.up_proj.weight"] = (
                c.shared_expert_intermediate_size,
                h,
            )
            shapes[f"{p}.mlp.shared_expert.down_proj.weight"] = (
                h,
                c.shared_expert_intermediate_size,
            )
            shapes[f"{p}.mlp.shared_expert_gate.weight"] = (1, h)
        if kind == "linear_attention":
            key = c.linear_num_key_heads * c.linear_key_head_dim
            value = c.linear_num_value_heads * c.linear_value_head_dim
            attn = f"{p}.linear_attn"
            for name in ("A_log", "dt_bias"):
                shapes[f"{attn}.{name}"] = (c.linear_num_value_heads,)
            shapes[f"{attn}.conv1d.weight"] = (
                2 * key + value,
                1,
                c.linear_conv_kernel_dim,
            )
            shapes[f"{attn}.in_proj_qkv.weight"] = (2 * key + value, h)
            shapes[f"{attn}.in_proj_z.weight"] = (value, h)
            for name in ("a", "b"):
                shapes[f"{attn}.in_proj_{name}.weight"] = (c.linear_num_value_heads, h)
            shapes[f"{attn}.norm.weight"] = (c.linear_value_head_dim,)
            shapes[f"{attn}.out_proj.weight"] = (h, value)
        else:
            attn = f"{p}.self_attn"
            # q_proj has interleaved query/output-gate rows per head.
            shapes[f"{attn}.q_proj.weight"] = (
                2 * c.num_attention_heads * c.head_dim,
                h,
            )
            for name in ("k", "v"):
                shapes[f"{attn}.{name}_proj.weight"] = (
                    c.num_key_value_heads * c.head_dim,
                    h,
                )
            shapes[f"{attn}.o_proj.weight"] = (h, c.num_attention_heads * c.head_dim)
            for name in ("q", "k"):
                shapes[f"{attn}.{name}_norm.weight"] = (c.head_dim,)
                shapes[f"{attn}.indexer.{name}_layernorm.weight"] = (
                    c.indexer_head_dim,
                )
            shapes[f"{attn}.indexer.index_qk_proj.weight"] = (
                (c.indexer_n_heads + c.indexer_kv_heads) * c.indexer_head_dim,
                h,
            )
    for dense_id, layer in enumerate(sorted(c.ple_layer_indices)):
        p = f"model.language_model.layers.{layer}.ple"
        shapes[f"{p}.key_proj.weight"] = (hyper, c.ple_embed_dim)
        shapes[f"{p}.value_proj.weight"] = (h, c.ple_embed_dim)
        shapes[f"{p}.conv1d.weight"] = (hyper, 1, c.ple_conv_kernel_size)
        for name in ("norm_key", "norm_query", "norm_conv"):
            shapes[f"{p}.{name}.weight"] = (hyper,)
        p += ".ple_embedding"
        shapes[f"{p}.layer_multipliers"] = (c.ngram_size,)
        shapes[f"{p}.ngram_heads_vocab_sizes"] = (c.ngram_heads,)
        shapes[f"{p}.ngram_heads_offsets"] = (c.ngram_heads,)
        _, _, total = ngram_vocab_layout(
            c.ngram_vocab_size_base, c.ngram_heads, dense_id
        )
        divisor = c.make_ngram_vocab_size_divisible_by
        total = ((total + divisor - 1) // divisor) * divisor
        shard_size = (total + c.split_ngram_parts - 1) // c.split_ngram_parts
        for shard in range(c.split_ngram_parts):
            rows = max(0, min(shard_size, total - shard * shard_size))
            shapes[f"{p}.ngram_embedding.shard_{shard}.weight"] = (rows, c.ple_head_dim)
    return shapes


def read_safetensors_header(path):
    """Read and structurally validate tensor ranges without tensor allocation."""
    path = Path(path)
    with path.open("rb") as file:
        size_bytes = file.read(8)
        if len(size_bytes) != 8:
            raise ValueError(f"Truncated safetensors header: {path}")
        header_size = struct.unpack("<Q", size_bytes)[0]
        if header_size > 100_000_000 or header_size > path.stat().st_size - 8:
            raise ValueError(f"Invalid safetensors header length: {path}")
        header = json.loads(file.read(header_size))
    payload_bytes = path.stat().st_size - 8 - header_size
    widths = {"BF16": 2, "F16": 2, "F32": 4, "I64": 8}
    ranges = []
    for key, metadata in header.items():
        if key == "__metadata__":
            continue
        start, end = metadata["data_offsets"]
        if not 0 <= start <= end <= payload_bytes:
            raise ValueError(f"Truncated or invalid tensor payload: {key} in {path}")
        shape = metadata["shape"]
        if any(dim < 0 for dim in shape):
            raise ValueError(f"Invalid tensor shape: {key}")
        width = widths.get(metadata["dtype"])
        if width is not None and math.prod(shape) * width != end - start:
            raise ValueError(f"Tensor byte length does not match shape: {key}")
        ranges.append((start, end))
    ranges.sort()
    if any(
        end > next_start for (_, end), (next_start, _) in itertools.pairwise(ranges)
    ):
        raise ValueError(f"Overlapping tensor payload ranges: {path}")
    return header


def audit_bf16_checkpoint(path):
    """Fail on missing shards, truncated payloads, absent keys or wrong shapes."""
    path = Path(path)
    raw = json.loads((path / "config.json").read_text())
    if raw.get("quantization_config") or raw.get("text_config", {}).get(
        "quantization_config"
    ):
        raise ValueError("Initial Qwen4Exp checkpoint audit expects unquantized BF16")
    config = parse_text_config(raw)
    index = json.loads((path / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    expected = text_tensor_shapes(config)
    missing = expected.keys() - index.keys()
    if missing:
        raise ValueError(
            f"Missing {len(missing)} backbone tensors: {sorted(missing)[:5]}"
        )
    unexpected = {
        k for k in index if k.startswith("model.language_model.")
    } - expected.keys()
    if unexpected:
        raise ValueError(f"Unmapped backbone tensors: {sorted(unexpected)[:5]}")
    # Check one file at a time: never retain all headers or tensor payloads.
    by_shard = {}
    for key, shard in index.items():
        by_shard.setdefault(shard, []).append(key)
    size = 0
    for shard, keys in by_shard.items():
        file = path / shard
        header = read_safetensors_header(file)
        size += file.stat().st_size
        for key in keys:
            if key not in header:
                raise ValueError(
                    f"Index maps {key} to a shard that does not contain it"
                )
            if key not in expected:
                continue
            metadata = header[key]
            if tuple(metadata["shape"]) != expected[key]:
                raise ValueError(
                    f"Wrong shape for {key}: {metadata['shape']} != {expected[key]}"
                )
            dtype = (
                "I64"
                if key.endswith(
                    (
                        "layer_multipliers",
                        "ngram_heads_vocab_sizes",
                        "ngram_heads_offsets",
                    )
                )
                else "BF16"
            )
            if metadata["dtype"] != dtype:
                raise ValueError(f"Wrong dtype for {key}: expected {dtype}")
    return {
        "shards": len(by_shard),
        "bytes": size,
        "backbone_tensors": len(expected),
        "gdn_layers": len(config.gdn_layer_ids),
        "qsa_layers": len(config.qsa_layer_ids),
        "ple_layer_indices": config.ple_layer_indices,
        "vision_tensors": sum(k.startswith("model.visual.") for k in index),
        "mtp_tensors": sum(k.startswith("mtp.") for k in index),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Audit a BF16 Qwen3.8-Flash-Next checkpoint"
    )
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_bf16_checkpoint(args.checkpoint), indent=2))
