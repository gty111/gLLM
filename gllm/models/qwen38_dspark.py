"""Qwen3.8 DSpark draft adapter: config and checkpoint mapping.

The published drafter (``RadixArk/Qwen3.8-27B-DSpark``) is a *standalone*
DSpark checkpoint rather than a ``mtp.*`` appendix to its target, so nothing
here borrows a parent model's rule table the way DeepSeek-V4's adapter does.
Its checkpoint keys are already the names its module tree will expose: there is
no ``model.`` prefix and no ``mtp.N.`` remap to undo, which is why
:meth:`Qwen38DSparkMapping.checkpoint_key` is the identity.

This module carries the half of the adapter that needs no device -- the config
reader and the weight mapping.  The module tree and the forward path are a
separate change; the rule table below is what pins the parameter names that
tree has to use.

Reference: SpecForge's ``specforge/modeling/draft/{dflash,dspark}.py``, the
code that trained this checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gllm.models.dspark_protocol import DSparkCheckpointMapping
from gllm.models.weight_loader import (
    LoadContext,
    WeightRule,
    contains,
    h_gate_up,
    h_proj_dim0,
    h_proj_dim1,
)
from gllm.models.weight_utils import get_tensor_from_dict


@dataclass(frozen=True)
class Qwen38DSparkConfig:
    """The slice of the checkpoint's ``config.json`` this adapter reads.

    Only fields the drafter cannot infer are carried.  The DSpark-specific ones
    (``block_size`` / ``target_layer_ids`` / ``mask_token_id`` / ``markov_rank``)
    are mirrored both at the top level and inside a ``dspark_config`` block;
    :meth:`from_dict` prefers the nested block and falls back to the top level.
    """

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    num_hidden_layers: int
    rms_norm_eps: float
    vocab_size: int
    block_size: int
    target_layer_ids: tuple[int, ...]
    mask_token_id: int
    markov_rank: int
    rope_theta: float
    max_position_embeddings: int
    rope_scaling: dict[str, Any] | None = None

    @property
    def target_hidden_size(self) -> int:
        """Width of the concatenated target-hidden taps ``fc`` consumes."""
        return len(self.target_layer_ids) * self.hidden_size

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Qwen38DSparkConfig":
        """Read the published ``config.json`` layout.

        The nested DSpark block is the authoritative copy of the drafter fields,
        but it is absent from some derived configs, so every lookup falls back
        to the top level.
        """
        dspark = raw.get("dspark_config") or raw.get("dflash_config") or {}

        def pick(name: str, default: Any = None) -> Any:
            if name in dspark:
                return dspark[name]
            return raw.get(name, default)

        target_layer_ids = pick("target_layer_ids")
        if target_layer_ids is None:
            num_target_layers = raw.get("num_target_layers")
            if num_target_layers is None:
                raise ValueError(
                    "Qwen3.8 DSpark config needs target_layer_ids or "
                    "num_target_layers to know which target taps to read"
                )
            target_layer_ids = _spread_target_layers(
                num_target_layers, raw["num_hidden_layers"]
            )

        mask_token_id = pick("mask_token_id")
        if mask_token_id is None:
            raise ValueError(
                "Qwen3.8 DSpark config needs mask_token_id: it is the noise "
                "token the draft block is filled with"
            )

        return cls(
            hidden_size=raw["hidden_size"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw["num_key_value_heads"],
            head_dim=raw.get(
                "head_dim", raw["hidden_size"] // raw["num_attention_heads"]
            ),
            intermediate_size=raw["intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            rms_norm_eps=raw["rms_norm_eps"],
            vocab_size=raw["vocab_size"],
            block_size=raw["block_size"],
            target_layer_ids=tuple(int(i) for i in target_layer_ids),
            mask_token_id=int(mask_token_id),
            markov_rank=int(pick("markov_rank", 0)),
            rope_theta=float(raw.get("rope_theta", 10000.0)),
            max_position_embeddings=int(raw.get("max_position_embeddings", 8192)),
            rope_scaling=raw.get("rope_scaling"),
        )



def _spread_target_layers(num_target_layers: int, num_draft_layers: int) -> list[int]:
    """Evenly spaced target taps, matching SpecForge's own default.

    Used only when a config omits ``target_layer_ids``; the published
    checkpoint spells the list out (``[5, 19, 33, 47, 61]``), so this is a
    safety net rather than the normal path.
    """
    if num_draft_layers <= 1:
        return [num_target_layers // 2]
    start, end = 1, num_target_layers - 3
    span = end - start
    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]



class Qwen38DSparkMapping(DSparkCheckpointMapping):
    """Maps the drafter onto its standalone ``model.safetensors`` layout.

    ``checkpoint_key`` is the identity: the published keys are exactly the
    module paths this adapter exposes.  That is worth stating explicitly,
    because the sibling DeepSeek-V4 adapter needs a substantial remap
    (``blocks.*`` -> ``mtp.0/1/2.*``, ``main_*`` -> ``mtp.0.*``); sharing one
    contract with an identity mapping is the point of the abstraction.

    ``q_proj`` / ``k_proj`` / ``v_proj`` stay separate rather than fused into
    ``qkv_proj`` because the dual-source attention projects them from different
    tensors.  Only the MLP fuses, and ``h_gate_up`` splits the checkpoint's
    ``gate_proj`` / ``up_proj`` back into it.
    """

    def __init__(self, draft: Any = None) -> None:
        # Only ``load_context`` needs the module (it reads head geometry and the
        # Markov tables' shard indices).  ``checkpoint_key`` and
        # ``weight_rules`` are pure, which keeps them testable on a machine
        # that cannot build the CUDA-resident module tree.
        self.draft = draft

    def checkpoint_key(self, param_name: str) -> str:
        return param_name

    def weight_rules(self) -> list[WeightRule]:
        return [
            # -- attention / MLP projections ------------------------------
            WeightRule(contains("self_attn.q_proj"), h_proj_dim0, "q_proj"),
            WeightRule(contains("self_attn.k_proj"), h_proj_dim0, "k_proj"),
            WeightRule(contains("self_attn.v_proj"), h_proj_dim0, "v_proj"),
            WeightRule(contains("self_attn.o_proj"), h_proj_dim1, "o_proj"),
            WeightRule(contains("mlp.gate_up_proj"), h_gate_up, "gate_up_proj"),
            WeightRule(contains("mlp.down_proj"), h_proj_dim1, "down_proj"),
            # -- Markov tables: vocab-parallel, one shard per rank ---------
            WeightRule(contains("markov_w"), _h_markov_vocab_shard, "markov"),
            # -- everything else is replicated (fc, norms, confidence) -----
            WeightRule(lambda _: True, _h_replicated, "replicated"),
        ]

    def load_context(self, weights: dict[str, Any]) -> LoadContext | None:
        """Head geometry for the projections, plus the Markov shard maps.

        The column/row handlers need ``num_heads`` / ``num_kv_heads`` /
        ``head_dim``, and the Markov handler needs each table's vocabulary
        shard -- neither is derivable from a bare key, which is what this hook
        exists for.  Both are read off the draft module, so this only returns a
        real context once the module tree (a separate change) exists.
        """
        if self.draft is None:
            return None
        attn = self.draft.layers[0].self_attn
        return LoadContext(
            weights=weights,
            num_heads=attn.num_heads,
            num_kv_heads=attn.num_kv_heads,
            head_dim=attn.head_dim,
            extra={
                "vocab_shards": {
                    "markov_head.markov_w1.weight": (
                        self.draft.markov_head.markov_w1.shard_indices
                    ),
                    "markov_head.markov_w2.weight": (
                        self.draft.markov_head.markov_w2.shard_indices
                    ),
                }
            },
        )


def _h_replicated(ctx: LoadContext, key: str, param: Any) -> None:
    """Whole tensor on every rank (``fc``, the norms, the confidence head)."""
    param.copy_(get_tensor_from_dict(ctx.weights, key))


def _h_markov_vocab_shard(ctx: LoadContext, key: str, param: Any) -> None:
    """Copy this rank's vocabulary window of a Markov table.

    Both tables are ``[vocab_size, markov_rank]`` -- 248320 x 256 at BF16, i.e.
    ~127 MB each -- so sharding the vocabulary axis keeps them off every rank
    in full.  The padded tail past ``num_org_elements`` stays zero so a padded
    id can never alias a real row.
    """
    shard = ctx.extra["vocab_shards"][key]
    param.zero_()
    weight = get_tensor_from_dict(ctx.weights, key)
    param[: shard.num_org_elements].copy_(
        weight[shard.org_vocab_start_index : shard.org_vocab_end_index].to(
            param.dtype
        )
    )

__all__ = ["Qwen38DSparkConfig", "Qwen38DSparkMapping"]
