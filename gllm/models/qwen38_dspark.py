"""Qwen3.8 DSpark draft model: a standalone DFlash backbone plus DSpark heads.

The published drafter (``RadixArk/Qwen3.8-27B-DSpark``) is a *standalone*
DSpark checkpoint rather than a ``mtp.*`` appendix to its target, so nothing
here borrows a parent model's rule table the way DeepSeek-V4's adapter does.
Its checkpoint keys are already the names its module tree exposes: there is no
``model.`` prefix and no ``mtp.N.`` remap to undo, which is why
:meth:`Qwen38DSparkMapping.checkpoint_key` is the identity.

Three things differ from the DeepSeek-V4 drafter and are easy to get subtly
wrong:

* **The draft block is dual-source.**  ``q`` comes from the noisy block alone
  while ``k``/``v`` come from ``cat([target context, noisy block])``, so a
  fused ``qkv_proj`` cannot express it and the projections stay separate.
* **``block_size`` is counted differently.**  The checkpoint's ``block_size``
  is the width of the noisy block the verifier consumes (anchor + proposals);
  the shared contract's counts proposals only.  The two differ by one, and
  :class:`Qwen38DSpark` reconciles them.
* **The context cache is stateful.**  :meth:`Qwen38DSpark.forward_draft`
  appends this step's accepted context K/V to :class:`Qwen38DSparkCache` in
  place, so a draft loop keeps one cache and never re-projects the prefix.

Reference: SpecForge's ``specforge/modeling/draft/{dflash,dspark}.py``, the code
that trained this checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn

from gllm.layers.layernorm import RMSNorm
from gllm.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from gllm.layers.rotary_embedding import YaRNScalingRotaryEmbedding
from gllm.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from gllm.models.dspark_protocol import (
    DSparkCheckpointMapping,
    DSparkForwardProtocol,
)
from gllm.models.qwen2 import Qwen2MLP
from gllm.models.weight_loader import (
    LoadContext,
    WeightRule,
    contains,
    h_gate_up,
    h_proj_dim0,
    h_proj_dim1,
    run_weight_loader,
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


@dataclass
class Qwen38DSparkCache:
    """Per-layer target-context K/V, **advanced in place** by ``forward_draft``.

    ``keys[layer]`` / ``values[layer]`` are ``[B, num_kv_heads, S, head_dim]``
    and hold *target context only*: the noisy block is attended against them
    and then discarded, exactly as SpecForge's ``crop(start)`` does after each
    draft step.

    Both routines apply their history to the same object:
    :meth:`Qwen38DSpark.prefill` seeds it from the prompt, and
    :meth:`Qwen38DSpark.forward_draft` appends the newly accepted rows.  A
    caller therefore passes one cache across a whole draft loop and never has
    to re-project the prefix.
    """

    keys: list[torch.Tensor]
    values: list[torch.Tensor]

    def context_length(self) -> int:
        return self.keys[0].shape[-2]


class Qwen38DSparkMarkovHead(nn.Module):
    """Low-rank previous-token logit bias (SpecForge ``VanillaMarkovHead``).

    ``markov_w2`` is stored ``[vocab, rank]`` in the checkpoint, i.e. a
    ``Linear(rank -> vocab)``, which is what ``ParallelLMHead`` exposes.
    """

    def __init__(self, vocab_size: int, markov_rank: int) -> None:
        super().__init__()
        self.markov_rank = markov_rank
        self.markov_w1 = VocabParallelEmbedding(vocab_size, markov_rank)
        self.markov_w2 = ParallelLMHead(vocab_size, markov_rank)

    def step_bias(self, prev_token_ids: torch.Tensor) -> torch.Tensor:
        """``[B]`` token ids -> ``[B, vocab]`` additive logit bias."""
        return self.markov_w2(self.markov_w1(prev_token_ids))


class Qwen38DSparkConfidenceHead(nn.Module):
    """Per-position acceptance predictor; a single bias-included linear."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.proj = ReplicatedLinear(input_dim, 1, bias=True)


class Qwen38DSparkAttention(nn.Module):
    """Dual-source GQA: ``q`` from the draft block, ``k``/``v`` from context too.

    Mirrors SpecForge's ``Qwen3DFlashAttention._compute_qkv``: the keys are
    ordered *context first, then draft*, ``k_norm`` sees both halves, and RoPE
    is applied to ``q`` at the draft positions while ``k`` gets the full
    context-plus-draft position range.
    """

    def __init__(self, config: Qwen38DSparkConfig) -> None:
        super().__init__()
        tp_size = _tp_size()
        if config.num_attention_heads % tp_size:
            raise ValueError(
                f"num_attention_heads ({config.num_attention_heads}) must be "
                f"divisible by TP size ({tp_size})"
            )
        self.total_num_heads = config.num_attention_heads
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5

        # Separate projections on purpose: ``q`` reads the draft block while
        # ``k``/``v`` read context ++ draft, so they cannot share an input.
        self.q_proj = ColumnParallelLinear(
            config.hidden_size,
            self.total_num_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = ColumnParallelLinear(
            config.hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = ColumnParallelLinear(
            config.hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

    def _project_kv(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``[B, S, H]`` -> normalised, unrotated ``k``/``v`` ``[B, kvh, S, d]``."""
        batch, length = x.shape[:-1]
        k = self.k_proj(x).view(batch, length, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, length, self.num_kv_heads, self.head_dim)
        return self.k_norm(k).transpose(1, 2), v.transpose(1, 2)

    def prefill_context(
        self,
        context: torch.Tensor,
        positions: torch.Tensor,
        rotary_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project the prompt's context once and hand it back ready to cache."""
        k, v = self._project_kv(context)
        return _rotate(k, rotary_cache, positions), v

    def forward(
        self,
        hidden_states: torch.Tensor,
        cached_k: torch.Tensor,
        cached_v: torch.Tensor,
        noise_positions: torch.Tensor,
        rotary_cache: torch.Tensor,
    ) -> torch.Tensor:
        """Attend the noisy block against the history plus itself.

        ``cached_k`` / ``cached_v`` already carry this step's accepted context
        rows -- :meth:`Qwen38DSpark.forward_draft` appends them before calling
        here -- so only the noisy rows are new.  Those are rotated at their own
        positions and dropped afterwards: nothing from the draft block is ever
        written back.
        """
        batch, query_len = hidden_states.shape[:-1]
        q = self.q_proj(hidden_states).view(
            batch, query_len, self.num_heads, self.head_dim
        )
        q = _rotate(
            self.q_norm(q).transpose(1, 2), rotary_cache, noise_positions
        )

        k_noise, v_noise = self._project_kv(hidden_states)
        k_noise = _rotate(k_noise, rotary_cache, noise_positions)

        k = torch.cat([cached_k, k_noise], dim=2)
        v = torch.cat([cached_v, v_noise], dim=2)
        k = _repeat_kv(k, self.num_key_value_groups)
        v = _repeat_kv(v, self.num_key_value_groups)
        # No mask: the reference offline path passes ``attention_mask=None``
        # with ``is_causal=False``, so the block sees everything.
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=self.scaling
        )
        out = out.transpose(1, 2).reshape(batch, query_len, -1)
        return self.o_proj(out)


class Qwen38DSparkDecoderLayer(nn.Module):
    """Pre-norm residual block: dual-source attention then a SwiGLU MLP."""

    def __init__(self, config: Qwen38DSparkConfig) -> None:
        super().__init__()
        self.self_attn = Qwen38DSparkAttention(config)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cached_k: torch.Tensor,
        cached_v: torch.Tensor,
        noise_positions: torch.Tensor,
        rotary_cache: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states),
            cached_k,
            cached_v,
            noise_positions,
            rotary_cache,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        return residual + self.mlp(self.post_attention_layernorm(hidden_states))


class Qwen38DSpark(nn.Module, DSparkForwardProtocol):
    """The Qwen3.8 draft network, driven through the shared DSpark contract."""

    def __init__(
        self,
        config: Qwen38DSparkConfig,
        *,
        embed: nn.Module,
        lm_head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        # The checkpoint's ``block_size`` counts the width the verifier
        # consumes (anchor + proposals); the contract counts proposals.  The
        # heads below read ``block_size`` rows starting at the anchor's own
        # position, so one of them is the anchor and only the rest are drafted.
        self.block_size = config.block_size - 1
        self.noise_width = config.block_size
        self.mask_token_id = config.mask_token_id

        # The drafter ships no vocabulary: both of these belong to the target
        # and are held in lists so they are not registered a second time.
        self._embed = [embed]
        self._lm_head = [lm_head]

        self.fc = ReplicatedLinear(
            config.target_hidden_size, config.hidden_size, bias=False
        )
        self.hidden_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.layers = nn.ModuleList(
            Qwen38DSparkDecoderLayer(config)
            for _ in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.markov_head = Qwen38DSparkMarkovHead(
            config.vocab_size, config.markov_rank
        )
        self.confidence_head = Qwen38DSparkConfidenceHead(
            config.hidden_size + config.markov_rank
        )
        self.rotary_emb = _build_rope(config)

    # -- internals ---------------------------------------------------------

    def _project_context(self, main_hidden: torch.Tensor) -> torch.Tensor:
        """``[B, S, taps * H]`` -> ``[B, S, H]`` (``hidden_norm(fc(...))``)."""
        return self.hidden_norm(self.fc(main_hidden))

    def _embed_noise(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Anchor token ids -> the ``noise_width``-row embedding block."""
        anchor = _anchor_ids(input_ids)
        block = anchor.new_full(
            (anchor.shape[0], self.noise_width), self.mask_token_id
        )
        block[:, 0] = anchor
        return self._embed[0](block)

    # -- DSparkForwardProtocol --------------------------------------------

    @torch.no_grad()
    def prefill(
        self,
        main_hidden: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> Qwen38DSparkCache:
        """Project the prompt and cache each layer's rotated context K/V."""
        context = self._project_context(main_hidden)
        positions = torch.arange(context.shape[1], device=context.device)
        keys, values = [], []
        for layer in self.layers:
            k, v = layer.self_attn.prefill_context(
                context, positions, self.rotary_emb.cos_sin_cache
            )
            keys.append(k)
            values.append(v)
        return Qwen38DSparkCache(keys=keys, values=values)

    def forward_draft(
        self,
        main_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        start_pos: int,
        caches: Qwen38DSparkCache,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the noisy block and return ``(hidden, draft_ids)``.

        ``main_hidden`` carries only the *newly accepted* context rows, and this
        call **advances ``caches`` in place**: each layer's rotated K/V for
        those rows is appended before the noisy block is attended, so the next
        call sees the grown history without re-projecting the prefix.

        ``start_pos`` is the position of the first noisy row (the anchor, i.e.
        the checkpoint's ``start``); the new context rows are taken to occupy
        ``[start_pos - len(main_hidden), start_pos)``.  The caller therefore
        has to satisfy ``start_pos - len(main_hidden) == caches.context_length()``
        -- passing anything else silently overlaps or skips positions.
        Documentation only: this method states the convention, it does not
        enforce it.

        Only context is appended.  The noisy block is attended against the
        history and then dropped, which is the net effect of SpecForge's
        ``past_key_values.update(...)`` followed by ``crop(start)``.

        Documentation only -- the signature is unchanged by the note below.

        ``draft_ids`` is returned because :class:`DSparkForwardProtocol`
        declares it, not because anything consumes it: nothing in the tree
        reads it, and it is exactly ``[anchor, mask_token_id * (noise_width -
        1)]`` -- reconstructible from the anchor the caller already passed and
        a checkpoint constant, so it carries no information.  SpecForge goes
        the other way, handing the noise *in* as ``noise_embedding`` and
        returning only hidden states.  Worth dropping from the shared
        signature when the protocol is next revised, at which point the noisy
        block becomes a caller-supplied input instead.
        """
        context = self._project_context(main_hidden)
        noise = self._embed_noise(input_ids)
        anchor = _anchor_ids(input_ids)
        draft_ids = anchor.new_full(
            (anchor.shape[0], self.noise_width), self.mask_token_id
        )
        draft_ids[:, 0] = anchor

        # The block sits *after* the context it was given: the new rows occupy
        # ``[start_pos - len(context), start_pos)`` and the noisy rows follow.
        context_positions = torch.arange(
            start_pos - context.shape[1], start_pos, device=noise.device
        )
        noise_positions = torch.arange(
            start_pos, start_pos + self.noise_width, device=noise.device
        )
        rotary_cache = self.rotary_emb.cos_sin_cache

        hidden = noise
        for index, layer in enumerate(self.layers):
            new_k, new_v = layer.self_attn.prefill_context(
                context, context_positions, rotary_cache
            )
            cached_k = torch.cat([caches.keys[index], new_k], dim=2)
            cached_v = torch.cat([caches.values[index], new_v], dim=2)
            caches.keys[index] = cached_k
            caches.values[index] = cached_v
            hidden = layer(
                hidden, cached_k, cached_v, noise_positions, rotary_cache
            )
        return self.norm(hidden), draft_ids

    def forward_head(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        sample: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """LM head + Markov correction + confidence, per draft position."""
        if self._lm_head[0] is None:
            raise ValueError("Qwen38DSpark.forward_head needs the target LM head")
        # Row 0 sits at the anchor's own position and predicts nothing; the
        # reference drops it (``draft_hidden[:, -block_size + 1:]``).
        proposal_hidden = hidden_states[:, 1:, :]
        # ``forward_draft`` already applied ``self.norm``; the reference hands
        # that same tensor straight to the target LM head.
        logits = self._lm_head[0](proposal_hidden)
        if sample is None:
            sample = lambda x: x.argmax(dim=-1)

        anchor = _anchor_ids(input_ids)
        output_ids = anchor.new_empty((anchor.shape[0], self.block_size + 1))
        output_ids[:, 0] = anchor
        prev = anchor
        markov_embeds = []
        for position in range(self.block_size):
            markov_embed = self.markov_head.markov_w1(prev)
            logits[:, position].add_(self.markov_head.markov_w2(markov_embed))
            markov_embeds.append(markov_embed)
            output_ids[:, position + 1] = sample(logits[:, position])
            prev = output_ids[:, position + 1]

        markov_hidden = torch.stack(markov_embeds, dim=1)
        confidence = self.confidence_head.proj(
            torch.cat([proposal_hidden, markov_hidden], dim=-1)
        ).squeeze(-1)
        return output_ids, logits, confidence

    @torch.no_grad()
    def load_weights(self, weights, mp_load_progress=None) -> None:
        """Load the standalone draft checkpoint through the shared mapping."""
        mapping = Qwen38DSparkMapping(self)
        run_weight_loader(
            self,
            weights,
            mapping.weight_rules(),
            mp_load_progress,
            pp_idx_offset=1,
            start_layer=0,
            ctx=mapping.load_context(weights),
            src_key_fn=mapping.checkpoint_key,
        )


def _tp_size() -> int:
    from gllm.distributed.parallel_state import get_tp_size

    return get_tp_size()


def _rope_cache_length(config: Qwen38DSparkConfig) -> int:
    """Pre-scaling length to hand YaRN so its cache spans the serving window.

    ``YaRNScalingRotaryEmbedding`` builds ``max_position_embeddings *
    scaling_factor`` cache rows, so this argument must be the length *before*
    scaling.  Passing the already-extended ``config.max_position_embeddings``
    multiplies the window a second time: at the published 262144 x 32 that asks
    for 8.4M rows (~4 GiB) where the serving window needs 262144 (~128 MiB).
    DeepSeek-V3.2 avoids the same trap by passing
    ``original_max_position_embeddings``.
    """
    scaling = config.rope_scaling or {}
    original = scaling.get("original_max_position_embeddings")
    if original is not None:
        return int(original)
    # No pre-scaling length in the config: undo the factor instead of applying
    # it, so the cache still lands on the serving window.
    factor = float(scaling.get("factor", 1.0))
    return max(int(config.max_position_embeddings / max(factor, 1.0)), 1)


def _build_rope(config: Qwen38DSparkConfig) -> YaRNScalingRotaryEmbedding:
    """Qwen's generic YaRN: one ``mscale`` term, no DeepSeek ratio form."""
    scaling = config.rope_scaling or {}
    return YaRNScalingRotaryEmbedding(
        config.head_dim,
        config.head_dim,
        _rope_cache_length(config),
        config.rope_theta,
        True,  # neox / half-split, matching HF's ``rotate_half``
        float(scaling.get("factor", 1.0)),
        beta_fast=int(scaling.get("beta_fast", 32)),
        beta_slow=int(scaling.get("beta_slow", 1)),
    )


def _cos_sin(
    cache: torch.Tensor, positions: torch.Tensor, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Index a rotary cache into ``(cos, sin)`` shaped ``[1, 1, S, D]``.

    gLLM's ``RotaryEmbedding`` applies RoPE through a fused op that takes one
    position tensor for *both* q and k.  The dual-source block cannot use it:
    ``q`` sits on the draft positions only while ``k`` spans context ++ draft.
    Reading the same cache the op would (YaRN's ``mscale`` is baked into it)
    and rotating the two with different slices is equivalent.

    Positions can exceed ``max_position_embeddings``; the YaRN cache is built
    for ``max_position_embeddings * scaling_factor`` entries to cover that.
    """
    freqs = cache[positions].to(dtype)
    cos_half, sin_half = freqs.chunk(2, dim=-1)
    # gLLM stores ``cat(cos, sin)`` with each half only ``head_dim // 2`` wide
    # (the fused op expands them internally).  ``_rotate`` uses HF's
    # ``rotate_half`` form, which wants both at full width -- duplicating each
    # half is the same rotation, written the other way round.
    cos = torch.cat([cos_half, cos_half], dim=-1)
    sin = torch.cat([sin_half, sin_half], dim=-1)
    return cos[None, None], sin[None, None]


def _rotate(
    x: torch.Tensor, cache: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """Apply RoPE to ``x`` at ``positions``, reading gLLM's rotary cache.

    SpecForge rotates ``cat([context, draft])`` in one call and lets ``q`` take
    the tail slice; rotating each source with its own positions is the same
    rotation and is what the dual-source layout needs.
    """
    cos, sin = _cos_sin(cache, positions, x.dtype)
    return x * cos + _rotate_half(x) * sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Half-split rotation, matching HF's ``rotate_half`` / neox style."""
    first, second = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


def _repeat_kv(x: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return x
    return x.repeat_interleave(groups, dim=1)


def _anchor_ids(input_ids: torch.Tensor) -> torch.Tensor:
    """Accept ``[B, 1]`` or ``[B]`` and return ``[B]``."""
    if input_ids.ndim == 2 and input_ids.shape[1] == 1:
        return input_ids[:, 0]
    if input_ids.ndim != 1:
        raise ValueError("Qwen3.8 DSpark anchor ids must have shape [B] or [B,1]")
    return input_ids


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

__all__ = [
    "Qwen38DSpark",
    "Qwen38DSparkCache",
    "Qwen38DSparkConfig",
    "Qwen38DSparkMapping",
]
