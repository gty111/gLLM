"""Qwen3.8 DSpark draft model: a standalone DFlash backbone plus DSpark heads.

The published drafter (``RadixArk/Qwen3.8-27B-DSpark``) is a *standalone*
DSpark checkpoint rather than a ``mtp.*`` appendix to its target, so nothing
here borrows a parent model's rule table the way DeepSeek-V4's adapter does.
Its checkpoint keys are already the names its module tree exposes: there is no
``model.`` prefix and no ``mtp.N.`` remap to undo, which is why
:meth:`Qwen38DSparkMapping.checkpoint_key` is the identity.

Three details of this drafter are easy to get subtly wrong:

* **The draft block is dual-source.**  ``q`` comes from the noisy block alone
  while ``k``/``v`` come from ``cat([target context, noisy block])``, so a
  fused ``qkv_proj`` cannot express it and the projections stay separate.
* **``block_size`` counts proposals.**  Each noisy row predicts one next
  token, including the row whose input is the anchor.  The shared contract
  uses the same proposal count; verification consumes the anchor plus those
  proposals, so its width is one larger than the noisy block.
* **The context cache is stateful.**  :meth:`Qwen38DSpark.forward_draft`
  appends this step's accepted context K/V to :class:`Qwen38DSparkCache` in
  place, so a draft loop keeps one cache and never re-projects the prefix.

Reference: SpecForge's ``specforge/modeling/draft/{dflash,dspark}.py``, the code
that trained this checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

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


if TYPE_CHECKING:
    from gllm.runtime.memory_manager import DSparkRequestCache, Segment


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


@dataclass
class Qwen38DSparkPagedCache:
    """A view of one request's arena storage; allocation stays with the runtime."""

    segment: Segment
    state: DSparkRequestCache

    def context_length(self) -> int:
        return self.state.context_length


@dataclass
class _Qwen38DSparkBatchTable:
    """Reusable GPU metadata, not ownership of requests or arena slots.

    Each history request contributes noise_width equal table rows to the
    existing decode kernel. Sources stay request-owned; stable bindings copy
    only newly visible pages D2D. A changed binding/buffer refreshes its prefix.
    Access is serialized on the consuming stream, as for DSparkRequestCache.
    """

    noise_width: int
    table: torch.Tensor | None = None
    sources: tuple = ()

    def prepare(
        self,
        caches: list[Qwen38DSparkPagedCache],
        lengths: tuple[int, ...],
    ) -> torch.Tensor:
        device = caches[0].segment.k_cache[0].device
        current = []
        for cache, length in zip(caches, lengths):
            if length:
                source = cache.state.get_device_page_table(device)
                pages = (length + cache.segment.page_size - 1) // cache.segment.page_size
                current.append((cache.state, source, pages))
        if not current:
            return torch.empty((0, 0), device=device, dtype=torch.int32)
        rows = len(current) * self.noise_width
        pages = max(record[2] for record in current)
        grown = (
            self.table is None or self.table.device != device
            or self.table.shape[0] < rows or self.table.shape[1] < pages
        )
        if grown:
            old = self.table
            old_rows, old_pages = (0, 0) if old is None else old.shape
            # Growing history must not also double the request capacity.
            row_capacity = max(rows, 2 * old_rows) if rows > old_rows else old_rows
            page_capacity = max(pages, 2 * old_pages) if pages > old_pages else old_pages
            self.table = torch.empty(
                (row_capacity, page_capacity), device=device, dtype=torch.int32,
            )
        for index, (state, source, count) in enumerate(current):
            previous = self.sources[index] if index < len(self.sources) else None
            unchanged = (
                not grown and previous is not None and previous[0] is state
                and previous[1].data_ptr() == source.data_ptr()
                and previous[2] <= count
            )
            begin = previous[2] if unchanged else 0
            if begin < count:
                self.table[
                    index * self.noise_width:(index + 1) * self.noise_width, begin:count
                ].copy_(
                    source[:, begin:count].expand(self.noise_width, -1), non_blocking=True
                )
        # Retain source buffers: reset/reuse cannot masquerade as the previous
        # binding, and an enqueued D2D copy cannot lose its source allocation.
        self.sources = tuple(current)
        return self.table[:rows, :pages]


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
        *,
        page_table: torch.Tensor | None = None,
        context_length: int | tuple[int, ...] = 0,
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

        if page_table is not None:
            out = self._paged_attention(
                q, k_noise, v_noise, cached_k, cached_v, page_table, context_length
            )
        else:
            k = torch.cat([cached_k, k_noise], dim=2)
            v = torch.cat([cached_v, v_noise], dim=2)
            k = _repeat_kv(k, self.num_key_value_groups)
            v = _repeat_kv(v, self.num_key_value_groups)
            # The dense validation path also sees all history and noise.
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, scale=self.scaling
            )
        out = out.transpose(1, 2).reshape(batch, query_len, -1)
        return self.o_proj(out)

    def _paged_attention(
        self,
        q: torch.Tensor,
        k_noise: torch.Tensor,
        v_noise: torch.Tensor,
        cached_k: torch.Tensor,
        cached_v: torch.Tensor,
        page_table: torch.Tensor,
        context_length: int | tuple[int, ...],
    ) -> torch.Tensor:
        """Merge paged history and noncausal noise without sharing histories.

        The single-request table keeps its zero row stride. A batched table
        already has noise_width rows per nonempty request, in query order.
        Empty histories use noise only and never enter the paged reducer.
        """
        batch, _, query_len, _ = q.shape
        lengths = (context_length,) if isinstance(context_length, int) else context_length
        if len(lengths) != batch or any(length < 0 for length in lengths):
            raise ValueError("paged attention needs one valid length per request")
        active = [index for index, length in enumerate(lengths) if length]
        noise_k = _repeat_kv(k_noise, self.num_key_value_groups)
        noise_v = _repeat_kv(v_noise, self.num_key_value_groups)
        scores = torch.matmul(q.float(), noise_k.float().transpose(-1, -2))
        scores = scores * self.scaling
        noise_output = torch.matmul(scores.softmax(-1), noise_v.float()).to(q.dtype)
        if not active:
            return noise_output

        from gllm.layers.ops.merge_attn_states import merge_attn_states
        from gllm.layers.ops.triton_decode_attention import decode_attention_fwd

        def pack_active(tensor):
            if len(active) == batch:
                return tensor.reshape(batch * query_len, *tensor.shape[2:]).contiguous()
            return torch.cat([tensor[index] for index in active], dim=0).contiguous()

        query = pack_active(q.transpose(1, 2))
        history_output = torch.empty_like(query)
        history_lse = torch.empty(
            query.shape[0], self.num_heads, device=q.device, dtype=torch.float32
        )
        block_table = (
            page_table.expand(query_len, -1) if isinstance(context_length, int) else page_table
        )
        seq_lens = torch.cat([
            torch.full((query_len,), lengths[index], device=q.device, dtype=torch.int32)
            for index in active
        ])
        num_splits = 4
        scratch = torch.empty(
            query.shape[0], self.num_heads, num_splits, self.head_dim + 1,
            device=q.device, dtype=torch.float32,
        )
        decode_attention_fwd(
            query, cached_k, cached_v, history_output, history_lse,
            block_table, seq_lens, scratch, num_splits, self.scaling,
            cached_k.shape[1],
        )
        noise_lse = pack_active(scores.logsumexp(-1).transpose(1, 2))
        suffix_output = pack_active(noise_output.transpose(1, 2))
        merged = torch.empty_like(history_output)
        merge_attn_states(
            merged, history_output, history_lse, suffix_output, noise_lse
        )
        merged = merged.view(
            len(active), query_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        if len(active) == batch:
            return merged
        outputs = list(noise_output.unbind(0))
        for index, request in enumerate(active):
            outputs[request] = merged[index]
        return torch.stack(outputs)


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
        *,
        page_table: torch.Tensor | None = None,
        context_length: int | tuple[int, ...] = 0,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states),
            cached_k,
            cached_v,
            noise_positions,
            rotary_cache,
            page_table=page_table,
            context_length=context_length,
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
        # Every noisy row predicts one proposal, including the anchor row.
        # The verifier also consumes the anchor id, so its width is one larger.
        self.block_size = config.block_size
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
        self._paged_batch_table = _Qwen38DSparkBatchTable(self.noise_width)

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

    @torch.no_grad()
    def prefill_paged(
        self,
        main_hidden: torch.Tensor,
        *,
        segment: Segment,
        cache: DSparkRequestCache,
    ) -> Qwen38DSparkPagedCache:
        """Seed or append a prompt chunk in storage reserved by the caller."""
        self.append_paged_context(
            main_hidden, start_pos=cache.context_length + main_hidden.shape[1],
            segment=segment, cache=cache,
        )
        return Qwen38DSparkPagedCache(segment=segment, state=cache)

    @torch.no_grad()
    def append_paged_context(
        self,
        main_hidden: torch.Tensor,
        *,
        start_pos: int,
        segment: Segment,
        cache: DSparkRequestCache,
    ) -> None:
        """Append accepted context to an already-reserved draft segment.

        This single-request BF16/TP=1 path uses the same position convention as
        forward_draft: start_pos is the exclusive context end (the anchor
        position), and the suffix starts at cache.context_length. Logical slot
        IDs address the segment's strided views; they are not physical page IDs.

        Only new rows are projected. Publish the valid length after all layer
        writes are enqueued, preserving commit_dspark_context's invariant.
        A layer failure leaves the old valid prefix unchanged; an unpublished
        suffix may be overwritten on retry. The caller serializes access on
        one CUDA stream (or supplies synchronization), and must not retry a
        fatal asynchronous device error as though it were a Python exception.
        """
        self._validate_paged_context(main_hidden, start_pos, segment, cache)
        length = main_hidden.shape[1]
        if length == 0:
            return

        from gllm.layers.ops.cache_kernels import reshape_and_cache_flash

        context = self._project_context(main_hidden)
        positions = torch.arange(
            cache.context_length, start_pos, device=main_hidden.device
        )
        page_table = cache.get_device_page_table(main_hidden.device)[0]
        slot_mapping = (
            page_table[positions // segment.page_size].long() * segment.page_size
            + positions % segment.page_size
        )
        for index, layer in enumerate(self.layers):
            key, value = layer.self_attn.prefill_context(
                context, positions, self.rotary_emb.cos_sin_cache
            )
            # The writer requires packed head/dim axes. Copy only the new
            # suffix, never the existing history. "auto" ignores scale tensors.
            key = key[0].transpose(0, 1).contiguous()
            value = value[0].transpose(0, 1).contiguous()
            reshape_and_cache_flash(
                key, value, segment.k_cache[index], segment.v_cache[index],
                slot_mapping, "auto", key, value,
            )
        cache.context_length = start_pos

    def _validate_paged_context(self, main_hidden, start_pos, segment, cache):
        if (
            main_hidden.ndim != 3
            or main_hidden.shape[0] != 1
            or main_hidden.shape[2] != self.config.target_hidden_size
        ):
            raise ValueError("paged context requires [1, S, target_hidden_size]")
        if _tp_size() != 1:
            raise ValueError("paged DSpark context currently requires TP=1")
        geometry = (segment.num_layers, segment.kv_head_num, segment.kv_head_dim)
        expected = (
            len(self.layers), self.config.num_key_value_heads, self.config.head_dim
        )
        if geometry != expected:
            raise ValueError((geometry, expected))
        key_cache = segment.k_cache[0]
        if (
            main_hidden.dtype != torch.bfloat16
            or key_cache.dtype != main_hidden.dtype
            or key_cache.device != main_hidden.device
        ):
            raise ValueError("paged context and cache must share a BF16 device")
        length = main_hidden.shape[1]
        if type(start_pos) is not int or start_pos < 0:
            raise ValueError("paged context position must be a nonnegative integer")
        if start_pos - length != cache.context_length:
            raise ValueError((start_pos, length, cache.context_length))
        if start_pos > len(cache.page_table) * segment.page_size:
            raise ValueError("draft context exceeds reserved page capacity")
        if start_pos > self.rotary_emb.cos_sin_cache.shape[0]:
            raise ValueError("draft context exceeds the rotary position range")

    def _validate_paged_batch(self, main_hiddens, start_positions, caches):
        if not caches or not len(main_hiddens) == len(start_positions) == len(caches):
            raise ValueError("paged batch needs one context, position and cache per request")
        if any(not isinstance(cache, Qwen38DSparkPagedCache) for cache in caches):
            raise ValueError("paged batch requires request-owned paged cache views")
        if len({id(cache.state) for cache in caches}) != len(caches):
            raise ValueError("paged batch cannot repeat a request cache")
        segment = caches[0].segment
        if any(cache.segment is not segment for cache in caches):
            raise ValueError("paged batch must use one shared draft segment")
        for hidden, end, cache in zip(main_hiddens, start_positions, caches):
            self._validate_paged_context(hidden, end, segment, cache.state)

    def _append_paged_context_batched(self, main_hiddens, start_positions, caches):
        """Write only the ragged new suffixes; publish all lengths last.

        Validation is complete before any write. Only the newly accepted
        context is packed, never the historical KV. A layer failure leaves all
        old valid lengths intact; callers must not retry fatal CUDA errors.
        """
        nonempty = [
            (hidden, end, cache) for hidden, end, cache in zip(
                main_hiddens, start_positions, caches
            ) if hidden.shape[1]
        ]
        if not nonempty:
            return
        from gllm.layers.ops.cache_kernels import reshape_and_cache_flash

        segment = caches[0].segment
        context = self._project_context(torch.cat([row[0] for row in nonempty], dim=1))
        positions, slots = [], []
        for hidden, end, cache in nonempty:
            position = torch.arange(cache.state.context_length, end, device=hidden.device)
            page_table = cache.state.get_device_page_table(hidden.device)[0]
            positions.append(position)
            slots.append(
                page_table[position // segment.page_size].long() * segment.page_size
                + position % segment.page_size
            )
        positions, slots = torch.cat(positions), torch.cat(slots)
        for index, layer in enumerate(self.layers):
            key, value = layer.self_attn.prefill_context(
                context, positions, self.rotary_emb.cos_sin_cache
            )
            key = key[0].transpose(0, 1).contiguous()
            value = value[0].transpose(0, 1).contiguous()
            reshape_and_cache_flash(
                key, value, segment.k_cache[index], segment.v_cache[index],
                slots, "auto", key, value,
            )
        for end, cache in zip(start_positions, caches):
            cache.state.context_length = end

    @torch.no_grad()
    def prefill_paged_batched(
        self,
        main_hiddens: list[torch.Tensor],
        *,
        segment: Segment,
        caches: list[DSparkRequestCache],
    ) -> list[Qwen38DSparkPagedCache]:
        """Append independent prompt chunks with one packed context projection."""
        if any(hidden.ndim != 3 for hidden in main_hiddens):
            raise ValueError("paged context requires [1, S, target_hidden_size]")
        views = [Qwen38DSparkPagedCache(segment, cache) for cache in caches]
        ends = [
            cache.context_length + hidden.shape[1] for hidden, cache in zip(main_hiddens, caches)
        ]
        self._validate_paged_batch(main_hiddens, ends, views)
        self._append_paged_context_batched(main_hiddens, ends, views)
        return views

    @torch.no_grad()
    def forward_draft_batched(
        self,
        main_hiddens: list[torch.Tensor],
        input_ids: torch.Tensor,
        *,
        start_positions: list[int],
        caches: list[Qwen38DSparkPagedCache],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a real batch with ragged committed suffixes and private anchors.

        Each context is [1, S_i, taps*H], ending at start_positions[i].
        Returns [B, noise_width, H] and the noisy input IDs [B, noise_width].
        The shared forward protocol's B=1/dense entry is unchanged. All
        reservations and stream/completion ownership remain with the caller.
        Context is published before noise; a later failure does not roll it
        back. Callers must not replay an already appended suffix on failure.
        """
        self._validate_paged_batch(main_hiddens, start_positions, caches)
        anchor = _anchor_ids(input_ids)
        device = caches[0].segment.k_cache[0].device
        if anchor.shape != (len(caches),) or anchor.device != device or anchor.dtype != torch.int64:
            raise ValueError("paged batch needs one CUDA int64 anchor per request")
        if any(
            end + self.noise_width > self.rotary_emb.cos_sin_cache.shape[0]
            for end in start_positions
        ):
            raise ValueError("draft noise exceeds the rotary position range")
        if len(caches) == 1:
            return self.forward_draft(
                main_hiddens[0], input_ids, start_pos=start_positions[0], caches=caches[0]
            )
        self._append_paged_context_batched(main_hiddens, start_positions, caches)
        lengths = tuple(start_positions)
        page_table = self._paged_batch_table.prepare(caches, lengths)
        noise_positions = torch.stack([
            torch.arange(end, end + self.noise_width, device=device) for end in lengths
        ])
        hidden = self._embed_noise(input_ids)
        draft_ids = anchor.new_full((len(caches), self.noise_width), self.mask_token_id)
        draft_ids[:, 0] = anchor
        segment = caches[0].segment
        for index, layer in enumerate(self.layers):
            hidden = layer(
                hidden, segment.k_cache[index], segment.v_cache[index],
                noise_positions, self.rotary_emb.cos_sin_cache,
                page_table=page_table, context_length=lengths,
            )
        return self.norm(hidden), draft_ids

    def forward_draft(
        self,
        main_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        start_pos: int,
        caches: Qwen38DSparkCache | Qwen38DSparkPagedCache,
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
        The paged path checks this convention and the noise position range
        before appending. Once appended, context remains valid even if later
        noise computation fails. The dense path retains its existing API.

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
        anchor = _anchor_ids(input_ids)
        page_table = None
        if isinstance(caches, Qwen38DSparkPagedCache):
            if anchor.numel() != 1:
                raise ValueError("paged draft requires a single anchor")
            if start_pos + self.noise_width > self.rotary_emb.cos_sin_cache.shape[0]:
                raise ValueError("draft noise exceeds the rotary position range")
            self.append_paged_context(
                main_hidden, start_pos=start_pos,
                segment=caches.segment, cache=caches.state,
            )
            num_pages = (
                start_pos + caches.segment.page_size - 1
            ) // caches.segment.page_size
            page_table = caches.state.get_device_page_table(
                main_hidden.device
            )[:, :num_pages]
        else:
            context = self._project_context(main_hidden)
        noise = self._embed_noise(input_ids)
        draft_ids = anchor.new_full(
            (anchor.shape[0], self.noise_width), self.mask_token_id
        )
        draft_ids[:, 0] = anchor

        # The block sits *after* the context it was given: the new rows occupy
        # ``[start_pos - len(context), start_pos)`` and the noisy rows follow.
        context_positions = torch.arange(
            start_pos - main_hidden.shape[1], start_pos, device=noise.device
        )
        noise_positions = torch.arange(
            start_pos, start_pos + self.noise_width, device=noise.device
        )
        rotary_cache = self.rotary_emb.cos_sin_cache

        hidden = noise
        for index, layer in enumerate(self.layers):
            if page_table is not None:
                cached_k = caches.segment.k_cache[index]
                cached_v = caches.segment.v_cache[index]
            else:
                new_k, new_v = layer.self_attn.prefill_context(
                    context, context_positions, rotary_cache
                )
                cached_k = torch.cat([caches.keys[index], new_k], dim=2)
                cached_v = torch.cat([caches.values[index], new_v], dim=2)
                caches.keys[index] = cached_k
                caches.values[index] = cached_v
            hidden = layer(
                hidden, cached_k, cached_v, noise_positions, rotary_cache,
                page_table=page_table, context_length=start_pos,
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
        # The anchor input row predicts the first proposal; retain every row.
        # The anchor id itself is prepended separately in output_ids below.
        proposal_hidden = hidden_states
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
    """Index shared [S] or request-specific [B, S] positions into cos/sin.

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
    if positions.ndim == 2:
        return cos[:, None], sin[:, None]
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
    "Qwen38DSparkPagedCache",
    "Qwen38DSparkConfig",
    "Qwen38DSparkMapping",
]
