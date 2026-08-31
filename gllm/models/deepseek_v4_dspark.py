"""DeepSeek-V4 DSpark speculative model in native checkpoint precision."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from gllm.layers.attention.deepseek_v4.dspark import (
    DeepseekV4DSparkAttention,
    DeepseekV4DSparkAttentionCache,
)
from gllm.layers.deepseek_v4_mhc import mhc_head, mhc_post
from gllm.layers.layernorm import RMSNorm
from gllm.layers.linear import ReplicatedLinear
from gllm.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from gllm.models.weight_loader import WeightRule, contains, run_weight_loader

from .deepseek_v4 import DeepseekV4DecoderLayer, _v4_src_key


class DeepseekV4DSparkBlock(DeepseekV4DecoderLayer):
    """One of the three score-routed ``mtp.*`` DSpark stages."""

    def __init__(self, stage_id: int, config: Any) -> None:
        self.stage_id = stage_id
        super().__init__(
            config.num_hidden_layers + stage_id,
            config,
            attention_cls=DeepseekV4DSparkAttention,
        )

    def prefill_main(
        self,
        main_hidden: torch.Tensor,
        cache: DeepseekV4DSparkAttentionCache | None = None,
    ) -> DeepseekV4DSparkAttentionCache:
        return self.attn.prefill_main(main_hidden, cache)

    def forward_draft(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        main_hidden: torch.Tensor,
        *,
        start_pos: int,
        cache: DeepseekV4DSparkAttentionCache,
    ) -> torch.Tensor:
        layer_input, residual, post, comb = self._attention_input(hidden_states)
        attention_output = self.attn.forward_draft_block(
            layer_input,
            main_hidden,
            start_pos=start_pos,
            cache=cache,
        )
        hidden_states = mhc_post(attention_output, residual, post, comb)
        return self._ffn(hidden_states, input_ids)


class DeepseekV4DSpark(nn.Module):
    """Reference DSpark data flow for the three native ``mtp.*`` stages.

    This module intentionally remains separate from gLLM's sequential NextN
    runtime: DSpark jointly predicts a fixed noisy block and adds a Markov-logit
    correction, which is a different protocol from repeatedly invoking a
    one-token MTP layer.
    """

    def __init__(
        self,
        config: Any,
        *,
        embed: VocabParallelEmbedding,
        lm_head: ParallelLMHead,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.block_size = config.dspark_block_size
        self.noise_token_id = config.dspark_noise_token_id
        self.target_layer_ids = tuple(config.dspark_target_layer_ids)
        self.num_stages = getattr(
            config, "dspark_num_stages", len(self.target_layer_ids)
        )
        if self.block_size <= 0:
            raise ValueError("DSpark block size must be positive")
        if not self.target_layer_ids:
            raise ValueError("DSpark requires at least one target hidden layer")
        if self.num_stages <= 0:
            raise ValueError("DSpark stage count must be positive")

        # Avoid registering the already-owned base embedding/head twice.
        self._embed = [embed]
        self._lm_head = [lm_head]
        quant_config = config.quantization_config
        main_width = self.hidden_size * len(self.target_layer_ids)
        self.main_proj = ReplicatedLinear(
            main_width,
            self.hidden_size,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=quant_config,
        )
        self.main_norm = RMSNorm(
            self.hidden_size, self.norm_eps, params_dtype=torch.bfloat16
        )

        self.blocks = nn.ModuleList(
            [DeepseekV4DSparkBlock(i, config) for i in range(self.num_stages)]
        )
        self.norm = RMSNorm(
            self.hidden_size, self.norm_eps, params_dtype=torch.bfloat16
        )

        markov_rank = config.dspark_markov_rank
        self.markov_w1 = VocabParallelEmbedding(
            config.vocab_size,
            markov_rank,
            params_dtype=torch.bfloat16,
        )
        self.markov_w2 = ParallelLMHead(
            config.vocab_size,
            markov_rank,
            params_dtype=torch.float32,
        )
        self.confidence_proj = ReplicatedLinear(
            self.hidden_size + markov_rank,
            1,
            bias=False,
            params_dtype=torch.float32,
        )
        hc_dim = self.hc_mult * self.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, hc_dim, dtype=torch.float32, device="cuda")
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32, device="cuda")
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32, device="cuda")
        )

    @staticmethod
    def _current_ids(input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim == 2 and input_ids.shape[1] == 1:
            return input_ids[:, 0]
        if input_ids.ndim != 1:
            raise ValueError("DSpark current token ids must have shape [B] or [B,1]")
        return input_ids

    def prepare_inputs(
        self,
        main_hidden: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project target hiddens and create the official noisy draft block."""
        if main_hidden.ndim != 3 or main_hidden.shape[-1] != (
            self.hidden_size * len(self.target_layer_ids)
        ):
            raise ValueError("DSpark main hidden shape does not match target layers")
        current_ids = self._current_ids(input_ids)
        main_x = self.main_norm(self.main_proj(main_hidden))
        draft_ids = current_ids.new_full(
            (current_ids.shape[0], self.block_size), self.noise_token_id
        )
        draft_ids[:, 0] = current_ids
        hidden = self._embed[0](draft_ids)
        hidden = hidden.unsqueeze(-2).expand(
            *hidden.shape[:-1], self.hc_mult, self.hidden_size
        ).contiguous()
        return hidden, main_x, draft_ids

    def prefill(
        self,
        main_hidden: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> list[DeepseekV4DSparkAttentionCache]:
        """Build every DSpark stage's target-history KV cache."""
        _, main_x, _ = self.prepare_inputs(main_hidden, input_ids)
        return [block.prefill_main(main_x) for block in self.blocks]

    def forward_draft(
        self,
        main_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        start_pos: int,
        caches: list[DeepseekV4DSparkAttentionCache],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return final HC draft states and the draft ids used for routing."""
        if len(caches) != len(self.blocks):
            raise ValueError("DSpark cache count does not match stage count")
        hidden, main_x, draft_ids = self.prepare_inputs(main_hidden, input_ids)
        for block, cache in zip(self.blocks, caches):
            hidden = block.forward_draft(
                hidden,
                draft_ids,
                main_x,
                start_pos=start_pos,
                cache=cache,
            )
        return hidden, draft_ids

    def forward_head(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        sample: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply DSpark LM, Markov and confidence heads."""
        hidden = mhc_head(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            norm_eps=self.norm_eps,
            hc_mult=self.hc_mult,
            hc_eps=self.hc_eps,
        )
        logits = self._lm_head[0](self.norm(hidden).float())
        current_ids = self._current_ids(input_ids)
        output_ids = current_ids.new_empty(
            current_ids.shape[0], self.block_size + 1
        )
        output_ids[:, 0] = current_ids
        markov_embeds = []
        if sample is None:
            sample = lambda x: x.argmax(dim=-1)
        for position in range(self.block_size):
            markov_embed = self.markov_w1(output_ids[:, position])
            # The checkpoint stores Markov embeddings in BF16 but the official
            # ParallelHead deliberately promotes its weight and input to FP32.
            logits[:, position].add_(self.markov_w2(markov_embed.float()))
            markov_embeds.append(markov_embed)
            output_ids[:, position + 1] = sample(logits[:, position])
        markov_hidden = torch.stack(markov_embeds, dim=1)
        confidence = self.confidence_proj(
            torch.cat([hidden, markov_hidden], dim=-1).float()
        ).squeeze(-1)
        return output_ids, logits, confidence



    def _src_key(self, key: str) -> str:
        """Map a DSpark parameter path to its ``mtp.*`` checkpoint key.

        The three stages are ``mtp.0/1/2``; everything that is logically the
        *head* of the joint model (final norm, mHC head fold, Markov
        correction, confidence head) is stored on the last stage, and the
        target-hidden projection on the first.
        """
        last = f"mtp.{self.num_stages - 1}"
        if key.startswith("blocks."):
            return _v4_src_key("mtp." + key[len("blocks.") :])
        if key.startswith("main_"):
            return _v4_src_key(f"mtp.0.{key}")
        if key.startswith("markov_w"):
            name = key.split(".", 1)[0]
            return f"{last}.markov_head.{name}.weight"
        if key.startswith("confidence_proj"):
            return f"{last}.confidence_head.proj.weight"
        return _v4_src_key(f"{last}.{key}")

    def weight_rules(self, parent):
        """Reuse the target model's rules; only the head params differ."""
        from .deepseek_v4 import _h_fp32, _h_vocab_shard

        return [
            WeightRule(contains("markov_w"), _h_vocab_shard, "dspark_markov"),
            WeightRule(contains("confidence_proj"), _h_fp32, "dspark_confidence"),
        ] + parent.weight_rules()

    @torch.no_grad()
    def load_weights(self, weights, parent, mp_load_progress=None) -> None:
        ctx = parent._make_load_context(weights)
        last = self.num_stages - 1
        ctx.extra["vocab_shards"].update(
            {
                f"mtp.{last}.markov_head.markov_w1.weight": (
                    self.markov_w1.shard_indices
                ),
                f"mtp.{last}.markov_head.markov_w2.weight": (
                    self.markov_w2.shard_indices
                ),
            }
        )
        ctx.extra["experts"].update(
            {
                f"mtp.{stage}.ffn.experts": block.ffn.experts
                for stage, block in enumerate(self.blocks)
            }
        )
        run_weight_loader(
            self,
            weights,
            self.weight_rules(parent),
            mp_load_progress,
            pp_idx_offset=2,
            start_layer=0,
            ctx=ctx,
            src_key_fn=self._src_key,
        )

    def process_weights_after_loading(self) -> None:
        for block in self.blocks:
            block.ffn.experts.process_weights_after_loading()


__all__ = ["DeepseekV4DSpark", "DeepseekV4DSparkBlock"]
