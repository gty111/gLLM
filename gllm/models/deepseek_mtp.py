"""DeepSeek-V3.2 MTP (Multi-Token Prediction) head for speculative decoding.

The V3/V3.2 checkpoint ships ONE extra transformer block after the base model's
``num_hidden_layers`` — for V3.2 that is ``model.layers.61.*``. It is a *complete*
decoder layer (MLA + DSA indexer + MoE), plus the MTP-specific
``enorm`` / ``hnorm`` / ``eh_proj`` fusion and a self-contained
``embed_tokens`` + ``shared_head.{norm,head}`` (its own LM head — NOT tied to the
base model's ``lm_head``; the checkpoint stores a separate copy).

Forward (the standard DeepSeek V3/V3.2 MTP head computation):

    e = enorm(embed(input_id))         # embedding of the *next* token
    h = hnorm(prev_hidden)             # hidden state that produced prev token
    x = eh_proj(cat([e, h], dim=-1))   # fuse -> hidden_size
    x, residual = mtp_block(x)         # one full V3.2 decoder layer (w/ DSA)
    x = residual + x
    logits = shared_head.head(shared_head.norm(x))

Used only as the *draft* model in speculative decoding; it shares the target's
paged KV / DSA-index cache (it is layer_id 61 in the same MemoryManager).
"""

from typing import Optional

import torch
import torch.nn as nn

from gllm.input_data import InputData
from gllm.layers.layernorm import RMSNorm
from gllm.layers.linear import ReplicatedLinear
from gllm.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding

from .deepseek_v32 import DeepseekV32DecoderLayer


class DeepseekMTP(nn.Module):
    """A single DeepSeek MTP head (one nextn layer).

    ``layer_id`` is the block's global layer index (61 for V3.2), which is also
    the ``layer_id`` its MLA attention + DSA indexer use to index the shared
    paged KV / index cache in the MemoryManager.
    """

    def __init__(self, config, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        eps = config.rms_norm_eps
        hidden = config.hidden_size

        # MTP-specific fusion of (next-token embedding, previous hidden state).
        self.enorm = RMSNorm(hidden, eps=eps)
        self.hnorm = RMSNorm(hidden, eps=eps)
        # eh_proj: [2*hidden -> hidden], replicated (small, unsharded in ckpt).
        self.eh_proj = ReplicatedLinear(hidden * 2, hidden, bias=False)

        # Own token embedding (checkpoint stores model.layers.61.embed_tokens).
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, hidden)

        # One full V3.2 decoder layer (MLA + DSA indexer + MoE). glb_layer_id ==
        # layer_id so the MoE branch is selected (>= first_k_dense_replace) and
        # the DSA indexer keys into slot ``layer_id`` of the shared index cache.
        self.mtp_block = DeepseekV32DecoderLayer(layer_id, layer_id, config)

        # Self-contained head: RMSNorm + LM head (separate from base lm_head).
        self.shared_head_norm = RMSNorm(hidden, eps=eps)
        self.shared_head = ParallelLMHead(config.vocab_size, hidden)

    def forward(
        self,
        input_data: InputData,
        prev_hidden: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return the MTP block's post-norm hidden state ``[num_tokens, hidden]``.

        ``prev_hidden``: hidden state (pre-final-norm) of the position whose
        *next* token we are drafting. ``input_ids``: the already-known token id
        at each of those positions (the token the draft is conditioned on).
        """
        e = self.enorm(self.embed_tokens(input_ids))
        h = self.hnorm(prev_hidden)
        x = self.eh_proj(torch.cat([e, h], dim=-1))
        # mtp_block is a standard decoder layer: (input_data, hidden, residual).
        x, residual = self.mtp_block(input_data, x, None)
        x = residual + x
        return self.shared_head_norm(x)

    def logits_from_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.shared_head(hidden_states)

    def _src_key(self, param_name: str) -> str:
        """Map an MTP-module parameter name to its checkpoint key.

        The checkpoint stores the whole head under ``model.layers.{L}.``:

            enorm/hnorm/eh_proj/embed_tokens  -> model.layers.L.<same>
            mtp_block.<x>                     -> model.layers.L.<x>
                (input_layernorm, post_attention_layernorm, self_attn.*, mlp.*)
            shared_head_norm.<x>              -> model.layers.L.shared_head.norm.<x>
            shared_head.<x>                   -> model.layers.L.shared_head.head.<x>
        """
        L = self.layer_id
        if param_name.startswith("mtp_block."):
            rest = param_name[len("mtp_block."):]
        elif param_name.startswith("shared_head_norm."):
            rest = "shared_head.norm." + param_name[len("shared_head_norm."):]
        elif param_name.startswith("shared_head."):
            rest = "shared_head.head." + param_name[len("shared_head."):]
        else:
            # enorm / hnorm / eh_proj / embed_tokens: verbatim under the layer.
            rest = param_name
        return f"model.layers.{L}.{rest}"

    def load_weights(self, weights, parent_lm, mp_load_progress=None):
        """Load the layer-L MTP head weights, reusing the base V3.2 rule table.

        ``parent_lm`` is the already-built ``DeepseekV32ForCausalLM`` — we borrow
        its ``weight_rules()`` (MLA fused-qkv, column/row parallel, FusedMoE
        expert fusion, FP8 scales) and its ``LoadContext`` (expert_map, head
        geometry). We remap each of THIS module's parameter names to the
        checkpoint's ``model.layers.{L}.*`` key via :meth:`_src_key`, so the exact
        same handlers that loaded layers 0..60 load the MTP block unchanged.
        """
        from .weight_loader import get_tensor_from_dict, moe_expert_load_pool
        from .weight_utils import copy_single_proj_dim0

        rules = parent_lm.weight_rules()
        ctx = parent_lm._make_load_context(weights)
        params = dict(self.named_parameters())

        def _load_all():
            for name, p in params.items():
                src = self._src_key(name)
                # ``shared_head`` is this head's LM head (ParallelLMHead) and
                # ``embed_tokens`` is its VocabParallelEmbedding -- both are
                # dim-0 (row/vocab) TP-sharded, like the base lm_head/embed. The
                # base ``contains("embed_tokens","lm_head")`` rule keys off those
                # substrings, which the remapped ``shared_head.head`` key lacks,
                # so route these two explicitly through the dim-0 slicer.
                if name.startswith("shared_head.") or name.startswith("embed_tokens."):
                    copy_single_proj_dim0(p.data, get_tensor_from_dict(weights, src))
                    continue
                for rule in rules:
                    if rule.match(src):
                        rule.handler(ctx, src, p.data)
                        break
                else:
                    p.data.copy_(get_tensor_from_dict(weights, src))

        if ctx.num_experts is not None:
            with moe_expert_load_pool(ctx.num_experts) as pool:
                ctx.pool = pool
                _load_all()
        else:
            _load_all()

