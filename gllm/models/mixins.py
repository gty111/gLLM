"""Shared method boilerplate for the model wrappers.

Two recurring shapes lived as verbatim copies across the model files:

* every text causal LM gathered each seq's last hidden row and projected it
  through ``self.lm_head`` (:class:`StandardCausalLMMixin`);
* every multimodal wrapper re-delegated those two methods to its nested
  ``self.language_model`` (:class:`NestedLanguageModelMixin`).

A class that needs a different variant (ChatGLM's ``embed_input_ids``,
DeepSeek-V4's fp32 ``logits_from_hidden``) just overrides that one method.
"""

import torch

from gllm.runtime.input_data import InputData


class StandardCausalLMMixin:
    """Logits/embedding plumbing for a text LM with ``self.lm_head`` and a
    backbone ``self.model`` that itself exposes ``embed_input_ids``."""

    def compute_logits(self, input_data: InputData, hidden_states: torch.Tensor):
        idx_list = input_data.get_query_start_loc() - 1
        return self.logits_from_hidden(hidden_states[idx_list[1:]])

    def logits_from_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project the given hidden states to full-vocab logits.

        ``compute_logits`` gathers only each seq's last position (for
        sampling); this projects *every* supplied position and is used by the
        prompt-logprobs path. Keeping it here means LM-head placement (tied
        weights, TP gather, multimodal nesting) stays a model-internal detail.
        """
        return self.lm_head(hidden_states)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)


class NestedLanguageModelMixin:
    """Logits delegation for multimodal wrappers that nest the text model as
    ``self.language_model``."""

    def compute_logits(
        self,
        input_data: InputData,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(input_data, hidden_states)

    def logits_from_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.language_model.logits_from_hidden(hidden_states)
