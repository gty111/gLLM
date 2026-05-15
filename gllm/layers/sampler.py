import torch

from gllm.input_data import InputData


class Sampler:

    def forward(self, logits: torch.Tensor, input_data: InputData):
        # repetition_penalty
        logits /= torch.where(logits > 0, input_data.repetition_penalty, 1.0)
        logits *= torch.where(logits <= 0, 1.0, input_data.repetition_penalty)
        # temperature
        logits.div_(input_data.temperature.unsqueeze_(dim=1))
        # top_p top_k
        logits = self._apply_top_k_top_p(logits, input_data.top_p, input_data.top_k)
        probs = torch.softmax(logits, dim=1)

        q = torch.empty_like(probs)
        q.exponential_()
        return probs.div_(q).argmax(dim=1).cpu().numpy().tolist()
        # return torch.multinomial(probs, 1).squeeze(1).cpu().numpy().tolist()

    def _apply_top_k_top_p(
        self,
        logits: torch.Tensor,
        p: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        # Use topk instead of full sort for better performance.
        # Determine the maximum k value needed across the batch.
        k_long = k.to(torch.long)
        max_k = min(k_long.max().item(), logits.size(1))

        # Get top-k values and indices (descending order).
        top_values, top_indices = torch.topk(logits, max_k, dim=-1, sorted=True)

        # Apply per-sample top-k masking (for samples with k < max_k).
        if not (k_long == max_k).all():
            # Create a range tensor [0, 1, ..., max_k-1] and mask positions >= per-sample k
            range_tensor = torch.arange(max_k, device=logits.device).unsqueeze(0)
            topk_mask = range_tensor >= k_long.unsqueeze(1)
            top_values.masked_fill_(topk_mask, -float("inf"))

        # Apply top-p filtering on the top-k values.
        probs_sort = torch.softmax(top_values, dim=-1)
        probs_cumsum = probs_sort.cumsum(dim=-1)
        # Remove tokens with cumulative probability above the threshold,
        # but always keep at least one token (the top-1).
        top_p_mask = (probs_cumsum - probs_sort) >= p.unsqueeze(dim=1)
        top_values.masked_fill_(top_p_mask, -float("inf"))

        # Scatter filtered values back to original logit positions.
        # Fill with -inf first, then place the filtered top-k values.
        logits.fill_(-float("inf"))
        logits.scatter_(dim=-1, index=top_indices, src=top_values)
        return logits
