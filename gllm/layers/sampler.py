import torch

from gllm.input_data import InputData


class Sampler:

    def forward(self, logits: torch.Tensor, input_data: InputData):
        # repetition_penalty — input_data.repetition_penalty is now (N,) shaped.
        # Unsqueeze once here for broadcasting instead of expanding to (N×vocab).
        penalty = input_data.repetition_penalty.unsqueeze(1)  # (N, 1)
        logits = torch.where(logits > 0, logits / penalty, logits * penalty)
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
        logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

        # Apply top-k.
        top_k_mask = logits_sort.size(1) - k.to(torch.long)
        # Get all the top_k values.
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

        # Apply top-p.
        # Re-use the already-computed sort; one softmax is enough.
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = probs_sort.cumsum(dim=-1)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        # at least one
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

        # Invert the sort permutation to restore original vocab ordering.
        # Using scatter_ on a pre-allocated buffer avoids a temporary arange
        # expand, but the dominant cost is the sort above; this path is unchanged.
        logits_idx_inv = torch.empty_like(logits_idx).scatter_(
            dim=-1,
            index=logits_idx,
            src=torch.arange(
                logits_idx.shape[-1], device=logits_idx.device
            ).expand_as(logits_idx),
        )
        logits = torch.gather(logits_sort, dim=-1, index=logits_idx_inv)
        return logits
