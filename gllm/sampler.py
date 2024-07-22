import torch


class Sampler():
    def __init__(self, logits: torch.Tensor, top_p: float, temperature: float):
        self.logits = logits
        self.top_p = top_p
        self.temperature = temperature

    def forward(self):
        self.logits.div_(self.temperature)
        logits = self._apply_top_p(self.logits, torch.tensor(
            [self.top_p], device='cuda'))
        probs = torch.softmax(logits, dim=1)
        next_tokens = torch.multinomial(probs, 1).squeeze(1).cpu().numpy().tolist()
        return next_tokens

    def _apply_top_p(
        self,
        logits: torch.Tensor,
        p: torch.Tensor,
    ) -> torch.Tensor:
        logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

        # Apply top-p.
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = probs_sort.cumsum(dim=-1)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        # at least one
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

        # Re-sort the probabilities.
        src = torch.arange(logits_idx.shape[-1],
                           device=logits_idx.device).expand_as(logits_idx)
        logits_idx_inv = torch.empty_like(logits_idx).scatter_(dim=-1,
                                                               index=logits_idx,
                                                               src=src)
        logits = torch.gather(logits_sort, dim=-1, index=logits_idx_inv)
        return logits
