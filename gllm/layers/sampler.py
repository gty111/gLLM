import torch

from gllm.input_data import InputData

class Sampler():
    
    def forward(self, logits: torch.Tensor, input_data: InputData):
        # repetition_penalty
        logits /= torch.where(logits>0, input_data.repetition_penalty, 1.0)
        logits *= torch.where(logits<=0, 1.0, input_data.repetition_penalty)
        # temperature
        logits.div_(input_data.temperature.unsqueeze_(dim=1))
        # top_p top_k
        logits = self._apply_top_k_top_p(logits, input_data.top_p, input_data.top_k)
        probs = torch.softmax(logits, dim=1)
        # q = torch.empty_like(probs)
        # q.exponential_()
        # return probs.div_(q).argmax(dim=1).cpu().numpy().tolist()
        return torch.multinomial(probs, 1).squeeze(1).cpu().numpy().tolist()

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