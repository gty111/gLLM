import torch

try:
    from flashinfer.sampling import top_k_top_p_sampling_from_probs
except (ImportError, OSError):  # FlashInfer is unavailable on ROCm.
    top_k_top_p_sampling_from_probs = None

from gllm.runtime.input_data import InputData
from gllm.layers.repetition_penalty import apply_scaling_penalties


def _top_k_top_p_torch(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
) -> torch.Tensor:
    """Apply joint top-k/top-p filtering and sample without FlashInfer."""
    values, token_ids = probs.float().sort(dim=-1, descending=True)
    ranks = torch.arange(values.shape[-1], device=values.device).unsqueeze(0)
    ks = torch.where(top_ks <= 0, values.shape[-1], top_ks).unsqueeze(1)
    values = values.masked_fill(ranks >= ks, 0.0)

    ps = top_ps.to(values.dtype).unsqueeze(1)
    cumulative = values.cumsum(dim=-1)
    # Keep the token that first brings the cumulative probability above p.
    excluded = (cumulative - values > ps) & (ps > 0) & (ps < 1)
    values = values.masked_fill(excluded, 0.0)
    values = values / values.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    sampled_ranks = torch.multinomial(values, 1)
    return token_ids.gather(1, sampled_ranks).squeeze(1)


def _fused_top_k_top_p_sample(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
) -> torch.Tensor:
    """Use FlashInfer when available, otherwise sample with PyTorch."""
    if top_k_top_p_sampling_from_probs is None:
        return _top_k_top_p_torch(probs, top_ks, top_ps)
    return top_k_top_p_sampling_from_probs(
        probs.float().contiguous(),
        top_ks.to(torch.int32),
        top_ps,
        filter_apply_order="joint",
    ).to(torch.int64)  # Same wire dtype as argmax and the PP FutureMap receiver.


class Sampler:

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self._structured = None

    def prepare_structured(self, logits, seqs):
        if not any(getattr(s, "structured_output", None) is not None for s in seqs):
            return None
        if self._structured is None:
            from gllm.structured_output import StructuredSampler

            self._structured = StructuredSampler(self.tokenizer)
        return self._structured.prepare(seqs, logits.shape[-1], logits.device)

    def forward_gpu(
        self,
        logits: torch.Tensor,
        input_data: InputData,
        return_logprobs: bool = False,
        num_logprobs: int = 0,
        *,
        structured=None,
    ):
        """Sample on GPU; caller is responsible for D2H.

        When ``return_logprobs`` is set the return value becomes
        ``(next_tokens, logprobs)`` where ``logprobs`` is the tuple produced by
        :meth:`compute_logprobs` (sampled-token logprob + top-``num_logprobs``
        alternatives). Logprobs are computed from the same (penalty- and
        temperature-adjusted) logits used to sample, so they match the
        effective sampling distribution.
        """
        flags = self._get_sampling_flags(input_data)

        if flags["need_repetition_penalty"]:
            apply_scaling_penalties(logits, input_data.repetition_penalty)

        active = None
        if any(getattr(s, "structured_output", None) is not None for s in input_data.seqs):
            if structured is None:
                structured = self.prepare_structured(logits, input_data.seqs)
            active = self._structured.apply(logits, structured)

        if flags["is_all_greedy"]:
            # argmax is invariant to positive temperature scaling, so the
            # full-vocab div_ would be wasted work here -- skip it.
            next_tokens = torch.argmax(logits, dim=-1)
            if active is not None:
                self._structured.record(active, next_tokens)
            if return_logprobs:
                return next_tokens, self.compute_logprobs(
                    logits, next_tokens, num_logprobs
                )
            return next_tokens

        if flags["need_temperature"]:
            logits.div_(input_data.temperature.unsqueeze(1))

        probs = torch.softmax(logits, dim=-1)
        next_tokens = _fused_top_k_top_p_sample(
            probs, input_data.top_k, input_data.top_p
        )
        if active is not None:
            self._structured.record(active, next_tokens)
        if return_logprobs:
            return next_tokens, self.compute_logprobs(
                logits, next_tokens, num_logprobs
            )
        return next_tokens

    def forward(self, logits: torch.Tensor, input_data: InputData) -> list[int]:
        return self.forward_gpu(logits, input_data).cpu().tolist()

    def stage_structured_feedback(self, tokens, seqs, stream=None):
        if self._structured is not None and any(
            getattr(s, "structured_output", None) is not None for s in seqs
        ):
            self._structured.stage_feedback(tokens, stream=stream)

    @staticmethod
    def compute_logprobs(
        logits: torch.Tensor,
        next_tokens: torch.Tensor,
        num_logprobs: int,
    ):
        """Return ``(sampled_logprob, top_vals, top_ids)`` on GPU.

        ``sampled_logprob`` is ``[batch]`` (logprob of the chosen token, always
        reported). ``top_vals`` / ``top_ids`` are ``[batch, k]`` (the k most
        likely tokens and their logprobs); ``k == 0`` yields empty columns.
        """
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        sampled = logprobs.gather(1, next_tokens.view(-1, 1)).squeeze(1)
        k = max(0, min(num_logprobs, logprobs.shape[-1]))
        if k > 0:
            top_vals, top_ids = torch.topk(logprobs, k, dim=-1)
            # A grammar can leave fewer than k valid tokens. JSON cannot carry
            # -Infinity; use the API's sentinel for zero-probability alternatives.
            top_vals = top_vals.clamp_min(-9999.0)
        else:
            top_vals = logprobs.new_zeros((logprobs.shape[0], 0))
            top_ids = next_tokens.new_zeros((logprobs.shape[0], 0))
        return sampled, top_vals, top_ids

    @staticmethod
    def _get_sampling_flags(input_data: InputData) -> dict[str, bool]:
        seqs = input_data.seqs
        return {
            "is_all_greedy": all(seq.top_k == 1 for seq in seqs),
            "need_repetition_penalty": getattr(
                input_data, "needs_repetition_penalty", False
            ),
            "need_temperature": any(
                seq.temperature > 1e-5 and abs(seq.temperature - 1.0) > 1e-5
                for seq in seqs
            ),
        }
