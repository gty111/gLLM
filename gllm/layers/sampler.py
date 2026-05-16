import torch
from sgl_kernel import top_k_top_p_sampling_from_probs

from gllm.input_data import InputData


def _fused_top_k_top_p_sample(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
) -> torch.Tensor:
    """Fused top-k / top-p sampling via sgl_kernel."""
    return top_k_top_p_sampling_from_probs(
        probs.float().contiguous(),
        top_ks.to(torch.int32),
        top_ps,
        filter_apply_order="joint",
    )


class Sampler:

    def forward(self, logits: torch.Tensor, input_data: InputData) -> list[int]:
        flags = self._get_sampling_flags(input_data)

        if flags["need_repetition_penalty"]:
            self._apply_repetition_penalty(logits, input_data)

        if flags["is_all_greedy"]:
            if flags["need_temperature"]:
                logits.div_(input_data.temperature.unsqueeze(1))
            return torch.argmax(logits, dim=-1).cpu().tolist()

        if flags["need_temperature"]:
            logits.div_(input_data.temperature.unsqueeze(1))

        simple_case = (
            not flags["need_top_p_sampling"] and not flags["need_top_k_sampling"]
        )
        probs = torch.softmax(logits, dim=-1)

        if simple_case:
            batch_next_token_ids = torch.multinomial(probs, num_samples=1).view(-1)
        else:
            batch_next_token_ids = _fused_top_k_top_p_sample(
                probs, input_data.top_k, input_data.top_p
            )

        return batch_next_token_ids.cpu().tolist()

    @staticmethod
    def _get_sampling_flags(input_data: InputData) -> dict[str, bool]:
        seqs = input_data.seqs
        vocab_size = input_data.memory_manager.vocab_size
        return {
            "is_all_greedy": all(seq.top_k == 1 for seq in seqs),
            "need_top_p_sampling": any(seq.top_p < 1.0 for seq in seqs),
            "need_top_k_sampling": any(
                seq.top_k != -1 and seq.top_k < vocab_size for seq in seqs
            ),
            "need_repetition_penalty": getattr(
                input_data, "needs_repetition_penalty", False
            ),
            "need_temperature": any(
                seq.temperature > 1e-5 and abs(seq.temperature - 1.0) > 1e-5
                for seq in seqs
            ),
        }

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor, input_data: InputData
    ) -> None:
        penalty = input_data.repetition_penalty
        logits.div_(torch.where(logits > 0, penalty, 1.0))
        logits.mul_(torch.where(logits <= 0, 1.0, penalty))
