import torch
from sgl_kernel import top_k_top_p_sampling_from_probs

from gllm.input_data import InputData
from gllm.layers.repetition_penalty import apply_scaling_penalties


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

    def forward_gpu(self, logits: torch.Tensor, input_data: InputData) -> torch.Tensor:
        """Sample on GPU; caller is responsible for D2H."""
        flags = self._get_sampling_flags(input_data)

        if flags["need_repetition_penalty"]:
            apply_scaling_penalties(logits, input_data.repetition_penalty)

        if flags["is_all_greedy"]:
            # argmax is invariant to positive temperature scaling, so the
            # full-vocab div_ would be wasted work here -- skip it.
            return torch.argmax(logits, dim=-1)

        if flags["need_temperature"]:
            logits.div_(input_data.temperature.unsqueeze(1))

        probs = torch.softmax(logits, dim=-1)
        return _fused_top_k_top_p_sample(probs, input_data.top_k, input_data.top_p)

    def forward(self, logits: torch.Tensor, input_data: InputData) -> list[int]:
        return self.forward_gpu(logits, input_data).cpu().tolist()

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
