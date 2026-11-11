"""Shared scheduling contracts; algorithm execution stays with each backend."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SpeculativeCapabilities:
    """Permission to use existing execution gates, not proof they are enabled."""

    overlap: bool = True
    cuda_graph: bool = True


class TargetBackend:
    name = "none"
    capabilities = SpeculativeCapabilities()

    def schedule_method(self, scheduler):
        if scheduler.schedule_method in ("split_pd", "chunked_prefill"):
            return scheduler.chunked_prefill
        if scheduler.schedule_method == "token_throttling":
            return scheduler.token_throttling
        raise AssertionError("Unknown schedule method")

    def reservation_width(self, runner, num_decodes, *, phase=None, eligible=True, dp_attention=False):
        """Ordinary decode consumes one target input per request."""
        return 1


class MtpBackend(TargetBackend):
    name = "mtp"

    def reservation_width(self, runner, num_decodes, *, phase=None, eligible=True, dp_attention=False):
        """Preserve scheduler lookahead, independently of the execution decision.

        This is a reservation upper bound. Context/sampling gates may still
        decline MTP later. Do not replace the single-window budget check with
        a full-batch check: the scheduler uses this width to shrink the batch.
        """
        if not eligible or num_decodes <= 0 or dp_attention:
            return 1
        k = int(getattr(runner, "mtp_k", 0) or 0)
        if k <= 0 or getattr(runner.model, "mtp", None) is None:
            return 1
        max_batch = int(getattr(runner, "mtp_max_batch", 0) or 0)
        if max_batch > 0 and num_decodes > max_batch:
            return 1
        width = 1 + k
        return width if width <= runner.max_num_batched_tokens else 1


class DSparkBackend(TargetBackend):
    name = "dspark"
    capabilities = SpeculativeCapabilities(overlap=False, cuda_graph=False)

    def schedule_method(self, scheduler):
        return scheduler._schedule_dspark

    def reservation_width(self, runner, num_decodes, *, phase=None, eligible=True, dp_attention=False):
        """Prefill uses its ragged chunk length; these are fixed decode phases."""
        if phase == "verify":
            k = runner.dspark.block_size
            if type(k) is not int or k < 1:
                raise ValueError("DSpark verify requires a positive runtime proposal count")
            return k + 1
        if phase in ("bootstrap", "tail"):
            return 1
        raise ValueError("DSpark decode reservation requires a known decode phase")


_TARGET = TargetBackend()
_MTP = MtpBackend()
_DSPARK = DSparkBackend()


def select_speculative_backend(*, dspark_enabled: bool, mtp_enabled: bool):
    """Select after model loading resolves availability, not from auto config."""
    if dspark_enabled:
        return _DSPARK
    return _MTP if mtp_enabled else _TARGET
