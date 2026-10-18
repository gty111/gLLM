"""Helpers shared by the OpenAI-compatible serving endpoints."""

from typing import Optional

from gllm.entrypoints.protocol import UsageInfo


def get_finish_reason(seq) -> Optional[str]:
    """Best-effort OpenAI ``finish_reason`` for a completed sequence.

    ``length`` when the output-length cap was hit, ``stop`` when generation
    ended on an EOS/finish token, else ``stop`` as a generic fallback (e.g. an
    aborted/disconnected request). Returns ``None`` if no tokens were produced.
    """
    if seq is None or not seq.token_ids:
        return None
    generated = len(seq.token_ids) - seq.raw_prompt_len
    if not seq.ignore_eos and seq.token_ids[-1] in seq.finish_tokens:
        return "stop"
    if generated >= seq.output_len:
        return "length"
    return "stop"


def build_usage(seq):
    """OpenAI token-usage block computed from a (possibly finished) sequence."""
    if seq is None:
        return UsageInfo()
    prompt_tokens = seq.raw_prompt_len
    completion_tokens = max(0, len(seq.token_ids) - seq.raw_prompt_len)
    return UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
