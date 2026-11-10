"""Host-side output rules shared by ordinary and speculative decoding."""

from typing import Collection, Sequence


def output_prefix_length(
    tokens: Sequence[int],
    *,
    remaining_tokens: int,
    finish_tokens: Collection[int],
    ignore_eos: bool,
) -> int:
    """Count the visible prefix, including the first EOS within the budget.

    Tokens must already be on the host. This function does not mutate request
    state or decide whether an already launched completion may retain a token
    after its nominal budget is exhausted; callers supply that effective budget.
    """
    count = min(len(tokens), max(0, remaining_tokens))
    if not ignore_eos:
        for index in range(count):
            if int(tokens[index]) in finish_tokens:
                return index + 1
    return count


def append_output_row(
    ipc_package, seq, tokens, *, logprobs=None, prompt_logprobs=None
) -> None:
    """Append a finalized output row without changing token or cache progress.

    Preserve scalar/list wire shapes and row-aligned logprobs. A follower's
    explicit prompt entry, including None, takes precedence over local data;
    locally accumulated prompt logprobs are attached only once.
    """
    ipc_package.act_schedule_ids.append(seq.seq_id)
    ipc_package.next_tokens.append(tokens)
    ipc_package.logprobs.append(logprobs)
    if prompt_logprobs and seq.seq_id in prompt_logprobs:
        ipc_package.prompt_logprobs[seq.seq_id] = prompt_logprobs[seq.seq_id]
    elif (
        getattr(seq, "prompt_logprobs_enabled", False)
        and not seq._prompt_logprobs_sent
        and seq.prompt_logprobs_data is not None
    ):
        ipc_package.prompt_logprobs[seq.seq_id] = seq.prompt_logprobs_data
        seq._prompt_logprobs_sent = True
