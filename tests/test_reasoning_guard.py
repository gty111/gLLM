"""Sampling-time reasoning boundaries, including speculative rollback."""
import asyncio
import os
import re
from types import SimpleNamespace

import pytest
import torch

import gllm.structured_output as so
from gllm.runtime.sequence import GenerationSequence
from gllm.tokenizers.reasoning import ReasoningGuard, ThinkParser


class ByteTokenizer:
    markers = {"<eos>": 256, "<think>": 257, "</think>": 258}

    def convert_tokens_to_ids(self, marker):
        return self.markers.get(marker)

    def encode(self, text):
        result = []
        for part in re.split(r"(<think>|</think>)", text):
            result.extend([self.markers[part]] if part in self.markers else part.encode())
        return result

    def decode(self, ids, skip_special_tokens=False):
        pieces, pending = [], bytearray()
        for token in ids:
            if token < 256:
                pending.append(token)
            else:
                pieces.append(pending.decode(errors="replace"))
                pending.clear()
                if token != 256 or not skip_special_tokens:
                    pieces.append(next(k for k, v in self.markers.items() if v == token))
        pieces.append(pending.decode(errors="replace"))
        return "".join(pieces)


def guard_for(text, prefilled=True):
    tokenizer = ByteTokenizer()
    guard = ReasoningGuard(tokenizer, prefilled=prefilled)
    for token in tokenizer.encode(text):
        guard.accept(token)
    return guard


@pytest.mark.parametrize("text", [
    "Still reasoning.",
    '<tool_call><function=run><parameter=cmd>echo test</parameter></function></tool_call>',
    'An example: "</think>".',
    'An example: `</think>`.',
    '```text\n</think>\n```\n',
    '> </think>\n',
    '<!-- </think> -->',
])
def test_eos_blocked_for_unclosed_reasoning_and_literal_markers(text):
    assert not guard_for(text).allows_eos


@pytest.mark.parametrize("text", [
    'Ready.</think>Answer.',
    'An example: "</think>". Done.</think>Answer.',
    'An example: `</think>`. Done.</think>Answer.',
    '```text\n</think>\n```\nDone.</think>Answer.',
    'Unmatched ` example\n\nDone.</think>Answer.',
    'Unmatched " example </think>Answer.',
    '\u00e9\U0001f600</think>Answer.',
])
def test_eos_allowed_when_api_parser_closes_at_eof(text):
    assert guard_for(text).allows_eos


def test_non_native_marker_does_not_close_thinking():
    guard = ReasoningGuard(ByteTokenizer(), prefilled=True)
    for token in b'</think>':
        guard.accept(token)
    assert not guard.allows_eos


def test_speculative_guard_forks_and_rewinds():
    tok = ByteTokenizer()
    backend = so.StructuredSampler(tok)
    spec = so.StructuredOutput(None, 257, 258, True)
    seq = GenerationSequence(0, [1], [256], 64, structured_output=spec)
    logits = torch.zeros(3, 259)
    backend.mask_speculative(logits, [seq], [[1]], [[ord('a'), 258, ord('x')]])
    assert torch.isneginf(logits[0, 256])
    assert logits[1, 256] == 0
    assert not backend.states[seq].guard.allows_eos
    backend.mask_speculative(torch.zeros(1, 259), [seq], [[1, ord('a'), 258]], [[ord('x')]])
    assert backend.states[seq].guard.allows_eos
    logits = torch.zeros(1, 259)
    backend.mask_speculative(logits, [seq], [[1]], [[ord('b')]])
    assert not backend.states[seq].guard.allows_eos
    assert torch.isneginf(logits[0, 256])


def test_guard_fork_preserves_literal_waiting_state():
    guard = guard_for('An example: "</think>')
    trial = guard.fork()
    for token in ByteTokenizer().encode('". Continue thinking.'):
        trial.accept(token)
    assert not trial.allows_eos
    # EOF recovers the unmatched quote in the original, not the completed quote
    # in the speculative branch.
    assert guard.allows_eos


def test_detokenizer_window_does_not_grow_with_output():
    guard = ReasoningGuard(ByteTokenizer(), prefilled=True)
    for _ in range(10000):
        guard.accept(ord('a'))
    assert len(guard.decoder.token_ids) <= 1
    assert not guard.allows_eos


@pytest.mark.skipif(not os.environ.get('GLLM_TEST_QWEN_TOKENIZER'), reason='Checkpoint tokenizer unavailable')
def test_checkpoint_guard_tracks_literal_and_native_boundaries():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(os.environ['GLLM_TEST_QWEN_TOKENIZER'], local_files_only=True)
    guard = ReasoningGuard(tok, prefilled=True, prefix=tok.encode('<think>')[-1:])
    for token in tok.encode('Example: `</think>`. Keep reasoning.\n', add_special_tokens=False):
        guard.accept(token)
    assert not guard.allows_eos
    branch = guard.fork()
    for token in tok.encode('\u00e9\U0001f600</think>\nAnswer.', add_special_tokens=False):
        branch.accept(token)
    assert branch.allows_eos
    assert not guard.allows_eos
    assert len(branch.decoder.token_ids) <= 8


