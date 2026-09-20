"""Incrementally separate native thinking from answers without parsing examples."""
import copy

from gllm.tokenizers.literals import LiteralScanner


class ThinkParser:
    START = "<think>"
    END = "</think>"

    def __init__(self, *, prefilled=False):
        self.prefilled = prefilled
        self.started = prefilled
        self.state = "start"
        self.pending = ""
        self._pos = 0
        self._literals = LiteralScanner()
        self._native = set()
        self._token_aware = False

    def feed(self, text, *, control_tokens=None):
        if self.state == "content":
            return "", text
        if control_tokens is not None:
            self._token_aware = True
            self._native.update(len(self.pending) + offset for offset, _ in control_tokens)
        self.pending += text
        return self._consume(final=False)

    def _consume(self, *, final):
        text = self.pending
        if self.state == "start":
            head = text.lstrip()
            if not final and (not head or (len(head) < len(self.START) and self.START.startswith(head))):
                return "", ""
            start = len(text) - len(head)
            if head.startswith(self.START) and (not self._token_aware or start in self._native):
                self.started = True
                self._pos = start + len(self.START)
                self._literals = LiteralScanner(start=self._pos)
                self.state = "reasoning"
            elif self.prefilled:
                self.state = "reasoning"
            else:
                self.state = "content"
                self.pending = ""
                return "", text
        begin = self._pos
        while self._pos < len(text):
            pos = self._pos
            literal_end = self._literals.end(text, pos, final=final)
            if literal_end is None:
                break
            if literal_end > pos:
                self._pos = literal_end
                continue
            if text[pos] == "<":
                tail = text[pos:pos + len(self.END)]
                if not final and self.END.startswith(tail) and len(tail) < len(self.END):
                    break
                if tail == self.END and (not self._token_aware or pos in self._native):
                    self.state = "content"
                    reasoning, content = text[begin:pos], text[pos + len(self.END):]
                    self.pending = ""
                    self._native.clear()
                    return reasoning, content
            self._pos += 1
        return text[begin:self._pos], ""

    def finish(self):
        if self.state == "content":
            return "", ""
        return self._consume(final=True)

    def fork(self):
        """Copy mutable cursors; immutable buffered text can be shared."""
        result = copy.copy(self)
        result._literals = copy.copy(self._literals)
        result._native = self._native.copy()
        return result


class ReasoningGuard:
    """Block premature EOS using the same boundary rules as API parsing.

    Speculative sampling forks this state and commits only accepted tokens.
    Detokenization retains only the previous token and incomplete UTF-8 bytes,
    never the request's long prompt or its full generated-token history.
    """

    def __init__(self, tokenizer, *, prefilled, prefix=()):
        from gllm.runtime.sequence import GenerationSequence

        self.tokenizer = tokenizer
        self.controls = reasoning_control_tokens(tokenizer)
        self.parser = ThinkParser(prefilled=prefilled)
        self.decoder = GenerationSequence("reasoning_guard", list(prefix), [], 0)
        self.allows_eos = not prefilled

    def accept(self, token):
        if self.parser.state == "content":
            return
        text, controls = decode_stream_delta(self.decoder, self.tokenizer, [token], self.controls)
        self.parser.feed(text, control_tokens=controls)
        drop = max(0, self.decoder.cur_length - 1)
        if drop:
            del self.decoder.token_ids[:drop]
            self.decoder.cur_length -= drop
        self.allows_eos = not self.parser.started or self.parser.state == "content"
        if not self.allows_eos and any(
            pos >= self.parser._pos and self.parser.pending.startswith(ThinkParser.END, pos)
            for pos in self.parser._native
        ):
            # An unmatched inline delimiter may defer the end marker until
            # EOF. Probe EOF without committing that speculative recovery.
            probe = self.parser.fork()
            probe.finish()
            self.allows_eos = probe.state == "content"

    def fork(self):
        result = copy.copy(self)
        result.parser = self.parser.fork()
        result.decoder = copy.copy(self.decoder)
        result.decoder.token_ids = self.decoder.token_ids.copy()
        return result


def reasoning_control_tokens(tokenizer):
    """Discover native token IDs once, not in the per-delta hot path."""
    result = {}
    for marker in (ThinkParser.START, ThinkParser.END):
        token_id = tokenizer.convert_tokens_to_ids(marker)
        if token_id is not None and tokenizer.decode([token_id], skip_special_tokens=False) == marker:
            result[token_id] = marker
    return result if len(result) == 2 else {}


def decode_stream_delta(seq, tokenizer, tokens, controls):
    """Append committed tokens and preserve native delimiter character offsets.

    Ordinary deltas use the existing batched detokenizer. Only a batch containing
    a control token is split, including MTP batches containing both boundaries.
    Detokenize before/through each boundary to retain pending UTF-8 bytes. Keep
    native markers even on tokenizers that hide them with skip_special_tokens.
    """
    if not any(t in controls for t in tokens):
        for token in tokens:
            seq.append(token)
        return seq.detokenize_inc(tokenizer), ()
    pieces, positions = [], []
    size = 0
    for token in tokens:
        marker = controls.get(token)
        if marker is not None:
            prefix = seq.detokenize_inc(tokenizer)
            pieces.append(prefix)
            size += len(prefix)
        seq.append(token)
        if marker is not None:
            piece = seq.detokenize_inc(tokenizer)
            if not piece.endswith(marker):
                piece += marker
            positions.append((size + len(piece) - len(marker), marker))
            pieces.append(piece)
            size += len(piece)
    pieces.append(seq.detokenize_inc(tokenizer))
    return "".join(pieces), tuple(positions)


def create_reasoning_parser(tokenizer, token_ids):
    """Enable parsing only for tokenizers with native think delimiters.

    Inspect the actual generation prefix, not reasoning_effort: templates can
    enable thinking by default or override it through chat_template_kwargs.
    """
    if tokenizer is None:
        return None
    for marker in (ThinkParser.START, ThinkParser.END):
        token_id = tokenizer.convert_tokens_to_ids(marker)
        if token_id is None or tokenizer.decode([token_id], skip_special_tokens=False) != marker:
            return None
    suffix = tokenizer.decode(token_ids[-16:], skip_special_tokens=False)
    return ThinkParser(prefilled=suffix.rstrip().endswith(ThinkParser.START))


async def split_reasoning_stream(stream, parser):
    """Preserve engine metadata while splitting text, including EOF fragments."""
    from gllm.utils import StreamOutput

    async for item in stream:
        reasoning, content = parser.feed(item.text, control_tokens=item.control_tokens) if parser else ("", item.text)
        yield item, reasoning, content, False
    reasoning, content = parser.finish() if parser else ("", "")
    yield StreamOutput(""), reasoning, content, True
