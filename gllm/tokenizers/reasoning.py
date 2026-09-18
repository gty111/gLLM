"""Incrementally separate a leading model-native thinking block from its answer."""


class ThinkParser:
    START = "<think>"
    END = "</think>"

    def __init__(self, *, prefilled=False):
        self.prefilled = prefilled
        self.state = "start"
        self.pending = ""

    def feed(self, text):
        if self.state == "content":
            return "", text
        self.pending += text
        if self.state == "start":
            head = self.pending.lstrip()
            if not head or (len(head) < len(self.START) and self.START.startswith(head)):
                return "", ""
            if head.startswith(self.START):
                self.pending = head[len(self.START):]
                self.state = "reasoning"
            elif self.prefilled:
                self.state = "reasoning"
            else:
                self.state = "content"
                content, self.pending = self.pending, ""
                return "", content
        end = self.pending.find(self.END)
        if end >= 0:
            reasoning = self.pending[:end]
            content = self.pending[end + len(self.END):]
            self.pending = ""
            self.state = "content"
            return reasoning, content
        # Hold only a possible split closing delimiter, never the whole thought.
        keep = 0
        for size in range(1, min(len(self.pending), len(self.END) - 1) + 1):
            if self.END.startswith(self.pending[-size:]):
                keep = size
        cut = len(self.pending) - keep
        reasoning, self.pending = self.pending[:cut], self.pending[cut:]
        return reasoning, ""

    def finish(self):
        pending, self.pending = self.pending, ""
        if self.state == "reasoning" or (self.state == "start" and self.prefilled):
            # A length-limited thought (including a partial end tag) is never
            # reclassified as an answer just because generation ended.
            return pending, ""
        return "", pending


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
        reasoning, content = parser.feed(item.text) if parser else ("", item.text)
        yield item, reasoning, content, False
    reasoning, content = parser.finish() if parser else ("", "")
    yield StreamOutput(""), reasoning, content, True
