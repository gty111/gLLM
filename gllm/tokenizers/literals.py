"""Incremental literal shielding shared by reasoning and tool parsers.

Closed inline code/quoted examples are literal; unmatched inline delimiters
are ordinary text. Fenced code remains literal even if generation truncates.
Lookahead stops at paragraph boundaries and resumes at the last scanned byte,
so an unfinished example cannot cause quadratic rescanning on every delta.
"""


class LiteralScanner:
    def __init__(self, start=0):
        self._start = start
        self._waiting = None
        self._fence = None
        self._comment = False
        self._line_wait = None

    def _line_start(self, text, pos):
        begin = max(self._start, text.rfind("\n", self._start, pos) + 1)
        indent = text[begin:pos]
        return len(indent) <= 3 and not indent.strip(" ")

    def _line_end(self, text, start, final):
        scan = self._line_wait[1] if self._line_wait and self._line_wait[0] == start else start
        end = text.find("\n", scan)
        if end < 0 and not final:
            self._line_wait = (start, len(text))
            return None
        self._line_wait = None
        return len(text) if end < 0 else end

    def end(self, text, pos, *, final=False):
        """Return end of a literal span, pos for normal text, or None to wait."""
        char = text[pos]
        if self._comment:
            if "-->".startswith(text[pos:pos + 3]) and len(text) - pos < 3 and not final:
                return None
            if text.startswith("-->", pos):
                self._comment = False
                return pos + 3
            return pos + 1
        if self._fence is not None:
            fence_char, size = self._fence
            if char == fence_char and self._line_start(text, pos):
                end = pos
                while end < len(text) and text[end] == char:
                    end += 1
                if end == len(text) and not final:
                    return None
                if end - pos >= size:
                    line_end = self._line_end(text, end, final)
                    if line_end is None:
                        return None
                    if not text[end:line_end].strip():
                        self._fence = None
                        return line_end
            return pos + 1
        if char == "<":
            head = text[pos:pos + 4]
            if "<!--".startswith(head) and len(head) < 4 and not final:
                return None
            if head == "<!--":
                self._comment = True
                return pos + 4
        # Explicit Markdown block quotes and indented code are display text.
        if (char == ">" and self._line_start(text, pos)) or (
            (pos == self._start or (pos and text[pos - 1] == "\n")) and char in " \t"
        ):
            if char != ">" and char != "\t" and not text.startswith("    ", pos):
                if not final and len(text) - pos < 4 and not text[pos:].strip(" "):
                    return None
            else:
                return self._line_end(text, pos, final)
        if char == "\\":
            if pos + 1 == len(text) and not final:
                return None
            # Only Markdown punctuation is escapable; preserve ordinary slashes.
            if pos + 1 < len(text) and not text[pos + 1].isalnum() and not text[pos + 1].isspace():
                return pos + 2
            return pos + 1
        if char not in "`~\"'":
            return pos
        if char == "'" and pos and (text[pos - 1].isalnum() or text[pos - 1] == "_"):
            return pos  # Apostrophes in contractions are not quote openers.
        end = pos + 1
        if char in "`~":
            while end < len(text) and text[end] == char:
                end += 1
            if end == len(text) and not final:
                return None
            if end - pos >= 3 and self._line_start(text, pos):
                line_end = self._line_end(text, end, final)
                if line_end is None:
                    return None
                if char == "~" or "`" not in text[end:line_end]:
                    self._fence = (char, end - pos)
                    return line_end
            if char == "~":
                return end
        # Inline spans cannot cross a blank line. Do not commit to code until
        # an exact closing run exists (CommonMark unmatched-backtick semantics).
        key = (pos, char, end - pos)
        scan = self._waiting[1] if self._waiting and self._waiting[0] == key else end
        while scan < len(text):
            if text[scan] == "\n":
                next_pos = scan + 1
                while next_pos < len(text) and text[next_pos] in " \t\r":
                    next_pos += 1
                if next_pos == len(text) and not final:
                    self._waiting = (key, scan)
                    return None
                if next_pos < len(text) and text[next_pos] == "\n":
                    self._waiting = None
                    return end
            if char in "\"'" and text[scan] == "\\":
                if scan + 1 == len(text) and not final:
                    self._waiting = (key, scan)
                    return None
                scan += 2
                continue
            if text[scan] == char:
                close = scan + 1
                if char == "`":
                    while close < len(text) and text[close] == char:
                        close += 1
                    if close == len(text) and not final:
                        self._waiting = (key, scan)
                        return None
                if close - scan == end - pos:
                    self._waiting = None
                    return close
                scan = close
            else:
                scan += 1
        if not final:
            self._waiting = (key, scan)
            return None
        self._waiting = None
        return end
