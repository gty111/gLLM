# Reasoning and tool-call boundaries

The frontend treats protocol markers, literal examples, and tool payloads as
separate layers. Generation streams carry the character offsets of native
`<think>` / `</think>` tokens, including when an MTP step commits several tokens.
An identical string assembled from ordinary tokens is not a native boundary.
Legacy text-only streams use textual recognition as a compatibility fallback.

Both reasoning and Qwen tool parsing use the same literal scanner:

- Closed backtick spans, single/double quoted examples, escaped punctuation,
  fenced/indented code, explicit block-quote lines, and HTML comments protect
  markers from interpretation. Unclosed comments remain literal through EOF.
- Inline code requires an exact matching backtick run. It can span lines but
  not paragraphs. Unmatched inline delimiters are ordinary text, so a stray
  backtick cannot suppress a later complete tool call indefinitely.
- Fences start at the beginning of a line with up to three spaces. Matching
  fences must be at least as long, use the same character, and have only
  whitespace after them. An unclosed fence remains literal through EOF.
- Backticks inside inline code are not backslash-escaped; quotes do recognize
  escaped quotes. Apostrophes within words do not open a quoted example.
- Possible delimiter fragments and undecided inline examples are buffered.
  Published text is never retracted, and incomplete lookahead resumes where
  the prior delta ended.

Literal handling precedes protocol interpretation. Thus even a native end token
inside a closed code example remains reasoning text. Only the leading reasoning
block is split; ordinary answer text after its end is preserved. Tools are
recognized only outside protected literals and only after their native call
opener is confirmed. The complete argument payload is then parsed as one unit,
so marker-like strings inside arguments do not restart the outer scanner.

Incomplete reasoning continues to produce `response.failed`, or
`response.incomplete` when the output budget was exhausted. Actual malformed or
unfinished tool calls retain the existing explicit error handling. Neither case
is silently turned into a successful answer.

## Inherent ambiguity

If a model emits a native closing token outside literal syntax, it is a protocol
boundary. No parser can reliably infer that the model "meant to mention" it in
unquoted prose. Conversely, a well-formed call written outside literal syntax is
a tool invocation. Preserving native IDs and explicit quoting resolves ordinary
text collisions; guessing intentions from phrases or choosing the last marker
would create inconsistent streaming behavior and unsafe tool execution.

The regression suite covers text-only and native-token streams, every two-chunk
boundary for short examples, character deltas, batched MTP deltas, Unicode byte
fragments, both Qwen call formats, literal argument markers, unmatched inline
syntax, unclosed fences, EOF/budget failures, and reasoning-to-tool transitions.
Tests execute no generated commands. Set `GLLM_TEST_QWEN_TOKENIZER` to a local
checkpoint to also exercise its real tokenizer on CPU.
