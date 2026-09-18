# Codex through the Responses API

gLLM exposes a stateless `/v1/responses` endpoint for Codex and other Responses
clients. Native Codex CLI 0.154.0 has been exercised with Qwen3.8-27B, including
command execution, `apply_patch`, and returning tool results to the model.

## Client configuration

With gLLM serving Qwen3.8-27B on port 8000, a Codex provider can use:

```toml
model = "Qwen3.8-27B"
model_provider = "gllm"
model_context_window = 262144
model_auto_compact_token_limit = 220000
model_reasoning_effort = "none"
model_reasoning_summary = "none"
web_search = "disabled"

[model_providers.gllm]
name = "Local gLLM"
base_url = "http://127.0.0.1:8000/v1"
wire_api = "responses"
requires_openai_auth = false
supports_websockets = false

[features]
enable_request_compression = false
```

The example assumes a 262,144-token server context. Set the client's context
window and compaction threshold to match the server's configured context limit.
The provider address must be reachable from the machine running Codex; loopback
addresses refer to that machine. Tool execution remains the client's responsibility.

## Supported tool flow

- Function tools and custom tools, including tools grouped in a namespace.
  Namespaces require `name`, `description`, and `tools`. Two namespaces may use
  the same leaf tool name; calls retain their namespace in the response and history.
- Custom text, regex, and Lark formats. The model sees a function with a string
  `input` argument; the endpoint returns `custom_tool_call` items and the
  corresponding `response.custom_tool_call_input.*` events.
- Stateless continuation with `function_call_output` or `custom_tool_call_output`.
  Send the original user message and accumulated history on each request. Tool
  outputs may be strings or arrays of text content parts.
- Streaming and non-streaming responses. Tool calls are published after their
  complete arguments have been parsed and validated.
- `tool_choice="auto"` with an empty tool list, allowing a text-only response
  after tool execution. Instruction/developer messages are collected in their
  original order into one leading instruction message. This also handles Codex
  compaction histories that place developer context after a retained user message;
  conversation and tool-call ordering remain unchanged.

`client_metadata` is accepted without adding it to the prompt.
`include=["reasoning.encrypted_content"]` is accepted, but this backend produces
no encrypted reasoning items.

## Limits

When `max_output_tokens` is omitted or null, the output budget is the configured
context length minus the tokenized input length, including the chat template
and image tokens. An explicit limit is preserved; requests that leave no output
space or whose input plus output budget exceeds the context window are rejected.
The budget includes reasoning, answer text, and tool-call tokens. EOS can end a
response before the budget is exhausted. Limits explicitly supplied by a proxy
still apply. Chat Completions uses the same remaining-context default; legacy
Completions retains its default of 16 tokens when `max_tokens` is omitted.

Grammar checks validate generated input **after generation**, rather than
constraining token sampling. Lark formats use Python Lark; the empty-line regex
forms used by Codex patches (`/(.*)/` and `/.*/`) are adapted to optional nonempty
terminals. Other grammar dialect differences are not translated.

Invalid or undeclared generated tool calls are not published. Non-streaming
requests return HTTP 500 with `invalid_tool_output`; streams terminate with a
`response.failed` event using the Responses error code `server_error` and a
diagnostic message. Invalid grammar definitions are rejected before inference.

Hosted tools, forced tool choices, stored responses, `previous_response_id`,
background requests, and multimodal tool results are not supported. Unsupported
capabilities return an error. This endpoint does not implement the complete
OpenAI Responses API.

## Regression checks

Run in a Linux gLLM environment with its dependencies installed:

```sh
python -m pytest -q tests/test_codex_responses.py tests/test_openai_protocol_compat.py tests/test_tool_parsers.py
```

These tests validate response objects and streaming events with the OpenAI Python
SDK, including failed custom calls, namespace collisions, Qwen3.8 XML markup,
text and regex tools, and Codex patch grammars.

Protocol reference: [OpenAI Responses streaming events](https://developers.openai.com/api/reference/resources/responses/streaming-events).

## Compaction and misleading overload errors

The 256K example configuration uses a 220,000-token auto-compaction threshold.
Instructions, tool definitions, and skill descriptions already consume context
before the first substantive user task. With a small configured window, such as
16K, reading project files may trigger compaction after only a few exchanges.
Reducing unused skills or raising
the server and client context limits together can leave more room for tasks.

If Codex reports high demand against this local provider, inspect the server log
for the underlying HTTP failure. In particular, older Responses adapters could
raise `System message must be at the beginning` after compaction. The adapter now
normalizes instruction placement, and other template errors return HTTP 400
`invalid_input` with the original diagnostic instead of an uncaught HTTP 500.
