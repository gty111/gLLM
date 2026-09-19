import base64
import binascii
import io
import json
import mimetypes
import time
from pathlib import PurePosixPath
from typing import Any, Dict, List, Tuple
from urllib.parse import unquote, unquote_to_bytes, urlparse
from urllib.request import Request, urlopen

from PIL import Image

from gllm.engine.async_llm import AsyncStream
from gllm.entrypoints.protocol import ChatCompletionRequest, ResponseRequest
from gllm.entrypoints.response_tools import chat_tools, output_tool_call, tool_specs
from gllm.entrypoints.serving_chat import chat_completion_generator
from gllm.tokenizers.tool_parsers import ToolParser
from gllm.tokenizers.reasoning import ThinkParser, split_reasoning_stream
from gllm.utils import build_usage, get_finish_reason, random_uuid

_MAX_INLINE_FILE_BYTES = 50 * 1024 * 1024
_TEXT_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".css",
    ".csv",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".log",
    ".md",
    ".py",
    ".rb",
    ".rs",
    ".sh",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}


def _data_url_bytes(value: str, param: str) -> Tuple[bytes, str | None]:
    """Decode an OpenAI ``file_data`` value (data URL or raw base64)."""
    mime_type = None
    payload = value
    if value.startswith("data:"):
        try:
            header, payload = value.split(",", 1)
        except ValueError as exc:
            raise ValueError(param, "Invalid data URL in `file_data`.") from exc
        metadata = header[5:].split(";")
        mime_type = metadata[0] or None
        if "base64" not in metadata[1:]:
            return unquote_to_bytes(payload), mime_type
    try:
        return base64.b64decode(payload, validate=True), mime_type
    except (binascii.Error, ValueError) as exc:
        raise ValueError(param, "`file_data` must be Base64 or a data URL.") from exc


def _download_file(url: str, param: str) -> Tuple[bytes, str | None, str | None]:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(param, "`file_url` must use http or https.")
    request = Request(url, headers={"User-Agent": "gLLM/Responses"})
    try:
        with urlopen(request, timeout=30) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > _MAX_INLINE_FILE_BYTES:
                raise ValueError(param, "Input files are limited to 50 MB.")
            data = response.read(_MAX_INLINE_FILE_BYTES + 1)
            mime_type = response.headers.get_content_type()
            disposition = response.headers.get_filename()
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(param, f"Unable to download `file_url`: {exc}") from exc
    if len(data) > _MAX_INLINE_FILE_BYTES:
        raise ValueError(param, "Input files are limited to 50 MB.")
    url_name = PurePosixPath(unquote(parsed.path)).name or None
    return data, mime_type, disposition or url_name


def _decode_text_file(data: bytes, param: str) -> str:
    encodings = (
        ("utf-16", "utf-8-sig")
        if data.startswith((b"\xff\xfe", b"\xfe\xff"))
        else ("utf-8-sig",)
    )
    for encoding in encodings:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            pass
    if b"\x00" in data[:4096]:
        raise ValueError(param, "This binary document type is not supported yet.")
    return data.decode("latin-1")


def _input_file_parts(part: Dict[str, Any], param: str) -> List[Dict[str, Any]]:
    sources = [key for key in ("file_data", "file_url", "file_id") if part.get(key)]
    if len(sources) != 1:
        raise ValueError(
            param,
            "`input_file` requires exactly one of `file_data`, `file_url`, or `file_id`.",
        )
    source = sources[0]
    if source == "file_id":
        raise ValueError(
            f"{param}.file_id",
            "File IDs require the Files API, which this server does not implement yet; use `file_data` or `file_url`.",
        )

    filename = part.get("filename")
    if source == "file_data":
        data, mime_type = _data_url_bytes(part[source], f"{param}.file_data")
    else:
        data, mime_type, downloaded_name = _download_file(
            part[source], f"{param}.file_url"
        )
        filename = filename or downloaded_name
    if len(data) > _MAX_INLINE_FILE_BYTES:
        raise ValueError(param, "Input files are limited to 50 MB.")

    filename = filename or "input_file"
    extension = PurePosixPath(filename).suffix.lower()
    guessed_type = mimetypes.guess_type(filename)[0]
    mime_type = mime_type or guessed_type or "application/octet-stream"
    if mime_type == "application/octet-stream" and data.startswith(b"%PDF-"):
        mime_type = "application/pdf"

    if mime_type.startswith("image/"):
        try:
            image = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as exc:
            raise ValueError(param, f"Unable to decode image file: {exc}") from exc
        return [{"type": "image", "image": image}]
    if mime_type == "application/pdf" or extension == ".pdf":
        raise ValueError(
            param,
            "PDF file inputs are not supported by this server.",
        )
    if mime_type.startswith("text/") or extension in _TEXT_EXTENSIONS:
        text = _decode_text_file(data, param)
        return [{"type": "text", "text": f"\n[File: {filename}]\n{text}\n"}]
    raise ValueError(
        param,
        f"Unsupported input file type {mime_type!r}; use an image or a text/code/CSV file.",
    )


def _response_content_parts(
    content: List[Any], param: str
) -> Tuple[List[Dict[str, Any]], bool]:
    parts: List[Dict[str, Any]] = []
    has_media = False
    for part_index, part in enumerate(content):
        part_param = f"{param}.{part_index}"
        if not isinstance(part, dict):
            raise ValueError(part_param, "Content parts must be objects.")
        part_type = part.get("type")
        if part_type in ("input_text", "output_text", "text"):
            parts.append({"type": "text", "text": part.get("text", "")})
        elif part_type == "input_image":
            image_url = part.get("image_url")
            file_id = part.get("file_id")
            if bool(image_url) == bool(file_id):
                raise ValueError(
                    part_param,
                    "`input_image` requires exactly one of `image_url` or `file_id`.",
                )
            if file_id:
                raise ValueError(
                    f"{part_param}.file_id",
                    "Image file IDs require the Files API; use `image_url` with a URL or data URL.",
                )
            # This is the gLLM-native media shape. Do not route Responses input
            # through the Chat Completions ``image_url`` wire representation.
            parts.append({"type": "image", "image": image_url})
            has_media = True
        elif part_type == "input_file":
            file_parts = _input_file_parts(part, part_param)
            parts.extend(file_parts)
            has_media = has_media or any(p["type"] == "image" for p in file_parts)
        elif part_type == "input_audio":
            raise ValueError(part_param, "Audio input is not supported yet.")
        else:
            raise ValueError(
                part_param, f"Unsupported Responses content type: {part_type!r}."
            )
    return parts, has_media


def response_input_to_messages(request: ResponseRequest) -> List[Dict[str, Any]]:
    """Parse Responses input into the gLLM-native chat/media representation."""
    messages: List[Dict[str, Any]] = []
    if request.instructions:
        if not isinstance(request.instructions, str):
            raise ValueError("instructions", "Only string instructions are supported.")
        messages.append({"role": "developer", "content": request.instructions})

    if isinstance(request.input, str):
        messages.append({"role": "user", "content": request.input})
        return messages

    for index, item in enumerate(request.input):
        param = f"input.{index}"
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            raise ValueError(param, "Input items must be strings or objects.")
        item_type = item.get("type")
        if item_type == "reasoning":
            # Allow stateless replay of our own output. Previous-turn thoughts
            # are not user messages and must not be folded into answer text.
            continue
        if item_type in ("function_call", "custom_tool_call"):
            call_id = item.get("call_id") or item.get("id")
            name = item.get("name")
            if item.get("namespace"):
                name = f"{item['namespace']}.{name}"
            arguments = item.get("arguments", "{}")
            if item_type == "custom_tool_call":
                if not isinstance(item.get("input"), str):
                    raise ValueError(param, "Custom tool input must be a string.")
                arguments = json.dumps({"input": item["input"]}, ensure_ascii=False)
            call = {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
            # Responses represents assistant text and calls as separate items.
            # Reassemble the assistant message before applying a chat template;
            # a separate message per call inserts spurious turn-end tokens.
            if messages and messages[-1]["role"] == "assistant":
                messages[-1].setdefault("tool_calls", []).append(call)
            else:
                messages.append({
                    "role": "assistant", "content": None, "tool_calls": [call],
                })
            continue
        if item_type in ("function_call_output", "custom_tool_call_output"):
            output = item.get("output", "")
            if isinstance(output, list):
                parts, has_media = _response_content_parts(output, f"{param}.output")
                if has_media:
                    raise ValueError(param, "Multimodal tool outputs are not supported yet.")
                output = "".join(part["text"] for part in parts)
            elif not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id"),
                    "content": output,
                }
            )
            continue
        if item_type not in (None, "message") or "role" not in item:
            raise ValueError(param, f"Unsupported Responses input type: {item_type!r}.")
        content = item.get("content", "")
        if isinstance(content, list):
            parts, has_media = _response_content_parts(content, f"{param}.content")
            content = parts if has_media else "".join(part["text"] for part in parts)
        elif not isinstance(content, str):
            raise ValueError(f"{param}.content", "Message content must be text.")
        messages.append({"role": item["role"], "content": content})
    # Codex compaction can retain a user message before its developer context.
    # Templates such as Qwen accept instructions only at the beginning. Collect
    # instruction messages in their original order, leaving conversation and
    # tool-call ordering unchanged.
    instructions = []
    conversation = []
    for message in messages:
        if message["role"] in ("system", "developer"):
            if not isinstance(message["content"], str):
                raise ValueError("input", "Instruction messages must contain text only.")
            instructions.append(message)
        else:
            conversation.append(message)
    if instructions:
        messages = [{
            "role": instructions[0]["role"],
            "content": "\n\n".join(message["content"] for message in instructions),
        }, *conversation]
    if not any(message.get("role") == "user" for message in messages):
        raise ValueError(
            "input",
            "Stateless Responses requests must include a user message; previous_response_id is not supported.",
        )
    return messages


def response_tools_to_chat(tools):
    return chat_tools(tools)


def make_chat_request(request: ResponseRequest) -> ChatCompletionRequest:
    effort = (request.reasoning or {}).get("effort")
    return ChatCompletionRequest.model_validate(
        {
            "model": request.model,
            "messages": response_input_to_messages(request),
            "max_completion_tokens": request.max_output_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "tools": response_tools_to_chat(request.tools),
            "tool_choice": request.tool_choice,
            "parallel_tool_calls": request.parallel_tool_calls,
            "reasoning_effort": effort,
            "request_id": random_uuid(),
        }
    )


def _usage(chat_response):
    usage = chat_response.usage
    return {
        "input_tokens": usage.prompt_tokens,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens": usage.completion_tokens or 0,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": usage.total_tokens,
    }


def _base_response(request: ResponseRequest, *, response_id: str, created_at: int):
    effort = (request.reasoning or {}).get("effort")
    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": "in_progress",
        "background": False,
        "completed_at": None,
        "error": None,
        "incomplete_details": None,
        "instructions": request.instructions,
        "max_output_tokens": request.max_output_tokens,
        "max_tool_calls": request.max_tool_calls,
        "model": request.model,
        "output": [],
        "parallel_tool_calls": bool(request.parallel_tool_calls),
        "previous_response_id": request.previous_response_id,
        "reasoning": {"effort": effort, "summary": None},
        "store": False,
        "temperature": request.temperature,
        "text": request.text or {"format": {"type": "text"}},
        "tool_choice": request.tool_choice or ("auto" if request.tools else "none"),
        "tools": request.tools or [],
        "top_p": request.top_p,
        "truncation": request.truncation or "disabled",
        "usage": None,
        "user": request.user,
        "metadata": request.metadata or {},
        "service_tier": request.service_tier,
    }


def _reasoning_item(text, item_id, status="completed"):
    # Native model thoughts are reasoning text, not a generated summary.
    return {"id": item_id, "type": "reasoning", "status": status,
            "summary": [], "content": [{"type": "reasoning_text", "text": text}]}


def _response_completion_fields(finish_reason, reasoning_parser, has_output):
    """Resolve terminal status independently of whether reasoning is exposed."""
    status = "completed"
    error = None
    incomplete_details = None
    if finish_reason == "length":
        status = "incomplete"
        incomplete_details = {"reason": "max_output_tokens"}
    elif reasoning_parser is not None and reasoning_parser.started:
        if reasoning_parser.state != "content":
            error = {
                "code": "server_error",
                "message": "Generation ended before the reasoning block was closed.",
            }
        elif not has_output:
            error = {
                "code": "server_error",
                "message": "Generation ended after reasoning without an answer or tool call.",
            }
        if error is not None:
            status = "failed"
    return {
        "status": status,
        "completed_at": int(time.time()) if status == "completed" else None,
        "error": error,
        "incomplete_details": incomplete_details,
    }


async def response_completion_generator(
    stream: AsyncStream,
    request: ResponseRequest,
    chat_request: ChatCompletionRequest,
    tool_parser: ToolParser = None,
    reasoning_parser: ThinkParser = None,
):
    chat_response = await chat_completion_generator(
        stream, chat_request, tool_parser, reasoning_parser
    )
    response_id = f"resp_{random_uuid()}"
    created_at = int(time.time())
    response = _base_response(request, response_id=response_id, created_at=created_at)
    choice = chat_response.choices[0]
    output = []
    if choice.message.reasoning_content and (request.reasoning or {}).get("summary") != "none":
        status = "incomplete" if reasoning_parser.state != "content" else "completed"
        output.append(_reasoning_item(
            choice.message.reasoning_content, f"rs_{random_uuid()}", status
        ))
    started_reasoning = reasoning_parser is not None and reasoning_parser.started
    if choice.message.content or (
        not choice.message.tool_calls and not output and not started_reasoning
    ):
        output.append(
            {
                "id": f"msg_{random_uuid()}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "logprobs": [],
                        "text": choice.message.content or "",
                    }
                ],
            }
        )
    if choice.message.tool_calls:
        specs = tool_specs(request.tools)
        for tool_call in choice.message.tool_calls:
            output.append(output_tool_call(tool_call, specs))
    response.update(
        {
            **_response_completion_fields(
                get_finish_reason(stream.seq), reasoning_parser,
                bool((choice.message.content or "").strip() or choice.message.tool_calls),
            ),
            "output": output,
            "usage": _usage(chat_response),
        }
    )
    return response


def _sse(event: Dict[str, Any]):
    return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


async def response_stream_generator(
    stream: AsyncStream,
    request: ResponseRequest,
    chat_request: ChatCompletionRequest,
    tool_parser: ToolParser = None,
    reasoning_parser: ThinkParser = None,
):
    """Translate engine deltas directly into Responses API SSE events."""
    sequence = 0

    def event(event_type, **payload):
        nonlocal sequence
        value = {"type": event_type, "sequence_number": sequence, **payload}
        sequence += 1
        return value

    response_id = f"resp_{random_uuid()}"
    created_at = int(time.time())
    initial = _base_response(request, response_id=response_id, created_at=created_at)
    yield _sse(event("response.created", response=initial))
    yield _sse(event("response.in_progress", response=initial))

    parse_tools = (
        tool_parser is not None
        and bool(chat_request.tools)
        and chat_request.tool_choice != "none"
    )
    stream_parser = (
        tool_parser.stream_parser(chat_request.tools) if parse_tools else None
    )
    full_text = ""
    message_text = ""
    message_id = None
    message_index = None
    message_done = False
    outputs = []
    next_output_index = 0
    specs = tool_specs(request.tools)
    expose_reasoning = (request.reasoning or {}).get("summary") != "none"
    reasoning_id = None
    reasoning_index = None
    reasoning_text = ""
    reasoning_done = False

    def reasoning_events(text):
        nonlocal reasoning_id, reasoning_index, reasoning_text, next_output_index
        if not text or not expose_reasoning:
            return []
        events = []
        if reasoning_id is None:
            reasoning_id = f"rs_{random_uuid()}"
            reasoning_index = next_output_index
            next_output_index += 1
            events.append(_sse(event(
                "response.output_item.added", output_index=reasoning_index,
                item=_reasoning_item("", reasoning_id, "in_progress"),
            )))
        reasoning_text += text
        events.append(_sse(event(
            "response.reasoning_text.delta", output_index=reasoning_index,
            item_id=reasoning_id, content_index=0, delta=text,
        )))
        return events

    def finish_reasoning_events():
        nonlocal reasoning_done
        if reasoning_id is None or reasoning_done:
            return []
        reasoning_done = True
        status = "incomplete" if reasoning_parser.state != "content" else "completed"
        item = _reasoning_item(reasoning_text, reasoning_id, status)
        outputs.append(item)
        return [
            _sse(event("response.reasoning_text.done", output_index=reasoning_index,
                       item_id=reasoning_id, content_index=0, text=reasoning_text)),
            _sse(event("response.output_item.done", output_index=reasoning_index,
                       item=item)),
        ]

    def start_message_events():
        nonlocal message_id, message_index, next_output_index
        if message_id is not None:
            return []
        message_id = f"msg_{random_uuid()}"
        message_index = next_output_index
        next_output_index += 1
        added = {
            "id": message_id,
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        }
        empty_part = {
            "type": "output_text",
            "annotations": [],
            "logprobs": [],
            "text": "",
        }
        return [
            _sse(
                event(
                    "response.output_item.added",
                    output_index=message_index,
                    item=added,
                )
            ),
            _sse(
                event(
                    "response.content_part.added",
                    output_index=message_index,
                    item_id=message_id,
                    content_index=0,
                    part=empty_part,
                )
            ),
        ]

    def finish_message_events():
        nonlocal message_done
        if message_id is None or message_done:
            return []
        message_done = True
        part = {
            "type": "output_text",
            "annotations": [],
            "logprobs": [],
            "text": message_text,
        }
        item = {
            "id": message_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [part],
        }
        outputs.append(item)
        return [
            _sse(
                event(
                    "response.output_text.done",
                    output_index=message_index,
                    item_id=message_id,
                    content_index=0,
                    text=message_text,
                    logprobs=[],
                )
            ),
            _sse(
                event(
                    "response.content_part.done",
                    output_index=message_index,
                    item_id=message_id,
                    content_index=0,
                    part=part,
                )
            ),
            _sse(
                event(
                    "response.output_item.done",
                    output_index=message_index,
                    item=item,
                )
            ),
        ]

    # Without model-native tool markup the output is known to be a message, so
    # announce its item/part before waiting for the first generated token.
    if stream_parser is None and reasoning_parser is None:
        for wire_event in start_message_events():
            yield wire_event

    async for stream_item, reasoning, text, final in split_reasoning_stream(stream, reasoning_parser):
        for wire_event in reasoning_events(reasoning):
            yield wire_event
        if reasoning_parser is not None and reasoning_parser.state == "content":
            for wire_event in finish_reasoning_events():
                yield wire_event
        if not text and not (final and stream_parser is not None):
            continue
        full_text += text
        if stream_parser is None:
            deltas = [text]
            tool_deltas = []
        else:
            parsed_deltas = []
            # One engine delta may finish both a text prefix and one or more
            # tool-call blocks. Drain every newly available parser delta.
            while True:
                parsed = stream_parser.process(full_text, final=final)
                if parsed is None:
                    break
                parsed_deltas.append(parsed)
            deltas = [delta.content for delta in parsed_deltas if delta.content]
            tool_deltas = [
                tool_call
                for delta in parsed_deltas
                for tool_call in (delta.tool_calls or [])
            ]

        for text_delta in deltas:
            for wire_event in start_message_events():
                yield wire_event
            message_text += text_delta
            yield _sse(
                event(
                    "response.output_text.delta",
                    output_index=message_index,
                    item_id=message_id,
                    content_index=0,
                    delta=text_delta,
                    logprobs=[],
                )
            )

        for tool_call in tool_deltas:
            # Natural-language content, when present, precedes tool-call output
            # items and must be finalized before the next item is announced.
            for wire_event in finish_message_events():
                yield wire_event
            function = tool_call.function
            if function is None or function.name is None:
                continue
            output_index = next_output_index
            next_output_index += 1
            try:
                item = output_tool_call(tool_call, specs)
            except ValueError as exc:
                # Embedded Response errors use the SDK's closed error-code
                # enum, unlike the top-level HTTP error envelope.
                failed = dict(
                    initial,
                    status="failed",
                    output=outputs,
                    error={"code": "server_error", "message": str(exc)},
                )
                yield _sse(event("response.failed", response=failed))
                return
            custom = item["type"] == "custom_tool_call"
            field = "input" if custom else "arguments"
            arguments = item[field]
            prefix = "response.custom_tool_call_input" if custom else "response.function_call_arguments"
            added = {**item, field: ""}
            if not custom:
                added["status"] = "in_progress"
            yield _sse(
                event(
                    "response.output_item.added",
                    output_index=output_index,
                    item=added,
                )
            )
            if arguments:
                yield _sse(
                    event(
                        f"{prefix}.delta",
                        output_index=output_index,
                        item_id=item["id"],
                        delta=arguments,
                    )
                )
            yield _sse(
                event(
                    f"{prefix}.done",
                    output_index=output_index,
                    item_id=item["id"],
                    **{field: arguments},
                )
            )
            yield _sse(
                event(
                    "response.output_item.done",
                    output_index=output_index,
                    item=item,
                )
            )
            outputs.append(item)

    for wire_event in finish_reasoning_events():
        yield wire_event
    # Preserve ordinary empty generations, but never fabricate an answer for
    # reasoning-only output (including when the client hides reasoning).
    started_reasoning = reasoning_parser is not None and reasoning_parser.started
    if not outputs and message_id is None and not started_reasoning:
        for wire_event in start_message_events():
            yield wire_event
    for wire_event in finish_message_events():
        yield wire_event

    finish_reason = get_finish_reason(stream.seq)
    usage = build_usage(stream.seq)
    final = dict(initial)
    final.update(
        {
            **_response_completion_fields(
                finish_reason, reasoning_parser,
                bool(message_text.strip()) or any(
                    item["type"] in ("function_call", "custom_tool_call")
                    for item in outputs
                ),
            ),
            "output": outputs,
            "usage": {
                "input_tokens": usage.prompt_tokens,
                "input_tokens_details": {
                    "cached_tokens": 0,
                    "cache_write_tokens": 0,
                },
                "output_tokens": usage.completion_tokens or 0,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": usage.total_tokens,
            },
        }
    )
    terminal_event = f"response.{final['status']}"
    yield _sse(event(terminal_event, response=final))
