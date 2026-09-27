"""Responses tool results keep media, call identity and content ordering."""

import base64
import io
import os

import pytest
from PIL import Image

from gllm.entrypoints.protocol import ResponseRequest
from gllm.entrypoints.serving_responses import make_chat_request
from gllm.multimodal.mixin import MmMixin
from gllm.tokenizers.tool_parsers import normalize_chat_template_messages


@pytest.fixture
def image_url():
    data = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(data, format="PNG")
    return "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()


def request(output, custom=False):
    call = {"type": "custom_tool_call" if custom else "function_call",
            "call_id": "view-1", "name": "view_image"}
    call.update({"input": "plot.png"} if custom else {"arguments": '{"path":"plot.png"}'})
    return ResponseRequest(model="test", input=[
        {"role": "user", "content": "Describe the image returned by the tool."},
        call,
        {"type": "custom_tool_call_output" if custom else "function_call_output",
         "call_id": "view-1", "output": output},
    ])


@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("file", [False, True])
@pytest.mark.parametrize("text", [False, True])
def test_media_survives_chat_validation_and_extraction(image_url, custom, file, text):
    part = ({"type": "input_file", "filename": "plot.png", "file_data": image_url}
            if file else {"type": "input_image", "image_url": image_url, "detail": "high"})
    output = ([{"type": "input_text", "text": "Before"}, part,
               {"type": "input_text", "text": "After"}] if text else [part])
    req = request(output, custom)
    original = req.model_dump_json()
    chat = make_chat_request(req)
    normalize_chat_template_messages(chat.messages)
    tool = chat.messages[-1]
    assert tool["role"] == "tool"
    assert tool["tool_call_id"] == "view-1"
    parts = tool["content"]
    assert [p["type"] for p in parts] == (["text", "image", "text"] if text else ["image"])
    if text:
        assert parts[0]["text"] == "Before" and parts[-1]["text"] == "After"
    media = MmMixin.extract_modify_mm(None, chat.messages)
    assert media["video"] == [] and len(media["image"]) == 1
    if file:
        assert media["image"][0].getpixel((0, 0)) == (255, 0, 0)
    else:
        assert media["image"][0] == image_url
    assert req.model_dump_json() == original


@pytest.mark.parametrize("custom", [False, True])
def test_text_tool_results_remain_strings(custom):
    for output, expected in [("OK", "OK"), ([], ""),
                             ([{"type": "input_text", "text": "A"},
                               {"type": "input_text", "text": "B"}], "AB")]:
        tool = make_chat_request(request(output, custom)).messages[-1]
        assert tool["content"] == expected
        assert tool["tool_call_id"] == "view-1"


@pytest.mark.parametrize("part,param", [
    ({"type": "input_image", "file_id": "file_123"}, "input.2.output.0.file_id"),
    ({"type": "input_image"}, "input.2.output.0"),
    ({"type": "input_audio", "input_audio": {"data": "AAAA"}}, "input.2.output.0"),
    ({"type": "unknown"}, "input.2.output.0"),
])
def test_unsupported_parts_keep_precise_errors(part, param):
    with pytest.raises(ValueError) as error:
        make_chat_request(request([part]))
    assert error.value.args[0] == param


@pytest.mark.parametrize("custom", [False, True])
def test_qwen_template_keeps_images_inside_tool_response(image_url, custom):
    path = os.environ.get("GLLM_TEST_QWEN_TOKENIZER")
    if not path:
        pytest.skip("Set GLLM_TEST_QWEN_TOKENIZER to a local Qwen checkpoint")
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    output = [{"type": "input_text", "text": "Before"},
              {"type": "input_image", "image_url": image_url},
              {"type": "input_text", "text": "After"}]
    chat = make_chat_request(request(output, custom))
    normalize_chat_template_messages(chat.messages)
    for message in chat.messages:
        if message.get("content") is None:
            message["content"] = ""
    rendered = processor.apply_chat_template(chat.messages, tokenize=False, add_generation_prompt=True)
    tool_text = rendered.split("<tool_response>", 1)[1].split("</tool_response>", 1)[0]
    assert tool_text.index("Before") < tool_text.index("<|image_pad|>") < tool_text.index("After")
    encoded = processor.apply_chat_template(chat.messages, tokenize=True, add_generation_prompt=True)
    token_ids = encoded[0]
    assert token_ids.count(processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")) > 0
