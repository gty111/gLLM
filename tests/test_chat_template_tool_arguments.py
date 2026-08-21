from gllm.entrypoints.protocol import ChatCompletionRequest
from gllm.tokenizers.tool_parsers import (
    normalize_chat_template_messages,
    normalize_chat_template_tool_arguments,
)


def test_openai_tool_argument_json_is_decoded_for_chat_templates():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "calculate",
                        "arguments": '{"expression":"1+2"}',
                    },
                }
            ],
        }
    ]

    normalize_chat_template_tool_arguments(messages)

    assert messages[0]["tool_calls"][0]["function"]["arguments"] == {
        "expression": "1+2"
    }


def test_invalid_or_non_object_tool_arguments_are_unchanged():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "bad", "arguments": "{"}},
                {"function": {"name": "list", "arguments": "[1, 2]"}},
            ],
        }
    ]

    normalize_chat_template_tool_arguments(messages)

    assert messages[0]["tool_calls"][0]["function"]["arguments"] == "{"
    assert messages[0]["tool_calls"][1]["function"]["arguments"] == "[1, 2]"


def test_pydantic_tool_call_iterator_is_materialized_and_decoded():
    request = ChatCompletionRequest.model_validate(
        {
            "model": "test",
            "messages": [
                {"role": "user", "content": "1+2"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "calculate",
                                "arguments": '{"expression":"1+2"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "3"},
            ],
        }
    )

    normalize_chat_template_tool_arguments(request.messages)

    tool_calls = request.messages[1]["tool_calls"]
    assert isinstance(tool_calls, list)
    assert tool_calls[0]["function"]["arguments"] == {"expression": "1+2"}


def test_developer_role_is_normalized_for_legacy_model_templates():
    messages = [{"role": "developer", "content": "Be concise"}]

    normalize_chat_template_messages(messages)

    assert messages == [{"role": "system", "content": "Be concise"}]
