# Adapted from
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/protocol/openai_api_protocol.py
import time
from typing import Any, Dict, List, Literal, Optional, Union

import openai.types.chat
import torch
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

# pydantic needs the TypedDict from typing_extensions
from typing_extensions import Required, TypedDict

from gllm.utils import random_uuid

_LONG_INFO = torch.iinfo(torch.long)


class CustomChatCompletionContentPartParam(TypedDict, total=False):
    __pydantic_config__ = ConfigDict(extra="allow")  # type: ignore

    type: Required[str]
    """The type of the content part."""


ChatCompletionContentPartParam = Union[
    openai.types.chat.ChatCompletionContentPartParam,
    CustomChatCompletionContentPartParam,
]


class CustomChatCompletionMessageParam(TypedDict, total=False):
    """Enables custom roles in the Chat Completion API."""

    role: Required[str]
    """The role of the message's author."""

    content: Union[str, List[ChatCompletionContentPartParam]]
    """The contents of the message."""

    name: str
    """An optional name for the participant.

    Provides the model information to differentiate between participants of the
    same role.
    """


ChatCompletionMessageParam = Union[
    openai.types.chat.ChatCompletionMessageParam, CustomChatCompletionMessageParam
]


class OpenAIBaseModel(BaseModel):
    # OpenAI API does not allow extra fields
    model_config = ConfigDict(extra="forbid")


class ErrorDetail(OpenAIBaseModel):
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[Union[str, int]] = None


class ErrorResponse(OpenAIBaseModel):
    """OpenAI error envelope returned for every HTTP/API error."""

    error: ErrorDetail


class ModelPermission(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"modelperm-{random_uuid()}")
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: bool = False


class ModelCard(OpenAIBaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "gllm"
    root: Optional[str] = None
    parent: Optional[str] = None
    max_model_len: Optional[int] = None
    permission: List[ModelPermission] = Field(default_factory=list)


class ModelList(OpenAIBaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


class UsageInfo(OpenAIBaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0
    prompt_tokens_details: Optional[Dict[str, int]] = None
    completion_tokens_details: Optional[Dict[str, int]] = None


class StructuralTag(OpenAIBaseModel):
    begin: str
    # schema is the field, but that causes conflicts with pydantic so
    # instead use structural_tag_schema with an alias
    structural_tag_schema: Optional[dict[str, Any]] = Field(
        default=None, alias="schema"
    )
    end: str


class StructuralTagResponseFormat(OpenAIBaseModel):
    type: Literal["structural_tag"]
    structures: list[StructuralTag]
    triggers: list[str]


class ResponseFormat(OpenAIBaseModel):
    type: Literal["text", "json_object"]


class JSONSchemaDefinition(OpenAIBaseModel):
    name: str
    description: Optional[str] = None
    schema_: Dict[str, Any] = Field(alias="schema")
    strict: Optional[bool] = None


class JSONSchemaResponseFormat(OpenAIBaseModel):
    type: Literal["json_schema"]
    json_schema: JSONSchemaDefinition


AnyResponseFormat = Union[
    ResponseFormat, JSONSchemaResponseFormat, StructuralTagResponseFormat
]


class StreamOptions(OpenAIBaseModel):
    include_usage: Optional[bool] = False
    include_obfuscation: Optional[bool] = None
    # gLLM extension retained for backwards compatibility.
    continuous_usage_stats: Optional[bool] = False


class FunctionDefinition(OpenAIBaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    strict: Optional[bool] = None


class ChatCompletionToolsParam(OpenAIBaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class ChatCompletionCustomToolParam(OpenAIBaseModel):
    type: Literal["custom"]
    custom: Dict[str, Any]


class ChatCompletionNamedFunction(OpenAIBaseModel):
    name: str


class ChatCompletionNamedToolChoiceParam(OpenAIBaseModel):
    function: ChatCompletionNamedFunction
    type: Literal["function"] = "function"


class ChatCompletionAllowedTools(OpenAIBaseModel):
    mode: Literal["auto", "required"]
    tools: List[ChatCompletionNamedToolChoiceParam]


class ChatCompletionAllowedToolChoiceParam(OpenAIBaseModel):
    type: Literal["allowed_tools"]
    allowed_tools: ChatCompletionAllowedTools


class ChatCompletionRequest(OpenAIBaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/chat/create
    messages: list[ChatCompletionMessageParam]
    model: str
    audio: Optional[Dict[str, Any]] = None
    frequency_penalty: Optional[float] = 0.0
    function_call: Optional[Union[Literal["none", "auto"], Dict[str, str]]] = None
    functions: Optional[List[FunctionDefinition]] = None
    logit_bias: Optional[dict[str, float]] = None
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = 0
    max_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        deprecated="max_tokens is deprecated in favor of the max_completion_tokens field",
    )
    max_completion_tokens: Optional[int] = Field(default=None, gt=0)
    n: Optional[int] = 1
    modalities: Optional[List[Literal["text", "audio"]]] = None
    metadata: Optional[Dict[str, str]] = None
    moderation: Optional[Dict[str, Any]] = None
    presence_penalty: Optional[float] = 0.0
    prediction: Optional[Dict[str, Any]] = None
    prompt_cache_key: Optional[str] = None
    prompt_cache_options: Optional[Dict[str, Any]] = None
    prompt_cache_retention: Optional[Literal["in_memory", "24h"]] = None
    reasoning_effort: Optional[
        Literal["none", "minimal", "low", "medium", "high", "xhigh"]
    ] = None
    response_format: Optional[AnyResponseFormat] = None
    safety_identifier: Optional[str] = None
    seed: Optional[int] = Field(None, ge=_LONG_INFO.min, le=_LONG_INFO.max)
    service_tier: Optional[
        Literal["auto", "default", "flex", "scale", "priority", "fast"]
    ] = None
    stop: Optional[Union[str, list[str]]] = []
    store: Optional[bool] = False
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    tools: Optional[
        list[Union[ChatCompletionToolsParam, ChatCompletionCustomToolParam]]
    ] = None
    tool_choice: Optional[
        Union[
            Literal["none"],
            Literal["auto"],
            Literal["required"],
            ChatCompletionNamedToolChoiceParam,
            ChatCompletionAllowedToolChoiceParam,
            Dict[str, Any],
        ]
    ] = None

    # Accepted for OpenAI schema compatibility; the model and tool parser
    # determine whether calls are emitted in parallel.
    parallel_tool_calls: Optional[bool] = True
    user: Optional[str] = None
    verbosity: Optional[Literal["low", "medium", "high"]] = None
    web_search_options: Optional[Dict[str, Any]] = None

    # --8<-- [start:chat-completion-sampling-params]
    # vLLM-extension sampling knobs accepted by the wire schema
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    ignore_eos: bool = False
    prompt_logprobs: Optional[int] = None
    # --8<-- [end:chat-completion-sampling-params]

    # --8<-- [start:chat-completion-extra-params]
    chat_template_kwargs: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Additional keyword args to pass to the template renderer. "
            "Will be accessible by the chat template."
        ),
    )
    request_id: str = Field(
        default_factory=lambda: f"{random_uuid()}",
        description=(
            "The request_id related to this request. If the caller does "
            "not set it, a random_uuid will be generated. This id is used "
            "through out the inference process and return in response."
        ),
    )
    return_tokens_as_token_ids: Optional[bool] = Field(
        default=None,
        description=(
            "If specified with 'logprobs', tokens are represented "
            " as strings of the form 'token_id:{token_id}' so that tokens "
            "that are not JSON-encodable can be identified."
        ),
    )

    # --8<-- [end:chat-completion-extra-params]

    @model_validator(mode="before")
    @classmethod
    def validate_stream_options(cls, values):
        if values.get("stream_options") is not None and not values.get("stream"):
            raise ValueError("stream_options can only be set if stream is true")
        return values

    @model_validator(mode="before")
    @classmethod
    def translate_legacy_functions(cls, values):
        """Accept the deprecated functions/function_call wire format."""
        if not isinstance(values, dict):
            return values
        if values.get("functions") is not None and values.get("tools") is None:
            values = dict(values)
            values["tools"] = [
                {"type": "function", "function": function}
                for function in values["functions"]
            ]
        if (
            values.get("function_call") is not None
            and values.get("tool_choice") is None
        ):
            values = dict(values)
            function_call = values["function_call"]
            values["tool_choice"] = (
                function_call
                if isinstance(function_call, str)
                else {"type": "function", "function": function_call}
            )
        return values

    @model_validator(mode="before")
    @classmethod
    def check_tool_choice(cls, data):
        if not isinstance(data, dict):
            return data
        tool_choice = data.get("tool_choice")
        # Auto permits a text response even when no tools are available.
        # Only choices that require a tool call need a nonempty tool list.
        if tool_choice not in (None, "none", "auto") and not data.get("tools"):
            raise ValueError("When using `tool_choice`, `tools` must be set.")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_logprobs(cls, data):
        if "top_logprobs" in data and data["top_logprobs"] is not None:
            if "logprobs" not in data or data["logprobs"] is False:
                raise ValueError(
                    "when using `top_logprobs`, `logprobs` must be set to true."
                )
            elif data["top_logprobs"] < 0:
                raise ValueError("`top_logprobs` must be a value a positive value.")
        return data


class CompletionRequest(OpenAIBaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/completions/create
    model: str
    prompt: Union[List[StrictInt], List[List[StrictInt]], str, List[str]]
    best_of: Optional[int] = None
    echo: Optional[bool] = False
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[int] = None
    prompt_logprobs: Optional[int] = None
    max_tokens: Optional[int] = Field(default=16, gt=0)
    n: int = 1
    presence_penalty: Optional[float] = 0.0
    seed: Optional[int] = Field(
        None, ge=torch.iinfo(torch.long).min, le=torch.iinfo(torch.long).max
    )
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    suffix: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    user: Optional[str] = None

    # doc: begin-completion-sampling-params
    # vLLM-extension sampling knobs accepted by the wire schema
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    ignore_eos: Optional[bool] = False
    # doc: end-completion-sampling-params

    # doc: begin-completion-extra-params
    response_format: Optional[ResponseFormat] = Field(
        default=None,
        description=(
            "Similar to chat completion, this parameter specifies the format of "
            "output. Only {'type': 'json_object'} or {'type': 'text' } is "
            "supported."
        ),
    )

    # doc: end-completion-extra-params

    @model_validator(mode="before")
    @classmethod
    def check_logprobs(cls, data):
        if (
            "logprobs" in data
            and data["logprobs"] is not None
            and not data["logprobs"] >= 0
        ):
            raise ValueError("if passed, `logprobs` must be a positive value.")
        return data

    @model_validator(mode="before")
    @classmethod
    def validate_stream_options(cls, data):
        if data.get("stream_options") and not data.get("stream"):
            raise ValueError("Stream options can only be defined when stream is True.")
        return data


class CompletionLogProbs(OpenAIBaseModel):
    text_offset: List[int] = Field(default_factory=list)
    token_logprobs: List[Optional[float]] = Field(default_factory=list)
    tokens: List[str] = Field(default_factory=list)
    top_logprobs: List[Optional[Dict[str, float]]] = Field(default_factory=list)


class CompletionResponseChoice(OpenAIBaseModel):
    index: int
    text: str
    logprobs: Optional[CompletionLogProbs] = None
    # Per prompt-token logprobs (index 0 is null since the first token has no
    # preceding context). Each entry mirrors the chat logprob shape:
    # {token, logprob, bytes, top_logprobs}.
    prompt_logprobs: Optional[List[Optional[Dict[str, Any]]]] = None
    finish_reason: Optional[str] = None
    stop_reason: Optional[Union[int, str]] = Field(
        default=None,
        description=(
            "The stop string or token id that caused the completion "
            "to stop, None if the completion finished for some other reason "
            "including encountering the EOS token"
        ),
    )


class CompletionResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{random_uuid()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo


class CompletionResponseStreamChoice(OpenAIBaseModel):
    index: int
    text: str
    logprobs: Optional[CompletionLogProbs] = None
    prompt_logprobs: Optional[List[Optional[Dict[str, Any]]]] = None
    finish_reason: Optional[str] = None
    stop_reason: Optional[Union[int, str]] = Field(
        default=None,
        description=(
            "The stop string or token id that caused the completion "
            "to stop, None if the completion finished for some other reason "
            "including encountering the EOS token"
        ),
    )


class CompletionStreamResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{random_uuid()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None)


class FunctionCall(OpenAIBaseModel):
    name: str
    arguments: str


class ToolCall(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-tool-{random_uuid()}")
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatMessage(OpenAIBaseModel):
    role: str
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    refusal: Optional[str] = None
    annotations: Optional[List[Dict[str, Any]]] = None
    audio: Optional[Dict[str, Any]] = None
    function_call: Optional[FunctionCall] = None
    tool_calls: Optional[List[ToolCall]] = None


class ChatCompletionLogProb(OpenAIBaseModel):
    token: str
    logprob: float = -9999.0
    bytes: Optional[List[int]] = None


class ChatCompletionLogProbsContent(ChatCompletionLogProb):
    top_logprobs: List[ChatCompletionLogProb] = Field(default_factory=list)


class ChatCompletionLogProbs(OpenAIBaseModel):
    content: Optional[List[ChatCompletionLogProbsContent]] = None


class ChatCompletionResponseChoice(OpenAIBaseModel):
    index: int
    message: ChatMessage
    logprobs: Optional[ChatCompletionLogProbs] = None
    prompt_logprobs: Optional[List[Optional[Dict[str, Any]]]] = None
    finish_reason: Optional[str] = None
    stop_reason: Optional[Union[int, str]] = None


class ChatCompletionResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo
    service_tier: Optional[str] = None
    system_fingerprint: Optional[str] = "gllm"


class DeltaFunctionCall(OpenAIBaseModel):
    # All optional: a streaming chunk may carry just the name (first chunk of a
    # call) or just an ``arguments`` fragment (subsequent chunks).
    name: Optional[str] = None
    arguments: Optional[str] = None


class DeltaToolCall(OpenAIBaseModel):
    # ``index`` ties argument fragments across chunks to the same tool call.
    index: int
    id: Optional[str] = None
    type: Optional[Literal["function"]] = None
    function: Optional[DeltaFunctionCall] = None


class DeltaMessage(OpenAIBaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    refusal: Optional[str] = None
    tool_calls: Optional[List[DeltaToolCall]] = None


class ChatCompletionResponseStreamChoice(OpenAIBaseModel):
    index: int
    delta: DeltaMessage
    logprobs: Optional[ChatCompletionLogProbs] = None
    prompt_logprobs: Optional[List[Optional[Dict[str, Any]]]] = None
    finish_reason: Optional[str] = None
    stop_reason: Optional[Union[int, str]] = None


class ChatCompletionStreamResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None)
    service_tier: Optional[str] = None
    system_fingerprint: Optional[str] = "gllm"
    obfuscation: Optional[str] = None


class ResponseRequest(OpenAIBaseModel):
    """Core current Responses API request supported by the local runtime.

    Complex input and tool unions are validated semantically by the endpoint so
    the wire schema can evolve without coupling the server to an SDK release.
    """

    model: str
    input: Union[str, List[Any]]
    # Client routing/telemetry hints (e.g. Codex); never part of the prompt.
    client_metadata: Optional[Dict[str, Any]] = None
    instructions: Optional[Union[str, List[Any]]] = None
    background: Optional[bool] = False
    conversation: Optional[Union[str, Dict[str, Any]]] = None
    include: Optional[List[str]] = None
    max_output_tokens: Optional[int] = Field(default=None, gt=0)
    max_tool_calls: Optional[int] = None
    metadata: Optional[Dict[str, str]] = None
    moderation: Optional[Dict[str, Any]] = None
    parallel_tool_calls: Optional[bool] = True
    previous_response_id: Optional[str] = None
    prompt: Optional[Dict[str, Any]] = None
    prompt_cache_key: Optional[str] = None
    prompt_cache_options: Optional[Dict[str, Any]] = None
    prompt_cache_retention: Optional[Literal["in_memory", "24h"]] = None
    reasoning: Optional[Dict[str, Any]] = None
    safety_identifier: Optional[str] = None
    service_tier: Optional[str] = None
    store: Optional[bool] = False
    stream: Optional[bool] = False
    stream_options: Optional[Dict[str, Any]] = None
    temperature: Optional[float] = None
    text: Optional[Dict[str, Any]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    top_logprobs: Optional[int] = None
    top_p: Optional[float] = None
    truncation: Optional[Literal["auto", "disabled"]] = "disabled"
    user: Optional[str] = None
