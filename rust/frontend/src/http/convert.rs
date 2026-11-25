//! OpenAI request JSON -> `vllm-chat` / `vllm-text` request types.

use serde_json::{Map, Value};
use vllm_chat::{
    AssistantContentBlock, AssistantToolCall, ChatContent, ChatContentPart, ChatMessage, ChatTool,
    ChatToolChoice, SamplingParams,
};
use vllm_engine_core_client::protocol::structured_outputs::{
    StructuredOutputConstraint, StructuredOutputsParams,
};

use super::ApiError;
use super::types::Sampling;

/// Accepted for compatibility and ignored.
const IGNORED: &[&str] = &["user", "service_tier", "metadata", "store", "return_tokens_as_token_ids", "modalities"];

pub fn reject_extra(extra: &Map<String, Value>) -> Result<(), ApiError> {
    for (name, value) in extra {
        if value.is_null() || IGNORED.contains(&name.as_str()) {
            continue;
        }
        if name == "modalities" {
            continue;
        }
        return Err(ApiError::unsupported(name));
    }
    Ok(())
}

fn invalid(message: impl Into<String>, param: &str) -> ApiError {
    ApiError::invalid(message, Some(param), "invalid_request")
}

fn content(value: &Value, param: &str) -> Result<ChatContent, ApiError> {
    match value {
        Value::Null => Ok(ChatContent::Text(String::new())),
        Value::String(s) => Ok(ChatContent::Text(s.clone())),
        Value::Array(parts) => parts
            .iter()
            .map(|part| {
                let url = |key: &str| -> Result<String, ApiError> {
                    match &part[key] {
                        Value::String(s) => Ok(s.clone()),
                        Value::Object(o) => o
                            .get("url")
                            .and_then(Value::as_str)
                            .map(str::to_owned)
                            .ok_or_else(|| invalid(format!("`{key}.url` is required"), param)),
                        _ => Err(invalid(format!("`{key}` is required"), param)),
                    }
                };
                match part["type"].as_str() {
                    Some("text") => Ok(ChatContentPart::text(part["text"].as_str().unwrap_or_default())),
                    Some("image_url") => Ok(ChatContentPart::ImageUrl { image_url: url("image_url")?, detail: None, uuid: None }),
                    Some("video_url") => Ok(ChatContentPart::VideoUrl { video_url: url("video_url")?, uuid: None }),
                    Some(other) => Err(invalid(format!("Unsupported content part type `{other}`."), param)),
                    None => Err(invalid("Content parts need a `type`.", param)),
                }
            })
            .collect::<Result<Vec<_>, _>>()
            .map(ChatContent::Parts),
        _ => Err(invalid("`content` must be a string or a list of parts.", param)),
    }
}

fn text_of(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts.iter().filter_map(|p| p["text"].as_str()).collect(),
        _ => String::new(),
    }
}

pub fn tool(value: &Value) -> Result<ChatTool, ApiError> {
    if value["type"].as_str().unwrap_or("function") != "function" {
        return Err(invalid("Only function tools are supported.", "tools"));
    }
    let f = if value.get("function").is_some() { &value["function"] } else { value };
    let name = f["name"].as_str().ok_or_else(|| invalid("Tools need a `name`.", "tools"))?;
    Ok(ChatTool {
        name: name.to_owned(),
        description: f["description"].as_str().map(str::to_owned),
        parameters: f.get("parameters").cloned().unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}})),
        strict: f["strict"].as_bool(),
        defer_loading: None,
    })
}

pub fn message(value: &Value) -> Result<ChatMessage, ApiError> {
    let param = "messages";
    let role = value["role"].as_str().ok_or_else(|| invalid("Messages need a `role`.", param))?;
    let body = &value["content"];
    Ok(match role {
        "system" => ChatMessage::system(content(body, param)?),
        "developer" => ChatMessage::developer(content(body, param)?, None),
        "user" => ChatMessage::user(content(body, param)?),
        "assistant" => {
            let mut blocks = Vec::new();
            let reasoning = value.get("reasoning_content").or_else(|| value.get("reasoning"));
            if let Some(text) = reasoning.and_then(Value::as_str).filter(|t| !t.is_empty()) {
                blocks.push(AssistantContentBlock::Reasoning { text: text.to_owned() });
            }
            let text = text_of(body);
            if !text.is_empty() {
                blocks.push(AssistantContentBlock::Text { text });
            }
            for call in value["tool_calls"].as_array().into_iter().flatten() {
                let f = &call["function"];
                let arguments = match &f["arguments"] {
                    Value::String(s) if !s.is_empty() => s.clone(),
                    Value::Object(_) => f["arguments"].to_string(),
                    _ => "{}".to_owned(),
                };
                blocks.push(AssistantContentBlock::ToolCall(AssistantToolCall {
                    id: call["id"].as_str().unwrap_or_default().to_owned(),
                    name: f["name"].as_str().unwrap_or_default().to_owned(),
                    arguments,
                }));
            }
            ChatMessage::assistant_blocks(blocks)
        }
        "tool" => ChatMessage::tool_response(
            content(body, param)?,
            value["tool_call_id"].as_str().unwrap_or_default(),
        ),
        "function" => return Err(invalid("Function messages are not supported.", param)),
        other => ChatMessage::Custom { role: other.to_owned(), content: content(body, param)? },
    })
}

pub fn tool_choice(value: &Value) -> Result<ChatToolChoice, ApiError> {
    match value {
        Value::String(s) => match s.as_str() {
            "auto" => Ok(ChatToolChoice::Auto),
            "none" => Ok(ChatToolChoice::None),
            "required" => Ok(ChatToolChoice::Required),
            _ => Err(invalid(format!("Unknown tool_choice `{s}`."), "tool_choice")),
        },
        Value::Object(_) => {
            let name = value["function"]["name"]
                .as_str()
                .or_else(|| value["name"].as_str())
                .ok_or_else(|| invalid("Named tool_choice needs a function name.", "tool_choice"))?;
            Ok(ChatToolChoice::Function { name: name.to_owned() })
        }
        _ => Err(invalid("Invalid tool_choice.", "tool_choice")),
    }
}

/// `response_format` -> structured-output constraint.
pub fn response_format(value: Option<&Value>) -> Result<Option<StructuredOutputsParams>, ApiError> {
    let Some(fmt) = value else { return Ok(None) };
    let param = "response_format";
    let constraint = match fmt["type"].as_str() {
        None | Some("text") => return Ok(None),
        Some("json_object") => StructuredOutputConstraint::JsonObject,
        Some("json_schema") => {
            let spec = &fmt["json_schema"];
            let schema = spec.get("schema").or_else(|| spec.get("json_schema")).cloned();
            StructuredOutputConstraint::Json(schema.unwrap_or_else(|| serde_json::json!({})))
        }
        Some("structural_tag") => {
            let mut tag = fmt.clone();
            if let Some(obj) = tag.as_object_mut() {
                obj.insert("type".into(), Value::String("structural_tag".into()));
            }
            StructuredOutputConstraint::StructuralTag(tag.to_string())
        }
        Some(other) => return Err(invalid(format!("Unsupported response_format type `{other}`."), param)),
    };
    Ok(Some(StructuredOutputsParams { constraint, options: Default::default() }))
}

pub fn sampling(s: &Sampling, max_tokens: Option<u32>) -> Result<SamplingParams, ApiError> {
    if s.n.is_some_and(|n| n != 1) {
        return Err(ApiError::unsupported("n"));
    }
    let top_k = match s.top_k {
        None => None,
        Some(k) if k <= 0 => Some(0),
        Some(k) => Some(k as u32),
    };
    if s.logit_bias.as_ref().is_some_and(|b| !b.is_empty()) {
        return Err(ApiError::unsupported("logit_bias"));
    }
    for (name, set) in [
        ("min_p", s.min_p.is_some_and(|v| v != 0.0)),
        ("presence_penalty", s.presence_penalty.is_some_and(|v| v != 0.0)),
        ("frequency_penalty", s.frequency_penalty.is_some_and(|v| v != 0.0)),
        ("seed", s.seed.is_some()),
        ("min_tokens", s.min_tokens.is_some_and(|v| v != 0)),
    ] {
        if set {
            return Err(ApiError::unsupported(name));
        }
    }
    Ok(SamplingParams {
        temperature: s.temperature,
        top_p: s.top_p,
        top_k,
        repetition_penalty: s.repetition_penalty,
        max_tokens,
        ignore_eos: s.ignore_eos,
        stop_token_ids: s.stop_token_ids.clone(),
        prompt_logprobs: s.prompt_logprobs,
        ..SamplingParams::default()
    })
}

pub fn decode_options(s: &Sampling) -> vllm_text::output::TextDecodeOptions {
    vllm_text::output::TextDecodeOptions {
        skip_special_tokens: s.skip_special_tokens,
        include_stop_str_in_output: s.include_stop_str_in_output,
        stop_strings: s.stop.clone().map(|s| s.into_vec()).filter(|v| !v.is_empty()),
        min_tokens: 0,
        detokenize: true,
    }
}
