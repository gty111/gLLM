//! `POST /v1/chat/completions`.

use std::convert::Infallible;
use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::{Stream, StreamExt as _};
use serde_json::{Value, json};
use vllm_chat::{
    AssistantBlockKind, AssistantContentBlock, ChatEvent, ChatOptions, ChatRequest, ChatToolChoice,
    FinishReason, ResolvedToolContext,
};
use vllm_text::{DecodedLogprobs, DecodedPromptLogprobs};

use super::types::{ChatCompletionRequest, PromptLogprobEntry, PromptTopLogprob, TokenLogprobOut, TopLogprob, Usage};
use super::{ApiError, AppState, convert, unix_now};

pub fn token_logprobs(lp: &DecodedLogprobs) -> Vec<TokenLogprobOut> {
    lp.positions
        .iter()
        .filter_map(|pos| {
            let (first, rest) = pos.entries.split_first()?;
            Some(TokenLogprobOut {
                token: first.token.clone(),
                logprob: first.logprob,
                bytes: first.token.as_bytes().to_vec(),
                top_logprobs: rest
                    .iter()
                    .map(|e| TopLogprob { token: e.token.clone(), logprob: e.logprob, bytes: e.token.as_bytes().to_vec() })
                    .collect(),
            })
        })
        .collect()
}

/// `[null, entry, ...]`, one entry per prompt token.
pub fn prompt_logprobs(lp: &DecodedPromptLogprobs) -> Vec<Option<PromptLogprobEntry>> {
    let mut out = Vec::with_capacity(lp.scored_positions.len() + 1);
    out.push(None);
    for pos in &lp.scored_positions {
        let Some((first, rest)) = pos.entries.split_first() else {
            out.push(None);
            continue;
        };
        out.push(Some(PromptLogprobEntry {
            token_id: first.token_id,
            token: first.token.clone(),
            logprob: first.logprob,
            bytes: first.token.as_bytes().to_vec(),
            top_logprobs: rest
                .iter()
                .map(|e| PromptTopLogprob {
                    token_id: e.token_id,
                    token: e.token.clone(),
                    logprob: e.logprob,
                    bytes: e.token.as_bytes().to_vec(),
                })
                .collect(),
        }));
    }
    out
}

pub fn finish_reason_str(reason: &FinishReason, tool_calls: bool) -> &'static str {
    match reason {
        FinishReason::Stop(_) if tool_calls => "tool_calls",
        FinishReason::Stop(_) => "stop",
        FinishReason::Length => "length",
        FinishReason::Abort => "abort",
        FinishReason::Error(_) => "error",
        FinishReason::Repetition(_) => "repetition",
    }
}

/// A request rejected by the worker (e.g. it can never fit the KV cache).
pub fn finish_error(reason: &FinishReason) -> Option<ApiError> {
    let FinishReason::Error(err) = reason else { return None };
    let (message, code) = match err {
        Some(e) => (e.message.clone(), e.code.clone()),
        None => ("request failed".to_owned(), "internal_error".to_owned()),
    };
    Some(ApiError::invalid(message, None, &code))
}

/// Map a submit-time error from the chat/text/llm stack.
pub fn submit_error(err: impl std::error::Error + 'static) -> ApiError {
    let mut source: Option<&(dyn std::error::Error + 'static)> = Some(&err);
    while let Some(e) = source {
        if let Some(llm) = e.downcast_ref::<vllm_llm::Error>() {
            match llm {
                vllm_llm::Error::Unavailable { message } => {
                    let code = if message.starts_with(crate::sleep::WAKE_FAILED) { "wake_up_failed" } else { "service_unavailable" };
                    return ApiError::unavailable(message.clone(), code);
                }
                vllm_llm::Error::Unsupported { field } => return ApiError::unsupported(field),
                _ => {}
            }
        }
        source = e.source();
    }
    let mut message = err.to_string();
    let mut source = err.source();
    while let Some(e) = source {
        message = format!("{message}: {e}");
        source = e.source();
    }
    ApiError::invalid(message, None, "invalid_request")
}

/// Encoder disaggregation: the LM renders the text template with one
/// placeholder token per media item (the encoders process the pixels), so
/// media parts become their placeholder text and the references are returned
/// in prompt order. Qwen-VL placeholders, the families disaggregation serves.
fn skeleton_media(messages: &mut [Value]) -> Vec<(String, String)> {
    let mut items = Vec::new();
    for message in messages {
        let Some(parts) = message.get_mut("content").and_then(Value::as_array_mut) else { continue };
        for part in parts {
            let (modality, key, placeholder) = match part["type"].as_str() {
                Some("image_url") => ("image", "image_url", "<|vision_start|><|image_pad|><|vision_end|>"),
                Some("video_url") => ("video", "video_url", "<|vision_start|><|video_pad|><|vision_end|>"),
                _ => continue,
            };
            let url = match &part[key] {
                Value::String(s) => s.clone(),
                v => v["url"].as_str().unwrap_or_default().to_owned(),
            };
            items.push((modality.to_owned(), url));
            *part = json!({"type": "text", "text": placeholder});
        }
    }
    items
}

pub(super) struct Prepared {
    pub request: ChatRequest,
    pub stream: bool,
    include_usage: bool,
    continuous_usage: bool,
    logprobs: bool,
}

/// Reasoning controls go to the renderer unchanged: effort "none" switches
/// thinking off, any other effort (a name, or DeepSeek V4.1's 1-100 budget)
/// passes through, and models with a native renderer apply their own defaults.
fn template_kwargs(state: &AppState, body: &ChatCompletionRequest) -> std::collections::HashMap<String, Value> {
    let mut kwargs = body.chat_template_kwargs.clone().unwrap_or_default();
    if state.legacy_v4_effort {
        legacy_v4_effort(body, &mut kwargs);
        return kwargs;
    }
    match &body.reasoning_effort {
        Some(Value::String(effort)) if effort == "none" => {
            kwargs.entry("enable_thinking".into()).or_insert(Value::Bool(false));
            kwargs.entry("thinking".into()).or_insert(Value::Bool(false));
        }
        Some(effort) if !effort.is_null() => {
            kwargs.entry("reasoning_effort".into()).or_insert(effort.clone());
        }
        _ => {}
    }
    kwargs
}

/// The original DeepSeek V4 encoder: thinking is on unless switched off, and
/// only effort `max` adds a prefix (the renderer's `high` text); every other
/// effort renders the plain prompt (the renderer's `low`).
fn legacy_v4_effort(body: &ChatCompletionRequest, kwargs: &mut std::collections::HashMap<String, Value>) {
    let toggle = ["thinking", "enable_thinking"].iter().find_map(|k| kwargs.get(*k)).and_then(Value::as_bool);
    if toggle == Some(false) {
        return;
    }
    let effort = body.reasoning_effort.clone().filter(|e| !e.is_null())
        .or_else(|| kwargs.get("reasoning_effort").filter(|e| !e.is_null()).cloned());
    if effort.as_ref().and_then(Value::as_str) == Some("none") && toggle != Some(true) {
        kwargs.insert("thinking".into(), Value::Bool(false));
        return;
    }
    let max = effort.as_ref().and_then(Value::as_str) == Some("max");
    kwargs.insert("reasoning_effort".into(), Value::String(if max { "high" } else { "low" }.into()));
}

pub(super) fn prepare(state: &AppState, mut body: ChatCompletionRequest) -> Result<Prepared, ApiError> {
    state.check_model(body.model.as_deref())?;
    convert::reject_extra(&body.extra)?;
    if body.sampling.ignore_eos && body.response_format.as_ref().is_some_and(|f| f["type"] != "text") {
        return Err(ApiError::invalid("Structured output does not support ignore_eos.", Some("response_format"), "invalid_request"));
    }
    // Deprecated function-calling aliases.
    if body.tools.is_none()
        && let Some(functions) = body.functions.take()
    {
        body.tools = Some(functions.into_iter().map(|f| json!({"type": "function", "function": f})).collect());
    }
    if body.tool_choice.is_none()
        && let Some(call) = body.function_call.take()
    {
        body.tool_choice = Some(match call {
            Value::Object(_) => json!({"type": "function", "function": {"name": call["name"]}}),
            other => other,
        });
    }
    let disagg_items = if state.disagg_lm { skeleton_media(&mut body.messages) } else { Vec::new() };
    let messages = body.messages.iter().map(convert::message).collect::<Result<Vec<_>, _>>()?;
    let tools = body.tools.iter().flatten().map(convert::tool).collect::<Result<Vec<_>, _>>()?;
    let tool_choice: Option<ChatToolChoice> = body.tool_choice.as_ref().map(convert::tool_choice).transpose()?;
    let tool_context = ResolvedToolContext::new(&messages, tools, tool_choice, body.parallel_tool_calls.unwrap_or(true))
        .map_err(|e| ApiError::invalid(e.to_string(), Some("tools"), "invalid_request"))?;

    let max_tokens = body.max_completion_tokens.or(body.max_tokens);
    super::check_max_tokens(state, "max_tokens", max_tokens)?;
    let mut sampling = convert::sampling(&body.sampling, max_tokens)?;
    if body.logprobs {
        sampling.logprobs = Some(body.top_logprobs.unwrap_or(0).clamp(0, 20));
    }
    sampling.structured_outputs = convert::response_format(body.response_format.as_ref())?;
    if !disagg_items.is_empty() {
        sampling.vllm_xargs = Some(std::collections::HashMap::from([(
            crate::backend::DISAGG_ITEMS_KEY.to_owned(),
            json!(disagg_items),
        )]));
    }
    let request_id = format!("chatcmpl-{}", body.request_id.clone().unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string()));
    let options = body.stream_options.clone().unwrap_or_default();
    let request = ChatRequest {
        request_id,
        messages,
        sampling_params: sampling,
        chat_options: ChatOptions {
            generation_prompt_mode: Default::default(),
            chat_template: None,
            reasoning_effort: None,
            response_format: body.response_format.clone(),
            template_kwargs: template_kwargs(state, &body),
        },
        tool_context,
        decode_options: convert::decode_options(&body.sampling),
        intermediate: body.stream,
        prompt_truncation: None,
        priority: 0,
        documents: None,
        cache_salt: None,
        add_special_tokens: false,
        data_parallel_rank: None,
        session_id: None,
        lora_request: None,
        return_prompt_text: false,
    };
    Ok(Prepared {
        request,
        stream: body.stream,
        include_usage: options.include_usage.unwrap_or(false),
        continuous_usage: options.continuous_usage_stats.unwrap_or(false),
        logprobs: body.logprobs,
    })
}

pub async fn chat_completions(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    let body: ChatCompletionRequest = match serde_json::from_slice(&body) {
        Ok(b) => b,
        Err(e) => return ApiError::invalid(e.to_string(), None, "invalid_request").into_response(),
    };
    let prepared = match prepare(&state, body) {
        Ok(p) => p,
        Err(e) => return e.into_response(),
    };
    let id = prepared.request.request_id.clone();
    let events = match state.chat.chat(prepared.request).await {
        Ok(s) => s,
        Err(e) => return submit_error(e).into_response(),
    };
    let model = state.public_model().to_owned();
    if prepared.stream {
        let chunks = stream_chunks(events, id, model, prepared.include_usage, prepared.continuous_usage, prepared.logprobs);
        Sse::new(chunks).keep_alive(KeepAlive::default()).into_response()
    } else {
        collect(events, id, model, prepared.logprobs).await
    }
}

async fn collect(
    mut events: impl Stream<Item = vllm_chat::Result<ChatEvent>> + Unpin,
    id: String,
    model: String,
    want_logprobs: bool,
) -> Response {
    let mut logprobs = Vec::new();
    let mut prompt_lp = None;
    while let Some(event) = events.next().await {
        let event = match event {
            Ok(e) => e,
            Err(e) => return submit_error(e).into_response(),
        };
        match event {
            ChatEvent::Start { prompt_logprobs: Some(p), .. } => prompt_lp = Some(prompt_logprobs(&p)),
            ChatEvent::LogprobsDelta { logprobs: Some(lp), .. } => logprobs.extend(token_logprobs(&lp)),
            ChatEvent::Done { message, usage, finish_reason, .. } => {
                if let Some(err) = finish_error(&finish_reason) {
                    return err.into_response();
                }
                let mut text = String::new();
                let mut reasoning = String::new();
                let mut calls = Vec::new();
                for block in message.content {
                    match block {
                        AssistantContentBlock::Text { text: t } => text.push_str(&t),
                        AssistantContentBlock::Reasoning { text: t } => reasoning.push_str(&t),
                        AssistantContentBlock::ToolCall(c) => calls.push(json!({
                            "id": c.id, "type": "function",
                            "function": {"name": c.name, "arguments": c.arguments},
                        })),
                    }
                }
                let mut msg = json!({"role": "assistant", "content": if text.is_empty() && !calls.is_empty() { Value::Null } else { Value::String(text) }});
                if !reasoning.is_empty() {
                    msg["reasoning_content"] = json!(reasoning);
                }
                if !calls.is_empty() {
                    msg["tool_calls"] = json!(calls);
                }
                let mut choice = json!({
                    "index": 0,
                    "message": msg,
                    "logprobs": if want_logprobs { json!({"content": logprobs}) } else { Value::Null },
                    "finish_reason": finish_reason_str(&finish_reason, !calls.is_empty()),
                });
                if let Some(p) = &prompt_lp {
                    choice["prompt_logprobs"] = json!(p);
                }
                let usage = Usage::new(usage.prompt_token_count, usage.output_token_count);
                return Json(json!({
                    "id": id, "object": "chat.completion", "created": unix_now(), "model": model,
                    "choices": [choice], "usage": usage,
                }))
                .into_response();
            }
            _ => {}
        }
    }
    ApiError::server("chat stream ended without a result").into_response()
}

fn stream_chunks(
    mut events: impl Stream<Item = vllm_chat::Result<ChatEvent>> + Unpin + Send + 'static,
    id: String,
    model: String,
    include_usage: bool,
    continuous_usage: bool,
    want_logprobs: bool,
) -> impl Stream<Item = Result<Event, Infallible>> + Send + 'static {
    async_stream::stream! {
        let created = unix_now();
        let chunk = |choices: Value, usage: Option<Usage>| {
            let mut c = json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": choices});
            if let Some(u) = usage {
                c["usage"] = json!(u);
            }
            Event::default().data(c.to_string())
        };
        yield Ok(chunk(json!([{"index": 0, "delta": {"role": "assistant", "content": ""}, "logprobs": null, "finish_reason": null}]), None));
        let mut pending_lp: Vec<super::types::TokenLogprobOut> = Vec::new();
        let mut prompt_lp: Option<Value> = None;
        let mut tool_index = std::collections::HashMap::<usize, usize>::new();
        let mut prompt_tokens = 0usize;
        let mut completion_tokens = 0usize;
        while let Some(event) = events.next().await {
            let event = match event {
                Ok(e) => e,
                Err(e) => {
                    yield Ok(Event::default().data(submit_error(e).body().to_string()));
                    break;
                }
            };
            let delta = match event {
                ChatEvent::Start { prompt_token_ids, prompt_logprobs: p } => {
                    prompt_tokens = prompt_token_ids.len();
                    prompt_lp = p.map(|p| json!(prompt_logprobs(&p)));
                    None
                }
                ChatEvent::LogprobsDelta { logprobs, token_ids } => {
                    completion_tokens += token_ids.len();
                    if let Some(lp) = logprobs {
                        pending_lp.extend(token_logprobs(&lp));
                    }
                    None
                }
                ChatEvent::BlockDelta { kind: AssistantBlockKind::Text, delta, .. } => Some(json!({"content": delta})),
                ChatEvent::BlockDelta { kind: AssistantBlockKind::Reasoning, delta, .. } => Some(json!({"reasoning_content": delta})),
                ChatEvent::ToolCallStart { index, id: call_id, name } => {
                    let n = tool_index.len();
                    let i = *tool_index.entry(index).or_insert(n);
                    Some(json!({"tool_calls": [{"index": i, "id": call_id, "type": "function", "function": {"name": name, "arguments": ""}}]}))
                }
                ChatEvent::ToolCallArgumentsDelta { index, delta } => {
                    let i = tool_index.get(&index).copied().unwrap_or(0);
                    Some(json!({"tool_calls": [{"index": i, "function": {"arguments": delta}}]}))
                }
                ChatEvent::Done { usage, finish_reason, .. } => {
                    if let Some(err) = finish_error(&finish_reason) {
                        yield Ok(Event::default().data(err.body().to_string()));
                        break;
                    }
                    if !pending_lp.is_empty() || prompt_lp.is_some() {
                        let mut c = json!({"index": 0, "delta": {}, "logprobs": json!({"content": std::mem::take(&mut pending_lp)}), "finish_reason": null});
                        if let Some(p) = prompt_lp.take() { c["prompt_logprobs"] = p; }
                        yield Ok(chunk(json!([c]), None));
                    }
                    let reason = finish_reason_str(&finish_reason, !tool_index.is_empty());
                    yield Ok(chunk(json!([{"index": 0, "delta": {}, "logprobs": null, "finish_reason": reason}]), None));
                    if include_usage {
                        yield Ok(chunk(json!([]), Some(Usage::new(usage.prompt_token_count, usage.output_token_count))));
                    }
                    break;
                }
                _ => None,
            };
            if let Some(delta) = delta {
                let lp = if want_logprobs && !pending_lp.is_empty() { json!({"content": std::mem::take(&mut pending_lp)}) } else { Value::Null };
                let mut c = json!({"index": 0, "delta": delta, "logprobs": lp, "finish_reason": null});
                if let Some(p) = prompt_lp.take() { c["prompt_logprobs"] = p; }
                let usage = (include_usage && continuous_usage).then(|| Usage::new(prompt_tokens, completion_tokens));
                yield Ok(chunk(json!([c]), usage));
            }
        }
        yield Ok(Event::default().data("[DONE]"));
    }
}
