//! `POST /v1/completions`.

use std::convert::Infallible;
use std::sync::Arc;

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures::{Stream, StreamExt as _};
use serde_json::{Map, Value, json};
use vllm_text::{DecodedTextEvent, Prompt, TextRequest};

use super::chat::{finish_error, finish_reason_str, prompt_logprobs, submit_error, token_logprobs};
use super::types::{CompletionPrompt, CompletionRequest, TokenLogprobOut, Usage};
use super::{ApiError, AppState, convert, unix_now};

/// OpenAI completion logprobs; `top_logprobs` always includes the sampled token.
fn completion_logprobs(entries: &[TokenLogprobOut], mut offset: usize) -> Value {
    let mut tokens = Vec::new();
    let mut token_logprobs = Vec::new();
    let mut top = Vec::new();
    let mut offsets = Vec::new();
    for e in entries {
        tokens.push(json!(e.token));
        token_logprobs.push(json!(e.logprob));
        let mut map = Map::new();
        for t in &e.top_logprobs {
            map.insert(t.token.clone(), json!(t.logprob));
        }
        map.entry(e.token.clone()).or_insert(json!(e.logprob));
        top.push(Value::Object(map));
        offsets.push(json!(offset));
        offset += e.token.chars().count();
    }
    json!({"tokens": tokens, "token_logprobs": token_logprobs, "top_logprobs": top, "text_offset": offsets})
}

fn prepare(state: &AppState, body: CompletionRequest) -> Result<(TextRequest, bool, bool), ApiError> {
    state.check_model(body.model.as_deref())?;
    convert::reject_extra(&body.extra)?;
    if body.echo {
        return Err(ApiError::unsupported("echo"));
    }
    if body.best_of.is_some_and(|b| b != 1) {
        return Err(ApiError::unsupported("best_of"));
    }
    let prompt = match body.prompt {
        CompletionPrompt::Text(t) => Prompt::Text(t),
        CompletionPrompt::Tokens(t) => Prompt::TokenIds(t),
        CompletionPrompt::Texts(mut v) if v.len() == 1 => Prompt::Text(v.pop().unwrap()),
        CompletionPrompt::TokenLists(mut v) if v.len() == 1 => Prompt::TokenIds(v.pop().unwrap()),
        _ => return Err(ApiError::invalid("Only one prompt per request is supported.", Some("prompt"), "invalid_request")),
    };
    super::check_max_tokens(state, "max_tokens", body.max_tokens)?;
    let mut sampling = convert::sampling(&body.sampling, body.max_tokens.or(Some(16)))?;
    sampling.logprobs = body.logprobs.map(|n| n.clamp(0, 20));
    let request_id = format!("cmpl-{}", body.request_id.unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string()));
    let include_usage = body.stream_options.and_then(|o| o.include_usage).unwrap_or(false);
    let mut request = TextRequest::for_test();
    request.request_id = request_id;
    request.prompt = prompt;
    request.sampling_params = sampling;
    request.decode_options = convert::decode_options(&body.sampling);
    request.intermediate = body.stream;
    request.add_special_tokens = true;
    Ok((request, body.stream, include_usage))
}

pub async fn completions(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    let body: CompletionRequest = match serde_json::from_slice(&body) {
        Ok(b) => b,
        Err(e) => return ApiError::invalid(e.to_string(), None, "invalid_request").into_response(),
    };
    let (request, stream, include_usage) = match prepare(&state, body) {
        Ok(p) => p,
        Err(e) => return e.into_response(),
    };
    let id = request.request_id.clone();
    let events = match state.chat.text().generate(request).await {
        Ok(s) => Box::pin(s),
        Err(e) => return submit_error(e).into_response(),
    };
    let model = state.public_model().to_owned();
    if stream {
        Sse::new(stream_chunks(events, id, model, include_usage)).keep_alive(KeepAlive::default()).into_response()
    } else {
        collect(events, id, model).await
    }
}

async fn collect(
    mut events: impl Stream<Item = vllm_text::Result<DecodedTextEvent>> + Unpin,
    id: String,
    model: String,
) -> Response {
    let mut text = String::new();
    let mut logprobs: Option<Vec<TokenLogprobOut>> = None;
    let mut prompt_lp = None;
    while let Some(event) = events.next().await {
        match event {
            Err(e) => return submit_error(e).into_response(),
            Ok(DecodedTextEvent::Start { prompt_logprobs: p, .. }) => prompt_lp = p.map(|p| prompt_logprobs(&p)),
            Ok(DecodedTextEvent::TextDelta { decoded, sampled, finished }) => {
                text.push_str(&decoded.text);
                if let Some(lp) = &sampled.logprobs {
                    logprobs.get_or_insert_with(Vec::new).extend(token_logprobs(lp));
                }
                if let Some(f) = finished {
                    if let Some(err) = finish_error(&f.finish_reason) {
                        return err.into_response();
                    }
                    let mut choice = json!({
                        "index": 0, "text": text,
                        "logprobs": logprobs.as_deref().map(|e| completion_logprobs(e, 0)),
                        "finish_reason": finish_reason_str(&f.finish_reason, false),
                    });
                    if let Some(p) = &prompt_lp {
                        choice["prompt_logprobs"] = json!(p);
                    }
                    return Json(json!({
                        "id": id, "object": "text_completion", "created": unix_now(), "model": model,
                        "choices": [choice],
                        "usage": Usage::new(f.usage.prompt_token_count, f.usage.output_token_count),
                    }))
                    .into_response();
                }
            }
        }
    }
    ApiError::server("completion stream ended without a result").into_response()
}

fn stream_chunks(
    mut events: impl Stream<Item = vllm_text::Result<DecodedTextEvent>> + Unpin + Send + 'static,
    id: String,
    model: String,
    include_usage: bool,
) -> impl Stream<Item = Result<Event, Infallible>> + Send + 'static {
    async_stream::stream! {
        let created = unix_now();
        let chunk = |choices: Value, usage: Option<Usage>| {
            let mut c = json!({"id": id, "object": "text_completion", "created": created, "model": model, "choices": choices});
            if let Some(u) = usage {
                c["usage"] = json!(u);
            }
            Event::default().data(c.to_string())
        };
        let mut offset = 0usize;
        let mut prompt_lp: Option<Value> = None;
        while let Some(event) = events.next().await {
            match event {
                Err(e) => {
                    yield Ok(Event::default().data(submit_error(e).body().to_string()));
                    break;
                }
                Ok(DecodedTextEvent::Start { prompt_logprobs: p, .. }) => prompt_lp = p.map(|p| json!(prompt_logprobs(&p))),
                Ok(DecodedTextEvent::TextDelta { decoded, sampled, finished }) => {
                    let entries = sampled.logprobs.as_ref().map(token_logprobs);
                    if !decoded.text.is_empty() || entries.is_some() || prompt_lp.is_some() {
                        let mut c = json!({"index": 0, "text": decoded.text,
                                           "logprobs": entries.as_deref().map(|e| completion_logprobs(e, offset))});
                        if let Some(p) = prompt_lp.take() { c["prompt_logprobs"] = p; }
                        offset += decoded.text.chars().count();
                        yield Ok(chunk(json!([c]), None));
                    }
                    if let Some(f) = finished {
                        if let Some(err) = finish_error(&f.finish_reason) {
                            yield Ok(Event::default().data(err.body().to_string()));
                            break;
                        }
                        yield Ok(chunk(json!([{"index": 0, "text": "", "finish_reason": finish_reason_str(&f.finish_reason, false)}]), None));
                        if include_usage {
                            yield Ok(chunk(json!([]), Some(Usage::new(f.usage.prompt_token_count, f.usage.output_token_count))));
                        }
                        break;
                    }
                }
            }
        }
        yield Ok(Event::default().data("[DONE]"));
    }
}
