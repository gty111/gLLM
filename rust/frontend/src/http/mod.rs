//! OpenAI-compatible HTTP API plus gLLM's control routes.

mod chat;
mod completions;
mod convert;
mod responses;
pub mod types;

use std::any::Any;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{Query, RawQuery, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{Value, json};
use vllm_chat::ChatLlm;
use vllm_llm::Llm as _;

use crate::backend::GllmLlm;
use crate::sleep::SleepError;

pub struct AppState {
    pub chat: ChatLlm,
    pub responses: responses::ResponseStore,
    pub model_path: String,
    /// Names a request's `model` may use; the first is reported.
    pub model_names: Vec<String>,
    pub max_model_len: u32,
    /// LM node of encoder disaggregation (see `backend::DISAGG_ITEMS_KEY`).
    pub disagg_lm: bool,
    /// DeepSeek V4 checkpoint whose own encoder predates the 0731 effort
    /// prompts (see `chat::legacy_v4_effort`).
    pub legacy_v4_effort: bool,
}

impl AppState {
    pub fn engine(&self) -> &GllmLlm {
        let llm: &dyn Any = self.chat.text().llm();
        llm.downcast_ref::<GllmLlm>().expect("the front end always runs a GllmLlm")
    }

    pub fn public_model(&self) -> &str {
        &self.model_names[0]
    }

    pub fn check_model(&self, model: Option<&str>) -> Result<(), ApiError> {
        match model {
            None => Err(ApiError::invalid("Field required", Some("model"), "invalid_request")),
            Some(m) if self.model_names.iter().any(|n| n == m) => Ok(()),
            Some(m) => Err(ApiError {
                status: StatusCode::NOT_FOUND,
                message: format!("The model `{m}` does not exist or is not loaded by this server."),
                kind: "invalid_request_error",
                param: Some("model".into()),
                code: Some("model_not_found".into()),
            }),
        }
    }
}

#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
    pub kind: &'static str,
    pub param: Option<String>,
    pub code: Option<String>,
}

impl ApiError {
    pub fn invalid(message: impl Into<String>, param: Option<&str>, code: &str) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            kind: "invalid_request_error",
            param: param.map(str::to_owned),
            code: Some(code.to_owned()),
        }
    }

    pub fn unsupported(param: &str) -> Self {
        Self::invalid(
            format!("The parameter `{param}` is not supported by this server."),
            Some(param),
            "unsupported_parameter",
        )
    }

    pub fn server(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
            kind: "server_error",
            param: None,
            code: Some("internal_error".into()),
        }
    }

    pub fn unavailable(message: impl Into<String>, code: &str) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: message.into(),
            kind: "server_error",
            param: None,
            code: Some(code.to_owned()),
        }
    }

    pub fn body(&self) -> Value {
        json!({"error": {"message": self.message, "type": self.kind, "param": self.param, "code": self.code}})
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(self.body())).into_response()
    }
}

/// A budget the context can never hold. Smaller budgets are clamped to what
/// the prompt leaves free.
pub fn check_max_tokens(state: &AppState, param: &str, max_tokens: Option<u32>) -> Result<(), ApiError> {
    match max_tokens {
        Some(n) if n > state.max_model_len => Err(ApiError::invalid(
            format!(
                "This model's maximum context length is {} tokens; `{param}` ({n}) exceeds it.",
                state.max_model_len
            ),
            Some(param),
            "context_length_exceeded",
        )),
        _ => Ok(()),
    }
}

/// The chat request a body becomes (for `--render-request`).
pub fn render_prepare(state: &AppState, body: Value) -> Result<vllm_chat::ChatRequest, ApiError> {
    let body: types::ChatCompletionRequest =
        serde_json::from_value(body).map_err(|e| ApiError::invalid(e.to_string(), None, "invalid_request"))?;
    Ok(chat::prepare(state, body)?.request)
}

pub fn unix_now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// `max_body` bounds one request body: media arrive inline as data URLs, so
/// the framework's 2 MiB default would refuse ordinary images and clips.
pub fn router(state: Arc<AppState>, max_body: usize) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/version", get(version))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat::chat_completions))
        .route("/v1/completions", post(completions::completions))
        .route("/v1/responses", post(responses::create_response))
        .route("/server_info", get(server_info))
        .route("/start_profile", post(start_profile))
        .route("/stop_profile", post(stop_profile))
        .route("/sleep", post(sleep))
        .route("/wake_up", post(wake_up))
        .route("/is_sleeping", get(is_sleeping))
        .layer(axum::extract::DefaultBodyLimit::max(max_body))
        .with_state(state)
}

/// The gLLM release this front end serves (the Python package version).
const GLLM_VERSION: &str = env!("CARGO_PKG_VERSION");

async fn health(State(state): State<Arc<AppState>>) -> Response {
    if state.engine().is_healthy() {
        Json(json!({"status": "ok"})).into_response()
    } else {
        ApiError::unavailable("engine is not running", "engine_dead").into_response()
    }
}

async fn version() -> Json<Value> {
    Json(json!({"version": GLLM_VERSION}))
}

async fn server_info(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({"model": state.model_path, "version": GLLM_VERSION, "status": "running"}))
}

async fn models(State(state): State<Arc<AppState>>) -> Json<Value> {
    let created = unix_now();
    let data: Vec<Value> = state
        .model_names
        .iter()
        .take(1)
        .map(|id| {
            json!({"id": id, "object": "model", "created": created, "owned_by": "gllm",
                   "max_model_len": state.max_model_len})
        })
        .collect();
    Json(json!({"object": "list", "data": data}))
}

async fn start_profile(State(state): State<Arc<AppState>>) -> Response {
    match state.engine().control("start_profile").await {
        Ok(()) => StatusCode::OK.into_response(),
        Err(err) => ApiError::server(err.to_string()).into_response(),
    }
}

async fn stop_profile(State(state): State<Arc<AppState>>) -> Response {
    match state.engine().control("stop_profile").await {
        Ok(()) => StatusCode::OK.into_response(),
        Err(err) => ApiError::server(err.to_string()).into_response(),
    }
}

/// Repeated `tags=` query parameters.
fn query_tags(raw: Option<&str>) -> Option<Vec<String>> {
    let tags: Vec<String> = raw
        .unwrap_or("")
        .split('&')
        .filter_map(|kv| kv.split_once('='))
        .filter(|(k, _)| *k == "tags")
        .map(|(_, v)| v.to_owned())
        .collect();
    (!tags.is_empty()).then_some(tags)
}

fn sleep_error(err: SleepError) -> Response {
    let (status, kind, code) = match &err {
        SleepError::BadRequest(_) => (StatusCode::BAD_REQUEST, "invalid_request_error", "sleep_mode_error"),
        SleepError::Disabled => (StatusCode::BAD_REQUEST, "invalid_request_error", "sleep_mode_disabled"),
        SleepError::Failed(_) => (StatusCode::INTERNAL_SERVER_ERROR, "server_error", "sleep_mode_error"),
    };
    ApiError { status, message: err.to_string(), kind, param: None, code: Some(code.into()) }.into_response()
}

#[derive(serde::Deserialize)]
struct SleepQuery {
    #[serde(default = "one")]
    level: u32,
    #[serde(default = "wait")]
    mode: String,
}

fn one() -> u32 {
    1
}

fn wait() -> String {
    "wait".into()
}

async fn sleep(
    State(state): State<Arc<AppState>>,
    Query(q): Query<SleepQuery>,
    RawQuery(raw): RawQuery,
) -> Response {
    let abort = match q.mode.as_str() {
        "wait" => false,
        "abort" => true,
        other => return sleep_error(SleepError::BadRequest(format!("invalid mode {other:?}; expected 'wait' or 'abort'"))),
    };
    let engine = state.engine();
    // Runs to completion even if the caller goes away.
    let ctl = engine.sleep.clone();
    let memory = engine.memory();
    let tags = query_tags(raw.as_deref());
    match tokio::spawn(async move { ctl.sleep(&memory, tags, abort, q.level).await }).await {
        Ok(Ok(mut result)) => {
            result["status"] = json!("sleeping");
            Json(result).into_response()
        }
        Ok(Err(err)) => sleep_error(err),
        Err(err) => ApiError::server(err.to_string()).into_response(),
    }
}

async fn wake_up(State(state): State<Arc<AppState>>, RawQuery(raw): RawQuery) -> Response {
    let engine = state.engine();
    let ctl = engine.sleep.clone();
    let memory = engine.memory();
    let tags = query_tags(raw.as_deref());
    match tokio::spawn(async move { ctl.wake_up(memory.as_ref(), tags).await }).await {
        Ok(Ok(mut result)) => {
            let asleep = result["sleeping_tags"].as_array().is_some_and(|t| !t.is_empty());
            result["status"] = json!(if asleep { "sleeping" } else { "awake" });
            Json(result).into_response()
        }
        Ok(Err(err)) => sleep_error(err),
        Err(err) => ApiError::server(err.to_string()).into_response(),
    }
}

async fn is_sleeping(State(state): State<Arc<AppState>>) -> Json<Value> {
    let sleep = &state.engine().sleep;
    let tags = sleep.sleeping_tags();
    Json(json!({"is_sleeping": !tags.is_empty(), "sleeping_tags": tags, "level": sleep.level()}))
}
