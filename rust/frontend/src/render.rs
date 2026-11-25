//! `--render-request`: run one chat request through the HTTP conversion and
//! the chat stack, print what would reach the engine, and exit.

use std::num::NonZeroU32;
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context as _, Result};
use tokio::sync::watch;
use vllm_chat::{ChatLlm, LoadModelBackendsOptions, ParserSelection};
use vllm_llm::{BoxGenerateOutputStream, GenerateRequest, Llm, LlmInfo};
use vllm_text::TextLlm;

struct Capture {
    info: LlmInfo,
    seen: Arc<Mutex<Option<GenerateRequest>>>,
    health: watch::Sender<bool>,
}

#[async_trait::async_trait]
impl Llm for Capture {
    fn info(&self) -> &LlmInfo {
        &self.info
    }
    fn subscribe_health(&self) -> watch::Receiver<bool> {
        self.health.subscribe()
    }
    fn subscribe_readiness(&self) -> watch::Receiver<bool> {
        self.health.subscribe()
    }
    async fn generate_n(&self, request: GenerateRequest, _: NonZeroU32) -> vllm_llm::Result<Vec<BoxGenerateOutputStream>> {
        *self.seen.lock().unwrap() = Some(request);
        Err(vllm_llm::Error::Unavailable { message: "rendered".into() })
    }
    async fn abort(&self, _: &[String]) -> vllm_llm::Result<()> {
        Ok(())
    }
    async fn shutdown(self: Box<Self>) -> vllm_llm::Result<()> {
        Ok(())
    }
}

pub async fn run(model: &str, request: &Path, (tool, reasoning): (ParserSelection, ParserSelection)) -> Result<()> {
    let body = std::fs::read(request).with_context(|| format!("reading {}", request.display()))?;
    let loaded = vllm_chat::load_model_backends(model, LoadModelBackendsOptions::default()).await?;
    let seen = Arc::new(Mutex::new(None));
    let llm = Capture {
        info: LlmInfo {
            max_model_len: 1 << 20,
            model_dtype: serde_json::from_value(serde_json::json!("bfloat16")).ok(),
            supports_lora: false,
            allow_missing_mm_embeddings: false,
            vllm_version: String::new(),
        },
        seen: seen.clone(),
        health: watch::channel(true).0,
    };
    let chat = ChatLlm::new(TextLlm::new(llm, loaded.text_backend), loaded.chat_backend)
        .with_tool_call_parser(tool)
        .with_reasoning_parser(reasoning);
    let state = crate::http::AppState {
        chat,
        responses: Default::default(),
        model_path: model.to_owned(),
        model_names: vec![model.to_owned()],
        max_model_len: 1 << 20,
        disagg_lm: false,
        legacy_v4_effort: crate::legacy_v4_effort(model),
    };
    let mut body: serde_json::Value = serde_json::from_slice(&body)?;
    body["model"] = serde_json::json!(model);
    let prepared = crate::http::render_prepare(&state, body).map_err(|e| anyhow::anyhow!("{}", e.message))?;
    if let Err(err) = state.chat.chat(prepared).await
        && seen.lock().unwrap().is_none() {
            anyhow::bail!("request was refused before reaching the engine: {err}");
        }
    let r = seen.lock().unwrap().take().unwrap();
    println!("{}", serde_json::json!({
        "prompt_token_ids": r.prompt_token_ids,
        "reasoning_ended": r.reasoning_ended,
        "sampling": r.sampling_params,
    }));
    Ok(())
}
