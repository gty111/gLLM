//! Print what the chat stack sends to the engine for one request.
//! `GLLM_RENDER_MODEL=<dir> GLLM_RENDER_REQUEST=<json> cargo test --release --test render -- --ignored --nocapture`

use std::num::NonZeroU32;
use std::sync::{Arc, Mutex};

use tokio::sync::watch;
use vllm_chat::{ChatLlm, ChatOptions, ChatRequest, LoadModelBackendsOptions, ResolvedToolContext};
use vllm_llm::{BoxGenerateOutputStream, GenerateRequest, Llm, LlmInfo};
use vllm_text::TextLlm;

struct Capture {
    info: LlmInfo,
    seen: Arc<Mutex<Option<GenerateRequest>>>,
    tx: watch::Sender<bool>,
}

#[async_trait::async_trait]
impl Llm for Capture {
    fn info(&self) -> &LlmInfo {
        &self.info
    }
    fn subscribe_health(&self) -> watch::Receiver<bool> {
        self.tx.subscribe()
    }
    fn subscribe_readiness(&self) -> watch::Receiver<bool> {
        self.tx.subscribe()
    }
    async fn generate_n(&self, request: GenerateRequest, _n: NonZeroU32) -> vllm_llm::Result<Vec<BoxGenerateOutputStream>> {
        *self.seen.lock().unwrap() = Some(request);
        Err(vllm_llm::Error::Unavailable { message: "captured".into() })
    }
    async fn abort(&self, _: &[String]) -> vllm_llm::Result<()> {
        Ok(())
    }
    async fn shutdown(self: Box<Self>) -> vllm_llm::Result<()> {
        Ok(())
    }
}

#[tokio::test]
#[ignore]
async fn render() {
    let model = std::env::var("GLLM_RENDER_MODEL").unwrap();
    let body: serde_json::Value = serde_json::from_str(&std::env::var("GLLM_RENDER_REQUEST").unwrap()).unwrap();
    let loaded = vllm_chat::load_model_backends(&model, LoadModelBackendsOptions::default()).await.unwrap();
    let seen = Arc::new(Mutex::new(None));
    let llm = Capture {
        info: LlmInfo { max_model_len: 32768, model_dtype: serde_json::from_value(serde_json::json!("bfloat16")).ok(), supports_lora: false, allow_missing_mm_embeddings: false, vllm_version: String::new() },
        seen: seen.clone(),
        tx: watch::channel(true).0,
    };
    let chat = ChatLlm::new(TextLlm::new(llm, loaded.text_backend), loaded.chat_backend)
        .with_tool_call_parser(vllm_chat::ParserSelection::Explicit("qwen3_coder".into()));
    let mut parts = Vec::new();
    if let Some(url) = body["image"].as_str() {
        parts.push(vllm_chat::ChatContentPart::ImageUrl { image_url: url.to_owned(), detail: None, uuid: None });
    }
    if let Some(t) = body["prefix_text"].as_str() {
        parts.push(vllm_chat::ChatContentPart::text(t));
    }
    parts.push(vllm_chat::ChatContentPart::text(body["content"].as_str().unwrap()));
    let messages = vec![vllm_chat::ChatMessage::user(vllm_chat::ChatContent::Parts(parts))];
    let mut sampling = vllm_chat::SamplingParams { temperature: Some(0.0), max_tokens: Some(64), ..Default::default() };
    if let Some(schema) = body.get("schema") {
        sampling.structured_outputs = Some(vllm_engine_core_client::protocol::structured_outputs::StructuredOutputsParams::json(schema.clone()));
    }
    let request = ChatRequest {
        request_id: "r".into(),
        messages: messages.clone(),
        sampling_params: sampling,
        chat_options: ChatOptions {
            generation_prompt_mode: Default::default(),
            chat_template: None,
            reasoning_effort: None,
            response_format: body.get("response_format").cloned(),
            template_kwargs: serde_json::from_value(body["kwargs"].clone()).unwrap_or_default(),
        },
        tool_context: ResolvedToolContext::new(&messages, vec![], None, true).unwrap(),
        decode_options: Default::default(),
        intermediate: false,
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
    let _ = chat.chat(request).await;
    let r = seen.lock().unwrap().take().expect("request reached the engine");
    println!("PROMPT {:?}", r.prompt_token_ids);
    println!("REASONING_ENDED {:?}", r.reasoning_ended);
    println!("SAMPLING {}", serde_json::to_string(&r.sampling_params).unwrap());
    for f in r.mm_features.iter().flatten() {
        println!("FEATURE {:?} offset={} len={}", f.modality, f.mm_position.offset, f.mm_position.length);
        for (name, elem) in f.data.iter().flatten() {
            if let Some(vllm_engine_core_client::protocol::multimodal::MmKwargValue::Tensor(t)) = &elem.data {
                println!("  {name} {} {:?} field={:?}", t.dtype.as_str(), t.shape.to_vec(), elem.field);
            }
        }
    }
}
