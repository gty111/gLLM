//! OpenAI wire types for the routes gLLM serves.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum OneOrMany {
    One(String),
    Many(Vec<String>),
}

impl OneOrMany {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            OneOrMany::One(s) => vec![s],
            OneOrMany::Many(v) => v,
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: Option<bool>,
    #[serde(default)]
    pub continuous_usage_stats: Option<bool>,
}

/// Sampling fields shared by chat and completions.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Sampling {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i64>,
    pub repetition_penalty: Option<f32>,
    pub min_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub seed: Option<i64>,
    pub min_tokens: Option<u32>,
    #[serde(default)]
    pub ignore_eos: bool,
    pub stop: Option<OneOrMany>,
    pub stop_token_ids: Option<Vec<u32>>,
    #[serde(default)]
    pub include_stop_str_in_output: bool,
    pub skip_special_tokens: Option<bool>,
    pub prompt_logprobs: Option<i32>,
    pub n: Option<u32>,
    pub logit_bias: Option<HashMap<String, f32>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<Value>,
    pub max_tokens: Option<u32>,
    pub max_completion_tokens: Option<u32>,
    #[serde(default)]
    pub logprobs: bool,
    pub top_logprobs: Option<i32>,
    pub response_format: Option<Value>,
    #[serde(default)]
    pub stream: bool,
    pub stream_options: Option<StreamOptions>,
    pub reasoning_effort: Option<Value>,
    pub chat_template_kwargs: Option<HashMap<String, Value>>,
    pub tools: Option<Vec<Value>>,
    pub tool_choice: Option<Value>,
    pub parallel_tool_calls: Option<bool>,
    pub functions: Option<Vec<Value>>,
    pub function_call: Option<Value>,
    pub request_id: Option<String>,
    #[serde(flatten)]
    pub sampling: Sampling,
    /// Everything else: refused unless null or listed as ignorable.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum CompletionPrompt {
    Text(String),
    Texts(Vec<String>),
    Tokens(Vec<u32>),
    TokenLists(Vec<Vec<u32>>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    pub model: Option<String>,
    pub prompt: CompletionPrompt,
    pub max_tokens: Option<u32>,
    pub logprobs: Option<i32>,
    #[serde(default)]
    pub echo: bool,
    #[serde(default)]
    pub stream: bool,
    pub stream_options: Option<StreamOptions>,
    pub best_of: Option<u32>,
    pub request_id: Option<String>,
    #[serde(flatten)]
    pub sampling: Sampling,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

impl Usage {
    pub fn new(prompt: usize, completion: usize) -> Self {
        Self { prompt_tokens: prompt, completion_tokens: completion, total_tokens: prompt + completion }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TopLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TokenLogprobOut {
    pub token: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
    pub top_logprobs: Vec<TopLogprob>,
}

/// Same shape the Python front end used for prompt logprobs.
#[derive(Debug, Clone, Serialize)]
pub struct PromptLogprobEntry {
    pub token_id: u32,
    pub token: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
    pub top_logprobs: Vec<PromptTopLogprob>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PromptTopLogprob {
    pub token_id: u32,
    pub token: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
}
