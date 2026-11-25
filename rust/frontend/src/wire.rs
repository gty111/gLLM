//! Front end <-> worker messages; mirrors `python/distributed/wire.py`.
//!
//! Structs are msgpack arrays in field order (`rmp_serde::to_vec`), matching
//! msgspec's `array_like` structs on the Python side. Keep both files in sync.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize)]
pub struct Structured {
    pub schema: Option<String>,
    pub tag: Option<String>,
    pub think_start: Option<u32>,
    pub think_end: Option<u32>,
    pub thinking: bool,
    pub allow_think_start: bool,
    pub prefix_tail: Vec<u32>,
}

/// One processor output of one media item.
#[derive(Debug, Clone, Serialize)]
pub struct Tensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    /// One row per item (stacked) rather than concatenated on dim 0.
    pub batched: bool,
    #[serde(with = "serde_bytes")]
    pub data: Vec<u8>,
    /// Offset in the media arena when `data` is empty.
    pub offset: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MediaItem {
    pub modality: String,
    pub reference: Option<String>,
    pub tensors: Option<Vec<Tensor>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Request {
    pub seq_id: u32,
    pub token_ids: Vec<u32>,
    pub finish_tokens: Vec<u32>,
    pub output_len: u32,
    pub requested_output_len: Option<u32>,
    pub model_max_length: u32,
    pub ignore_eos: bool,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub repetition_penalty: f32,
    pub logprobs: Option<u32>,
    pub prompt_logprobs: Option<u32>,
    pub structured: Option<Structured>,
    pub media: Option<Vec<MediaItem>>,
    pub disagg_items: Option<Vec<MediaItem>>,
    /// All media tensors, merged per name in prompt order.
    pub media_tensors: Option<Vec<Tensor>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Control {
    pub cmd: String,
    pub tags: Option<Vec<String>>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Input {
    pub requests: Vec<Request>,
    pub aborts: Vec<u32>,
    pub control: Option<Control>,
    pub log: Option<bool>,
}

impl Input {
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty() && self.aborts.is_empty() && self.control.is_none()
    }

    pub fn encode(&self) -> Vec<u8> {
        rmp_serde::to_vec(self).expect("wire input always encodes")
    }
}

/// One acted row: a single token, or several when MTP commits more than one.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Tokens {
    One(u32),
    Many(Vec<u32>),
}

impl Tokens {
    pub fn into_vec(self) -> Vec<u32> {
        match self {
            Tokens::One(t) => vec![t],
            Tokens::Many(v) => v,
        }
    }
}

/// Sampled logprob, top ids, top logprobs.
pub type LogprobRow = (f32, Vec<u32>, Vec<f32>);
/// Prompt token id, its logprob, top ids, top logprobs.
pub type PromptLogprob = (u32, f32, Vec<u32>, Vec<f32>);

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Output {
    pub seq_ids: Vec<u32>,
    pub tokens: Vec<Tokens>,
    pub logprobs: Vec<Option<LogprobRow>>,
    pub prompt_logprobs: HashMap<u32, Vec<Option<PromptLogprob>>>,
    pub finished: Vec<u32>,
    pub errors: HashMap<u32, String>,
    pub control_ack: Option<serde_json::Value>,
    /// Error code per `errors` entry; absent means a cache-capacity error.
    #[serde(default)]
    pub error_codes: HashMap<u32, String>,
}

impl Output {
    pub fn decode(data: &[u8]) -> Result<Self, rmp_serde::decode::Error> {
        rmp_serde::from_slice(data)
    }
}
