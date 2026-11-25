//! `GllmLlm`: the `vllm_llm::Llm` implementation that drives gLLM workers.
//!
//! One actor task owns the ZMQ sockets and every in-flight request. Callers
//! talk to it over a channel; it batches pending work into one `Input` per DP
//! replica, and fans each worker `Output` back out to per-request streams.

use std::collections::{HashMap, VecDeque};
use std::num::NonZeroU32;
use std::sync::Arc;

use anyhow::{Context as _, Result, bail};
use futures::StreamExt as _;
use tokio::sync::{mpsc, oneshot, watch};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{error, warn};
use vllm_engine_core_client::protocol::output::{RequestError, StopReason};
use vllm_engine_core_client::protocol::multimodal::{MmFeatures, MmField, MmKwargValue, MmModality};
use vllm_engine_core_client::protocol::structured_outputs::StructuredOutputConstraint;
use vllm_llm::{
    AuxOutputRows, BoxGenerateOutputStream, FinishReason, GenerateOutput, GeneratePromptInfo,
    GenerateRequest, Llm, LlmInfo, Logprobs, PositionLogprobs, RequestTimestamps, TokenLogprob,
    WeightVersions,
};
use vllm_tokenizer::DynTokenizer;
use zeromq::{Socket as _, SocketRecv as _, SocketSend as _};

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use crate::arena::{MediaArena, Range};
use crate::launcher::Launcher;
use crate::sleep::{MemoryCommand, SleepCtl};

/// How long a request waits for room in a full media arena.
const MEDIA_WAIT: Duration = Duration::from_secs(30);
use crate::wire::{self, Control, Input, Output};

/// Upper bound for one sleep/wake round trip.
const MEMORY_CMD_TIMEOUT: Duration = Duration::from_secs(600);

async fn control(
    cmd_tx: &mpsc::UnboundedSender<Cmd>,
    cmd: &str,
    tags: Option<Vec<String>>,
) -> Result<Vec<serde_json::Value>> {
    let (reply, rx) = oneshot::channel();
    cmd_tx
        .send(Cmd::Control { control: Control { cmd: cmd.to_owned(), tags }, reply })
        .map_err(|_| anyhow::anyhow!("engine is not running"))?;
    rx.await.context("engine stopped before acknowledging")
}

/// Cheap handle used by the sleep controller.
struct Handle {
    cmd: mpsc::UnboundedSender<Cmd>,
    /// Starts and stops the engine for sleep level 2.
    launcher: Option<Arc<Launcher>>,
    dp_size: usize,
}

/// One request socket per DP replica.
async fn connect_replicas(ipc_prefix: &str, dp_size: usize) -> Result<Vec<zeromq::PushSocket>> {
    let mut pushes = Vec::with_capacity(dp_size);
    for rank in 0..dp_size {
        let suffix = if dp_size > 1 { format!("_dp{rank}") } else { String::new() };
        let mut push = zeromq::PushSocket::new();
        push.connect(&format!("ipc:///tmp/{ipc_prefix}_gllm_schedule{suffix}"))
            .await
            .with_context(|| format!("connecting to DP replica {rank}"))?;
        pushes.push(push);
    }
    Ok(pushes)
}

#[async_trait::async_trait]
impl MemoryCommand for Handle {
    async fn memory_command(&self, name: &str, tags: Vec<String>) -> Result<Vec<serde_json::Value>> {
        tokio::time::timeout(MEMORY_CMD_TIMEOUT, control(&self.cmd, name, Some(tags)))
            .await
            .map_err(|_| anyhow::anyhow!("timed out waiting for workers to {name}"))?
    }

    async fn abort_all(&self) {
        let _ = self.cmd.send(Cmd::Abort { external_ids: Vec::new() });
    }

    async fn stop_workers(&self) -> Result<()> {
        let launcher = self.launcher.as_ref().context("the engine was not started by this front end")?;
        let (reply, done) = oneshot::channel();
        self.cmd.send(Cmd::Replicas { pushes: Vec::new(), reply }).map_err(|_| anyhow::anyhow!("engine is not running"))?;
        let _ = done.await;
        launcher.stop().await;
        Ok(())
    }

    async fn start_workers(&self) -> Result<()> {
        let launcher = self.launcher.as_ref().context("the engine was not started by this front end")?;
        let started = async {
            let ready = launcher.start(Some(launcher.wake_timeout)).await?;
            if ready.dp_size != self.dp_size {
                bail!("the restarted engine has {} DP replicas instead of {}", ready.dp_size, self.dp_size);
            }
            connect_replicas(&ready.ipc_prefix, ready.dp_size).await
        };
        match started.await {
            Ok(pushes) => {
                let (reply, done) = oneshot::channel();
                self.cmd.send(Cmd::Replicas { pushes, reply }).map_err(|_| anyhow::anyhow!("engine is not running"))?;
                let _ = done.await;
                Ok(())
            }
            Err(err) => {
                // Do not leave a half-started engine holding GPU memory.
                launcher.stop().await;
                Err(err)
            }
        }
    }
}

/// Facts the Python engine reports once its workers are up.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct EngineReady {
    pub ipc_prefix: String,
    pub dp_size: usize,
    pub max_model_len: u32,
    pub use_mm: bool,
    #[serde(default)]
    pub dtype: String,
    pub sleep_mode: bool,
    #[serde(default)]
    pub sleep_idle_timeout: f64,
    #[serde(default = "one")]
    pub sleep_idle_level: u32,
    /// Sleep level 2 is supported: every worker is local to this front end.
    #[serde(default)]
    pub full_sleep: bool,
    /// LM node of encoder disaggregation: prompts are skeletons with one
    /// placeholder per media item, and items travel as raw references.
    #[serde(default)]
    pub disagg_lm: bool,
}

fn one() -> u32 {
    1
}

type OutputTx = mpsc::UnboundedSender<vllm_llm::Result<GenerateOutput>>;

enum Cmd {
    Submit {
        ranges: Vec<Range>,
        request: Box<wire::Request>,
        external_id: String,
        prompt: Arc<[u32]>,
        stop_eos: Option<u32>,
        dp_rank: Option<u32>,
        out: OutputTx,
    },
    /// The caller dropped its stream before the end.
    Drop { seq_id: u32 },
    Abort { external_ids: Vec<String> },
    Control { control: Control, reply: oneshot::Sender<Vec<serde_json::Value>> },
    /// Swap the request sockets: none while the workers are stopped (sleep
    /// level 2), new ones once they are back.
    Replicas { pushes: Vec<zeromq::PushSocket>, reply: oneshot::Sender<()> },
}

pub struct GllmLlm {
    info: LlmInfo,
    cmd: mpsc::UnboundedSender<Cmd>,
    handle: Arc<dyn MemoryCommand>,
    pub sleep: Arc<SleepCtl>,
    arena: Option<Arc<MediaArena>>,
    ids: Arc<std::sync::Mutex<IdPool>>,
    health: watch::Receiver<bool>,
    readiness: watch::Receiver<bool>,
    tokenizer: Option<DynTokenizer>,
}

/// Worker-side sequence ids, handed out like the engine's `IDAllocator`: fresh
/// ids first, then freed ones in the order they were freed. Worker-side state
/// (e.g. the encoder-disaggregation coordinator) may outlive a request by a
/// few steps, so an id must not come back right after it is freed.
struct IdPool {
    free: VecDeque<u32>,
    next: u32,
}

impl IdPool {
    const LIMIT: u32 = 100_000;

    fn allocate(&mut self) -> Option<u32> {
        if self.next < Self::LIMIT {
            self.next += 1;
            return Some(self.next - 1);
        }
        self.free.pop_front()
    }
}

impl GllmLlm {
    /// Bind the output socket. Call before starting the Python engine.
    pub async fn bind_output(ipc_prefix: &str) -> Result<zeromq::PullSocket> {
        let mut pull = zeromq::PullSocket::new();
        pull.bind(&format!("ipc:///tmp/{ipc_prefix}_gllm_output"))
            .await
            .context("binding the worker output socket")?;
        Ok(pull)
    }

    /// Connect to the workers and start the IO actor.
    pub async fn connect(
        ready: EngineReady,
        output: zeromq::PullSocket,
        tokenizer: Option<DynTokenizer>,
        arena: Option<Arc<MediaArena>>,
        launcher: Option<Arc<Launcher>>,
    ) -> Result<Self> {
        let pushes = connect_replicas(&ready.ipc_prefix, ready.dp_size).await?;
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
        let (health_tx, health) = watch::channel(true);
        let (_readiness_tx, readiness) = watch::channel(true);
        let ids = Arc::new(std::sync::Mutex::new(IdPool { free: VecDeque::new(), next: 0 }));
        let in_flight = Arc::new(AtomicUsize::new(0));
        let full_sleep = ready.full_sleep && launcher.is_some();
        let sleep = SleepCtl::new(ready.sleep_mode, full_sleep, ready.sleep_idle_level, in_flight.clone());
        let handle: Arc<dyn MemoryCommand> =
            Arc::new(Handle { cmd: cmd_tx.clone(), launcher, dp_size: ready.dp_size });
        sleep.spawn_idle_watcher(handle.clone(), Duration::from_secs_f64(ready.sleep_idle_timeout.max(0.0)));
        let actor = Actor {
            arena: arena.clone(),
            in_flight,
            dp_size: ready.dp_size,
            pushes,
            pull: output,
            ids: ids.clone(),
            requests: HashMap::new(),
            by_external: HashMap::new(),
            next_dp: 0,
            pending_acks: VecDeque::new(),
        };
        tokio::spawn(async move {
            if let Err(err) = actor.run(cmd_rx).await {
                error!("engine IO stopped: {err:#}");
            }
            let _ = health_tx.send(false);
        });
        Ok(Self {
            info: LlmInfo {
                max_model_len: ready.max_model_len,
                model_dtype: if ready.use_mm { serde_json::from_value(serde_json::Value::String(ready.dtype.clone())).ok() } else { None },
                supports_lora: false,
                allow_missing_mm_embeddings: false,
                vllm_version: format!("gllm-frontend {}", env!("CARGO_PKG_VERSION")),
            },
            cmd: cmd_tx,
            handle,
            sleep,
            arena,
            ids,
            health,
            readiness,
            tokenizer,
        })
    }

    /// Profiling commands go to every replica and are not acknowledged.
    pub async fn control(&self, cmd: &str) -> Result<()> {
        if self.sleep.level() == 2 {
            bail!("the workers are stopped (sleep level 2); wake the server first");
        }
        control(&self.cmd, cmd, None).await.map(|_| ())
    }

    pub fn memory(&self) -> Arc<dyn MemoryCommand> {
        self.handle.clone()
    }

    fn lower(&self, request: &GenerateRequest, seq_id: u32) -> vllm_llm::Result<(wire::Request, Option<u32>)> {
        let sp = &request.sampling_params;
        let unsupported = |field: &'static str| Err(vllm_llm::Error::Unsupported { field });
        if sp.min_p != 0.0 {
            return unsupported("min_p");
        }
        if sp.frequency_penalty != 0.0 || sp.presence_penalty != 0.0 {
            return unsupported("frequency_penalty/presence_penalty");
        }
        if sp.seed.is_some() {
            return unsupported("seed");
        }
        if sp.min_tokens != 0 {
            return unsupported("min_tokens");
        }
        if sp.logit_bias.as_ref().is_some_and(|b| !b.is_empty())
            || sp.allowed_token_ids.is_some()
            || sp.bad_words_token_ids.is_some()
            || sp.logprob_token_ids.is_some()
        {
            return unsupported("logit_bias/allowed_token_ids/bad_words");
        }
        if sp.prompt_logprob_token_ids.is_some() || sp.prompt_logprob_start.is_some() {
            return unsupported("prompt_logprob_token_ids");
        }
        if sp.thinking_token_budget.is_some() {
            return unsupported("thinking_token_budget");
        }
        if sp.repetition_detection.is_some() {
            return unsupported("repetition_detection");
        }
        if request.lora_request.is_some() {
            return unsupported("lora");
        }
        let logprobs = match sp.logprobs {
            None => None,
            Some(n) if n >= 0 => Some(n as u32),
            Some(_) => return unsupported("logprobs=-1"),
        };
        let prompt_logprobs = match sp.prompt_logprobs {
            None => None,
            Some(n) if n >= 0 => Some(n as u32),
            Some(_) => return unsupported("prompt_logprobs=-1"),
        };

        // EOS is a stop token unless the request ignores it; explicit stop ids
        // always stop.
        let mut finish_tokens = sp.stop_token_ids.clone();
        if let Some(eos) = sp.eos_token_id
            && !finish_tokens.contains(&eos)
        {
            finish_tokens.push(eos);
        }
        let structured = match &sp.structured_outputs {
            None => None,
            Some(params) => Some(self.structured(&params.params.constraint, &request.prompt_token_ids, request.reasoning_ended)?),
        };
        let prompt_len = request.prompt_token_ids.len() as u32;
        Ok((
            wire::Request {
                seq_id,
                token_ids: request.prompt_token_ids.clone(),
                finish_tokens,
                output_len: sp.max_tokens,
                requested_output_len: Some(sp.max_tokens),
                model_max_length: self.info.max_model_len.max(prompt_len + sp.max_tokens),
                ignore_eos: false,
                temperature: sp.temperature,
                top_p: sp.top_p,
                top_k: sp.top_k as i32,
                repetition_penalty: sp.repetition_penalty,
                logprobs,
                prompt_logprobs,
                structured,
                media: None,
                disagg_items: disagg_items(sp.extra_args.as_ref())?,
                media_tensors: None,
            },
            sp.eos_token_id,
        ))
    }

    /// Put the request's media tensors into the arena (or inline without one).
    async fn stage_media(&self, request: &GenerateRequest, out: &mut wire::Request) -> vllm_llm::Result<Vec<Range>> {
        let Some(features) = &request.mm_features else { return Ok(Vec::new()) };
        let (items, merged) = merge_media(features)?;
        let mut ranges = Vec::new();
        let mut tensors = Vec::with_capacity(merged.len());
        for (mut tensor, chunks) in merged {
            let bytes: Vec<u8> = if chunks.len() == 1 { chunks[0].to_vec() } else { chunks.concat() };
            match &self.arena {
                Some(arena) => match arena.write(&bytes, MEDIA_WAIT).await {
                    Some(range) => {
                        tensor.offset = Some(range.offset);
                        ranges.push(range);
                    }
                    None => {
                        for r in ranges {
                            arena.release(r);
                        }
                        return Err(vllm_llm::Error::Unavailable { message: "media arena is full; retry later".into() });
                    }
                },
                None => tensor.data = bytes,
            }
            tensors.push(tensor);
        }
        out.media = Some(items);
        out.media_tensors = Some(tensors);
        Ok(ranges)
    }

    /// Map a constraint onto the worker's structured-output spec, including the
    /// reasoning gate: the grammar applies after `</think>` unless the caller
    /// built it to cover reasoning from the first token.
    fn structured(
        &self,
        constraint: &StructuredOutputConstraint,
        prompt: &[u32],
        reasoning_ended: Option<bool>,
    ) -> vllm_llm::Result<wire::Structured> {
        let (schema, tag) = match constraint {
            StructuredOutputConstraint::Json(value) => (Some(value.to_string()), None),
            StructuredOutputConstraint::JsonObject => (Some(r#"{"type": "object"}"#.to_owned()), None),
            StructuredOutputConstraint::StructuralTag(tag) => (None, Some(tag.clone())),
            StructuredOutputConstraint::Regex(_) => {
                return Err(vllm_llm::Error::Unsupported { field: "structured_outputs.regex" });
            }
            StructuredOutputConstraint::Choice(_) => {
                return Err(vllm_llm::Error::Unsupported { field: "structured_outputs.choice" });
            }
            StructuredOutputConstraint::Grammar(_) => {
                return Err(vllm_llm::Error::Unsupported { field: "structured_outputs.grammar" });
            }
        };
        let mut spec = wire::Structured {
            schema,
            tag,
            think_start: None,
            think_end: None,
            thinking: false,
            allow_think_start: false,
            prefix_tail: prompt.last().copied().into_iter().collect(),
        };
        if reasoning_ended == Some(true) {
            return Ok(spec);
        }
        let Some(tok) = &self.tokenizer else { return Ok(spec) };
        let (Some(start), Some(end)) = (tok.token_to_id("<think>"), tok.token_to_id("</think>")) else {
            return Ok(spec);
        };
        let tail = &prompt[prompt.len().saturating_sub(16)..];
        let suffix = tok.decode(tail, false).unwrap_or_default();
        let suffix = suffix.trim_end();
        spec.think_start = Some(start);
        spec.think_end = Some(end);
        if suffix.ends_with("</think>") {
            return Ok(spec); // the template closed reasoning already
        }
        let prefilled = suffix.ends_with("<think>");
        spec.thinking = prefilled;
        spec.allow_think_start = !prefilled;
        Ok(spec)
    }
}

#[async_trait::async_trait]
impl Llm for GllmLlm {
    fn info(&self) -> &LlmInfo {
        &self.info
    }

    fn subscribe_health(&self) -> watch::Receiver<bool> {
        self.health.clone()
    }

    fn subscribe_readiness(&self) -> watch::Receiver<bool> {
        self.readiness.clone()
    }

    async fn generate_n(
        &self,
        request: GenerateRequest,
        n: NonZeroU32,
    ) -> vllm_llm::Result<Vec<BoxGenerateOutputStream>> {
        if n.get() != 1 {
            return Err(vllm_llm::Error::Unsupported { field: "n" });
        }
        if request.prompt_token_ids.is_empty() {
            return Err(vllm_llm::Error::EmptyPromptTokenIds { request_id: request.request_id });
        }
        self.sleep
            .admit(self.handle.clone())
            .await
            .map_err(|message| vllm_llm::Error::Unavailable { message })?;
        let Some(seq_id) = self.ids.lock().unwrap().allocate() else {
            self.sleep.unadmit();
            return Err(vllm_llm::Error::Unavailable { message: "too many requests in flight".into() });
        };
        let give_back = |err| {
            self.ids.lock().unwrap().free.push_back(seq_id);
            self.sleep.unadmit();
            Err(err)
        };
        let (mut wire_request, stop_eos) = match self.lower(&request, seq_id) {
            Ok(v) => v,
            Err(err) => return give_back(err),
        };
        let ranges = match self.stage_media(&request, &mut wire_request).await {
            Ok(r) => r,
            Err(err) => return give_back(err),
        };
        let (tx, rx) = mpsc::unbounded_channel();
        // From here the actor owns the admission (it retires the request).
        self.cmd
            .send(Cmd::Submit {
                ranges,
                request: Box::new(wire_request),
                external_id: request.request_id.clone(),
                prompt: Arc::from(request.prompt_token_ids),
                stop_eos,
                dp_rank: request.data_parallel_rank,
                out: tx,
            })
            .map_err(|_| {
                self.sleep.unadmit();
                vllm_llm::Error::Unavailable { message: "engine is not running".into() }
            })?;
        Ok(vec![Box::pin(GuardedStream {
            inner: Box::pin(UnboundedReceiverStream::new(rx)),
            seq_id,
            cmd: self.cmd.clone(),
            finished: false,
        })])
    }

    async fn abort(&self, external_ids: &[String]) -> vllm_llm::Result<()> {
        let _ = self.cmd.send(Cmd::Abort { external_ids: external_ids.to_vec() });
        Ok(())
    }

    async fn shutdown(self: Box<Self>) -> vllm_llm::Result<()> {
        Ok(())
    }
}

/// Output stream that aborts its request when dropped before the end.
struct GuardedStream {
    inner: futures::stream::BoxStream<'static, vllm_llm::Result<GenerateOutput>>,
    seq_id: u32,
    cmd: mpsc::UnboundedSender<Cmd>,
    finished: bool,
}

impl futures::Stream for GuardedStream {
    type Item = vllm_llm::Result<GenerateOutput>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let poll = self.inner.poll_next_unpin(cx);
        match &poll {
            std::task::Poll::Ready(None) => self.finished = true,
            std::task::Poll::Ready(Some(Ok(out))) if out.finish_reason.is_some() => self.finished = true,
            std::task::Poll::Ready(Some(Err(_))) => self.finished = true,
            _ => {}
        }
        poll
    }
}

impl Drop for GuardedStream {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.cmd.send(Cmd::Drop { seq_id: self.seq_id });
        }
    }
}

struct InFlight {
    /// Media arena ranges held until the request ends.
    ranges: Vec<Range>,
    external_id: String,
    out: Option<OutputTx>,
    prompt: Option<Arc<[u32]>>,
    stop_eos: Option<u32>,
    last_token: Option<u32>,
    finish_tokens: Vec<u32>,
    aborted: bool,
}

struct Actor {
    arena: Option<Arc<MediaArena>>,
    in_flight: Arc<AtomicUsize>,
    dp_size: usize,
    /// Empty while the workers are stopped (sleep level 2).
    pushes: Vec<zeromq::PushSocket>,
    pull: zeromq::PullSocket,
    ids: Arc<std::sync::Mutex<IdPool>>,
    requests: HashMap<u32, InFlight>,
    by_external: HashMap<String, Vec<u32>>,
    next_dp: usize,
    /// Waiters for sleep/wake acks: (reply, acks still expected, acks so far).
    pending_acks: VecDeque<(oneshot::Sender<Vec<serde_json::Value>>, usize, Vec<serde_json::Value>)>,
}

impl Actor {
    async fn run(mut self, mut cmd_rx: mpsc::UnboundedReceiver<Cmd>) -> Result<()> {
        loop {
            tokio::select! {
                cmd = cmd_rx.recv() => {
                    let Some(cmd) = cmd else { return Ok(()) };
                    let mut inputs: Vec<Input> = (0..self.dp_size).map(|_| Input::default()).collect();
                    let mut swap = None;
                    self.apply(cmd, &mut inputs, &mut swap);
                    // Batch everything already queued into this send.
                    while swap.is_none() && let Ok(cmd) = cmd_rx.try_recv() {
                        self.apply(cmd, &mut inputs, &mut swap);
                    }
                    if self.pushes.is_empty() {
                        self.refuse(inputs);
                    } else {
                        for (rank, input) in inputs.into_iter().enumerate() {
                            if !input.is_empty() {
                                self.pushes[rank].send(input.encode().into()).await.context("sending to worker")?;
                            }
                        }
                    }
                    if let Some((pushes, reply)) = swap {
                        self.pushes = pushes;
                        let _ = reply.send(());
                    }
                }
                msg = self.pull.recv() => {
                    let msg = msg.context("receiving from worker")?;
                    for frame in msg.iter() {
                        match Output::decode(frame) {
                            Ok(output) => self.dispatch(output),
                            Err(err) => bail!("undecodable worker output: {err}"),
                        }
                    }
                }
            }
        }
    }

    /// The workers are stopped: fail what would have gone to them. Requests
    /// are held while asleep, so only one that raced the sleep ends up here.
    fn refuse(&mut self, inputs: Vec<Input>) {
        for input in inputs {
            for request in input.requests {
                let seq_id = request.seq_id;
                if let Some(req) = self.requests.get_mut(&seq_id)
                    && let Some(tx) = req.out.take()
                {
                    let _ = tx.send(Err(vllm_llm::Error::Unavailable { message: "the server is asleep; retry".into() }));
                }
                self.retire(seq_id);
            }
            for seq_id in input.aborts {
                self.retire(seq_id);
            }
        }
        // Dropping the waiters fails their commands.
        self.pending_acks.clear();
    }

    fn apply(&mut self, cmd: Cmd, inputs: &mut [Input], swap: &mut Option<(Vec<zeromq::PushSocket>, oneshot::Sender<()>)>) {
        let dp = inputs.len();
        match cmd {
            Cmd::Submit { ranges, request, external_id, prompt, stop_eos, dp_rank, out } => {
                let rank = match dp_rank {
                    Some(r) => r as usize % dp,
                    None => {
                        self.next_dp = (self.next_dp + 1) % dp;
                        self.next_dp
                    }
                };
                let seq_id = request.seq_id;
                // Counted as in flight since admission (`SleepCtl::admit`).
                self.by_external.entry(external_id.clone()).or_default().push(seq_id);
                self.requests.insert(seq_id, InFlight {
                    ranges,
                    external_id,
                    out: Some(out),
                    prompt: Some(prompt),
                    stop_eos,
                    last_token: None,
                    finish_tokens: request.finish_tokens.clone(),
                    aborted: false,
                });
                inputs[rank].requests.push(*request);
            }
            Cmd::Drop { seq_id } => self.abort_seq(seq_id, inputs),
            Cmd::Abort { external_ids } => {
                let seq_ids: Vec<u32> = if external_ids.is_empty() {
                    self.requests.keys().copied().collect()
                } else {
                    external_ids.iter().filter_map(|id| self.by_external.get(id)).flatten().copied().collect()
                };
                for seq_id in seq_ids {
                    self.abort_seq(seq_id, inputs);
                }
            }
            Cmd::Control { control, reply } => {
                let acked = matches!(control.cmd.as_str(), "sleep" | "wake_up");
                for input in inputs.iter_mut() {
                    input.control = Some(control.clone());
                }
                if acked {
                    self.pending_acks.push_back((reply, dp, Vec::new()));
                } else {
                    let _ = reply.send(Vec::new());
                }
            }
            Cmd::Replicas { pushes, reply } => *swap = Some((pushes, reply)),
        }
    }

    /// Abort everywhere: a seq lives on one replica, and the others ignore
    /// unknown ids. The worker answers with the seq in `finished`.
    fn abort_seq(&mut self, seq_id: u32, inputs: &mut [Input]) {
        if let Some(req) = self.requests.get_mut(&seq_id)
            && !req.aborted
        {
            req.aborted = true;
            for input in inputs.iter_mut() {
                input.aborts.push(seq_id);
            }
        }
    }

    fn dispatch(&mut self, output: Output) {
        if let Some(ack) = output.control_ack {
            if let Some(front) = self.pending_acks.front_mut() {
                front.2.push(ack);
                front.1 -= 1;
                if front.1 == 0 {
                    let (reply, _, acks) = self.pending_acks.pop_front().unwrap();
                    let _ = reply.send(acks);
                }
            } else {
                warn!("unexpected control ack");
            }
        }
        let Output { seq_ids, tokens, logprobs, mut prompt_logprobs, finished, mut errors, mut error_codes, .. } = output;
        let mut error = |seq_id: u32| errors.remove(&seq_id).map(|m| (error_codes.remove(&seq_id), m));
        let mut finished_set: std::collections::HashSet<u32> = finished.iter().copied().collect();
        let mut logprobs = logprobs.into_iter();
        for (seq_id, tokens) in seq_ids.into_iter().zip(tokens) {
            let row = logprobs.next().flatten();
            let Some(req) = self.requests.get_mut(&seq_id) else { continue };
            let tokens = tokens.into_vec();
            req.last_token = tokens.last().copied().or(req.last_token);
            let mut out = new_output(&req.external_id, tokens);
            if let Some(prompt) = req.prompt.take() {
                let prompt_lp = prompt_logprobs.remove(&seq_id).map(prompt_logprobs_for_response);
                out.prompt_info = Some(Box::new(GeneratePromptInfo {
                    prompt_token_ids: prompt,
                    prompt_logprobs: prompt_lp,
                    prompt_token_id_logprobs: None,
                }));
            }
            if let Some((sampled, ids, vals)) = row
                && let Some(&token) = out.token_ids.last()
            {
                out.logprobs = Some(Logprobs { positions: vec![position(token, sampled, &ids, &vals)] });
            }
            if finished_set.remove(&seq_id) {
                out.finish_reason = Some(self.finish_reason(seq_id, error(seq_id)));
                self.send(seq_id, out);
                self.retire(seq_id);
            } else {
                self.send(seq_id, out);
            }
        }
        for seq_id in finished {
            if !finished_set.remove(&seq_id) {
                continue; // already retired with its last tokens above
            }
            if !self.requests.contains_key(&seq_id) {
                continue;
            }
            let reason = self.finish_reason(seq_id, error(seq_id));
            let external_id = self.requests[&seq_id].external_id.clone();
            let mut out = new_output(&external_id, Vec::new());
            if let Some(prompt) = self.requests.get_mut(&seq_id).and_then(|r| r.prompt.take()) {
                out.prompt_info = Some(Box::new(GeneratePromptInfo {
                    prompt_token_ids: prompt,
                    prompt_logprobs: None,
                    prompt_token_id_logprobs: None,
                }));
            }
            out.finish_reason = Some(reason);
            self.send(seq_id, out);
            self.retire(seq_id);
        }
    }

    fn finish_reason(&self, seq_id: u32, error: Option<(Option<String>, String)>) -> FinishReason {
        let req = &self.requests[&seq_id];
        if let Some((code, message)) = error {
            return FinishReason::Error(Some(Box::new(RequestError {
                code: code.unwrap_or_else(|| "cache_capacity_exceeded".into()),
                message,
                retryable: false,
            })));
        }
        if req.aborted {
            return FinishReason::Abort;
        }
        match req.last_token {
            Some(t) if Some(t) == req.stop_eos => FinishReason::Stop(None),
            Some(t) if req.finish_tokens.contains(&t) => FinishReason::Stop(Some(StopReason::TokenId(t))),
            _ => FinishReason::Length,
        }
    }

    fn send(&mut self, seq_id: u32, out: GenerateOutput) {
        if let Some(req) = self.requests.get_mut(&seq_id)
            && let Some(tx) = &req.out
            && tx.send(Ok(out)).is_err()
        {
            req.out = None;
        }
    }

    fn retire(&mut self, seq_id: u32) {
        if let Some(req) = self.requests.remove(&seq_id) {
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
            if let Some(arena) = &self.arena {
                for range in req.ranges {
                    arena.release(range);
                }
            }
            if let Some(ids) = self.by_external.get_mut(&req.external_id) {
                ids.retain(|&id| id != seq_id);
                if ids.is_empty() {
                    self.by_external.remove(&req.external_id);
                }
            }
            self.ids.lock().unwrap().free.push_back(seq_id);
        }
    }
}

/// Key under `vllm_xargs` carrying `[[modality, reference], ...]` for the
/// encoders of a disaggregated LM node.
pub const DISAGG_ITEMS_KEY: &str = "gllm_disagg_items";

fn disagg_items(extra: Option<&HashMap<String, serde_json::Value>>) -> vllm_llm::Result<Option<Vec<wire::MediaItem>>> {
    let Some(items) = extra.and_then(|e| e.get(DISAGG_ITEMS_KEY)) else { return Ok(None) };
    let invalid = || vllm_llm::Error::InvalidMultimodalInput { message: "malformed media references".into() };
    let items = items
        .as_array()
        .ok_or_else(invalid)?
        .iter()
        .map(|pair| match (pair[0].as_str(), pair[1].as_str()) {
            (Some(m @ ("image" | "video")), Some(r)) => Ok(wire::MediaItem { modality: m.into(), reference: Some(r.into()), tensors: None }),
            _ => Err(invalid()),
        })
        .collect::<vllm_llm::Result<Vec<_>>>()?;
    Ok((!items.is_empty()).then_some(items))
}

/// Media items in prompt order, and their tensors merged per name: batched
/// fields stack one row per item, flat fields concatenate along dim 0.
/// A merged tensor (without data yet) and its per-item byte chunks.
type MergedTensor = (wire::Tensor, Vec<bytes::Bytes>);

fn merge_media(features: &MmFeatures) -> vllm_llm::Result<(Vec<wire::MediaItem>, Vec<MergedTensor>)> {
    let invalid = |message: String| vllm_llm::Error::InvalidMultimodalInput { message };
    let mut items = Vec::with_capacity(features.len());
    let mut merged: Vec<MergedTensor> = Vec::new();
    for feature in features {
        let modality = match feature.modality {
            MmModality::Image => "image",
            MmModality::Video => "video",
            MmModality::Audio => return Err(vllm_llm::Error::Unsupported { field: "audio inputs" }),
        };
        items.push(wire::MediaItem { modality: modality.into(), reference: None, tensors: None });
        let data = feature.data.as_ref().ok_or_else(|| invalid("media item without data".into()))?;
        for (name, elem) in data {
            let Some(MmKwargValue::Tensor(t)) = &elem.data else {
                return Err(invalid(format!("media field `{name}` is not a tensor")));
            };
            t.validate_inline().map_err(invalid)?;
            let bytes = t.data.as_raw_view().expect("validated inline").clone();
            let batched = match &elem.field {
                MmField::Batched(_) => true,
                MmField::Flat(f) if f.dim == 0 => false,
                _ => return Err(invalid(format!("media field `{name}` has an unsupported layout"))),
            };
            let shape = t.shape.to_vec();
            match merged.iter_mut().find(|(m, _)| &m.name == name) {
                Some((m, chunks)) => {
                    if m.dtype != t.dtype.as_str() || m.batched != batched || m.shape[1..] != shape[(!batched) as usize..] {
                        return Err(invalid(format!("media field `{name}` differs across items")));
                    }
                    m.shape[0] += if batched { 1 } else { shape.first().copied().unwrap_or(0) };
                    chunks.push(bytes);
                }
                None => {
                    let merged_shape = if batched { std::iter::once(1).chain(shape).collect() } else { shape };
                    merged.push((
                        wire::Tensor { name: name.clone(), dtype: t.dtype.as_str().to_owned(), shape: merged_shape,
                                       batched, data: Vec::new(), offset: None },
                        vec![bytes],
                    ));
                }
            }
        }
    }
    Ok((items, merged))
}

fn new_output(request_id: &str, token_ids: Vec<u32>) -> GenerateOutput {
    let n = token_ids.len();
    GenerateOutput {
        request_id: request_id.to_owned(),
        prompt_info: None,
        token_ids,
        logprobs: None,
        finish_reason: None,
        prompt_cache_usage: Default::default(),
        kv_transfer_params: None,
        ec_transfer_params: None,
        sampling_mask: None,
        spec_decode_metrics: None,
        weight_versions: WeightVersions::new(None, n),
        routed_experts: AuxOutputRows::default(),
        timestamps: RequestTimestamps::default(),
    }
}

/// The sampled token first (with its rank), then the top alternatives.
fn position(token: u32, logprob: f32, top_ids: &[u32], top_vals: &[f32]) -> PositionLogprobs {
    let rank = top_ids.iter().position(|&t| t == token).map_or(top_ids.len() + 1, |i| i + 1) as u32;
    let mut entries = Vec::with_capacity(top_ids.len() + 1);
    entries.push(TokenLogprob { token_id: token, logprob, rank });
    for (i, (&token_id, &logprob)) in top_ids.iter().zip(top_vals).enumerate() {
        entries.push(TokenLogprob { token_id, logprob, rank: i as u32 + 1 });
    }
    PositionLogprobs { entries }
}

/// Workers keep one entry per prompt position (the first is always empty);
/// the response lists positions 1.. only.
fn prompt_logprobs_for_response(raw: Vec<Option<wire::PromptLogprob>>) -> Logprobs {
    let positions = raw
        .into_iter()
        .skip(1)
        .map(|entry| match entry {
            Some((token, logprob, ids, vals)) => position(token, logprob, &ids, &vals),
            None => PositionLogprobs { entries: Vec::new() },
        })
        .collect();
    Logprobs { positions }
}

