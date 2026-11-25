//! gLLM's HTTP front end.
//!
//! Starts the Python engine (`gllm.entrypoints.engine_server`), which only
//! launches and supervises the workers, then serves the OpenAI API and talks
//! to the workers directly over ZMQ.

mod arena;
mod backend;
mod http;
mod launcher;
mod render;
mod sleep;
mod wire;

use std::sync::Arc;

use anyhow::{Context as _, Result};
use clap::Parser;
use tracing::{error, info, warn};
use vllm_chat::{ChatLlm, LoadModelBackendsOptions, ParserSelection};
use vllm_text::TextLlm;

use crate::arena::MediaArena;
use crate::backend::GllmLlm;
use crate::launcher::Launcher;

#[derive(Debug, Parser)]
#[command(
    name = "gllm-frontend",
    about = "OpenAI-compatible server for gLLM",
    after_help = "Engine arguments go after `--`, e.g. `gllm-frontend --model-path M -- --tp 2`."
)]
struct Args {
    /// Model directory (or Hub id) served by the engine.
    #[arg(long)]
    model_path: String,
    #[arg(long, default_value = "0.0.0.0")]
    host: String,
    #[arg(long, default_value_t = 8000)]
    port: u16,
    /// Names a request's `model` may use; the first one is reported.
    #[arg(long, num_args = 1..)]
    served_model_name: Vec<String>,
    /// Tool-call parser (e.g. qwen3_coder, deepseek_v4); default: by architecture.
    #[arg(long)]
    tool_call_parser: Option<String>,
    /// Reasoning parser (e.g. qwen3, deepseek_v41); default: by architecture, else auto-detect.
    #[arg(long)]
    reasoning_parser: Option<String>,
    /// Shared-memory arena for media tensors, in GiB (0 sends them inline).
    #[arg(long, default_value_t = 4.0)]
    media_arena_gb: f64,
    /// Largest accepted request body, in MiB (data-URL media count in full).
    #[arg(long, default_value_t = 1024)]
    max_request_mb: usize,
    /// Python interpreter that runs the engine; default: the `python` installed
    /// next to this executable, else `python` on PATH.
    #[arg(long)]
    python: Option<String>,
    /// Longest wait, in seconds, for the engine to restart when waking from
    /// sleep level 2; on timeout the wake-up fails and the server stays asleep.
    #[arg(long, default_value_t = 1800)]
    wake_timeout_s: u64,
    /// Log filter, e.g. `info` or `gllm_frontend=debug,info`.
    #[arg(long, default_value = "info")]
    log_level: String,
    /// Debug: print the prompt token ids the chat stack builds for this
    /// chat-completions request (a JSON file) and exit; no engine is started.
    #[arg(long, hide = true)]
    render_request: Option<std::path::PathBuf>,
    /// Arguments forwarded to `gllm.entrypoints.engine_server`.
    #[arg(last = true)]
    engine_args: Vec<String>,
}

fn chat_parsers(args: &Args) -> (ParserSelection, ParserSelection) {
    let tool = match args.tool_call_parser.clone() {
        Some(name) if name == "none" => ParserSelection::None,
        Some(name) => ParserSelection::Explicit(name),
        None => tool_parser_for(&args.model_path).map_or(ParserSelection::Auto, |n| ParserSelection::Explicit(n.into())),
    };
    let reasoning = match args.reasoning_parser.clone() {
        Some(name) if name == "none" => ParserSelection::None,
        Some(name) => ParserSelection::Explicit(name),
        None => reasoning_parser_for(&args.model_path).map_or(ParserSelection::Auto, |n| ParserSelection::Explicit(n.into())),
    };
    (tool, reasoning)
}

fn architecture(model_path: &str) -> Option<String> {
    let config = std::fs::read_to_string(std::path::Path::new(model_path).join("config.json")).ok()?;
    let config: serde_json::Value = serde_json::from_str(&config).ok()?;
    Some(config["architectures"][0].as_str()?.to_lowercase())
}

/// A DeepSeek V4 checkpoint that ships its encoder without the 0731 effort
/// prompts: there only `max` adds a prefix, and it is the text the renderer
/// uses for `high`.
pub fn legacy_v4_effort(model_path: &str) -> bool {
    if !architecture(model_path).is_some_and(|a| a.contains("deepseekv4") && !a.contains("deepseekv41")) {
        return false;
    }
    let dir = std::path::Path::new(model_path).join("encoding");
    let cached = std::fs::read_dir(dir.join("__pycache__")).into_iter().flatten().flatten().map(|e| e.path());
    let encoder = std::iter::once(dir.join("encoding_dsv4.py"))
        .chain(cached.filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("encoding_dsv4."))))
        .find_map(|p| std::fs::read(p).ok());
    encoder.is_some_and(|src| !src.windows(15).any(|w| w == b"Beyond maximum "))
}

/// DeepSeek V4/V4.1 thinking markup differs per release; pick it from the
/// architecture so a checkpoint directory name does not matter.
fn reasoning_parser_for(model_path: &str) -> Option<&'static str> {
    let arch = architecture(model_path)?;
    if arch.contains("deepseekv41") {
        Some("deepseek_v41")
    } else if arch.contains("deepseekv4") {
        Some("deepseek_v4")
    } else {
        None
    }
}

fn default_python() -> String {
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join("python")))
        .filter(|p| p.exists())
        .map_or_else(|| "python".to_owned(), |p| p.display().to_string())
}

/// Pick the tool-call markup from the model architecture, as the Python front
/// end did: Qwen3.5+ emits `<function=..><parameter=..>`, older Qwen Hermes JSON.
fn tool_parser_for(model_path: &str) -> Option<&'static str> {
    let arch = architecture(model_path)?;
    if arch.contains("qwen3_5") || arch.contains("qwen3.5") {
        Some("qwen3_coder")
    } else if arch.contains("qwen") {
        Some("qwen3_json")
    } else if arch.contains("kimi") {
        Some("kimi_k2")
    } else if arch.contains("deepseekv32") || arch.contains("deepseek_v32") {
        Some("deepseek_v32")
    } else if arch.contains("deepseekv41") {
        Some("deepseek_v41")
    } else if arch.contains("deepseekv4") || arch.contains("deepseek_v4") {
        Some("deepseek_v4")
    } else {
        None
    }
}

/// Media decoding loads two things at run time; say up front when one is
/// missing instead of failing (video) or drifting from the reference decoder
/// (JPEG) on the first request that needs it.
fn check_media_runtime() {
    let turbojpeg = ["libturbojpeg.so.0", "libturbojpeg.so"].iter().any(|name| {
        let name = std::ffi::CString::new(*name).unwrap();
        // SAFETY: probing a shared library by name; the handle is closed at once.
        unsafe {
            let handle = libc::dlopen(name.as_ptr(), libc::RTLD_LAZY | libc::RTLD_LOCAL);
            !handle.is_null() && libc::dlclose(handle) == 0
        }
    });
    if !turbojpeg {
        warn!(
            "libturbojpeg.so.0 not found on the loader path: JPEG images fall back to another \
             decoder whose pixels differ slightly from PIL's. Install libturbojpeg (e.g. \
             `apt install libturbojpeg0` or `conda install libjpeg-turbo`) or add its directory \
             to LD_LIBRARY_PATH."
        );
    }
    let ffmpeg = std::env::var_os("PATH")
        .is_some_and(|paths| std::env::split_paths(&paths).any(|dir| dir.join("ffmpeg").is_file()));
    if !ffmpeg {
        warn!(
            "`ffmpeg` not found on PATH: video requests will fail. Install ffmpeg \
             (e.g. `apt install ffmpeg` or `conda install ffmpeg`)."
        );
    }
}

async fn shutdown_signal() {
    let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).unwrap();
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        _ = term.recv() => {}
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_new(&args.log_level).context("parsing --log-level")?)
        .init();
    if let Some(path) = &args.render_request {
        return render::run(&args.model_path, path, chat_parsers(&args)).await;
    }

    let ipc_prefix = uuid::Uuid::new_v4().simple().to_string();
    let output = GllmLlm::bind_output(&ipc_prefix).await?;
    // Tokenizer and chat template load while the workers start.
    let model_path = args.model_path.clone();
    let backends = tokio::spawn(async move {
        vllm_chat::load_model_backends(&model_path, LoadModelBackendsOptions::default()).await
    });
    let arena = (args.media_arena_gb > 0.0)
        .then(|| MediaArena::create(&format!("gllm_{ipc_prefix}_media"), (args.media_arena_gb * (1u64 << 30) as f64) as usize))
        .transpose()?
        .map(Arc::new);
    let (launcher, mut engine_exit) = Launcher::new(
        args.python.clone().unwrap_or_else(default_python),
        args.model_path.clone(),
        args.engine_args.clone(),
        ipc_prefix.clone(),
        arena.as_ref().map(|a| a.path.clone()),
        std::time::Duration::from_secs(args.wake_timeout_s),
    );
    let ready = launcher.start(None).await?;
    info!(dp_size = ready.dp_size, max_model_len = ready.max_model_len, "engine ready");
    if ready.use_mm && !ready.disagg_lm {
        check_media_runtime();
    }
    let loaded = backends.await?.context("loading tokenizer and chat template")?;
    let tokenizer = loaded.text_backend.tokenizer();
    let max_model_len = ready.max_model_len;
    let disagg_lm = ready.disagg_lm;
    let llm = GllmLlm::connect(ready, output, Some(tokenizer), arena, Some(launcher.clone())).await?;
    let text = TextLlm::new(llm, loaded.text_backend);
    let (tool_parser, reasoning_parser) = chat_parsers(&args);
    info!(?tool_parser, ?reasoning_parser, "parsers");
    let chat = ChatLlm::new(text, loaded.chat_backend)
        .with_tool_call_parser(tool_parser)
        .with_reasoning_parser(reasoning_parser);

    let model_names = if args.served_model_name.is_empty() {
        vec![args.model_path.clone()]
    } else {
        args.served_model_name.clone()
    };
    let state = Arc::new(http::AppState {
        chat,
        responses: Default::default(),
        model_path: args.model_path.clone(),
        model_names,
        max_model_len,
        disagg_lm,
        legacy_v4_effort: legacy_v4_effort(&args.model_path),
    });
    let app = http::router(state, args.max_request_mb << 20);
    let listener = tokio::net::TcpListener::bind((args.host.as_str(), args.port))
        .await
        .with_context(|| format!("binding {}:{}", args.host, args.port))?;
    info!("serving on http://{}:{}", args.host, args.port);

    let result = tokio::select! {
        r = axum::serve(listener, app).with_graceful_shutdown(shutdown_signal()) => r.map_err(Into::into),
        status = engine_exit.recv() => {
            error!("engine exited: {status:?}");
            Err(anyhow::anyhow!("engine exited"))
        }
    };
    launcher.stop().await;
    result
}
