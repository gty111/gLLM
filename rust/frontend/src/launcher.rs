//! The Python engine process (`gllm.entrypoints.engine_server`).
//!
//! Started once at boot. Sleep level 2 stops it, which takes its workers and
//! every byte of GPU memory they own with it, and starts it again on wake-up.
//! An exit nobody asked for is reported on the `fatal` channel.

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context as _, Result, bail};
use tokio::io::{AsyncBufReadExt as _, BufReader};
use tokio::process::Command;
use tokio::sync::{Mutex, mpsc, oneshot};
use tracing::warn;

use crate::backend::EngineReady;

/// Grace period for the engine (and then its workers) after SIGTERM.
const STOP_GRACE: Duration = Duration::from_secs(30);

pub struct Launcher {
    pub python: String,
    pub model_path: String,
    pub engine_args: Vec<String>,
    pub ipc_prefix: String,
    pub arena_path: Option<PathBuf>,
    /// Longest wait for a restarted engine's ready line (wake from sleep level 2).
    pub wake_timeout: Duration,
    fatal: mpsc::UnboundedSender<String>,
    running: Mutex<Option<Running>>,
}

struct Running {
    /// Also the process group of the engine and its workers.
    pid: i32,
    stopping: Arc<AtomicBool>,
    exited: oneshot::Receiver<()>,
}

fn signal_group(pid: i32, signal: libc::c_int) -> bool {
    // SAFETY: plain kill(2) on the engine's process group.
    unsafe { libc::kill(-pid, signal) == 0 }
}

impl Launcher {
    pub fn new(
        python: String,
        model_path: String,
        engine_args: Vec<String>,
        ipc_prefix: String,
        arena_path: Option<PathBuf>,
        wake_timeout: Duration,
    ) -> (Arc<Self>, mpsc::UnboundedReceiver<String>) {
        let (fatal, fatal_rx) = mpsc::unbounded_channel();
        let launcher =
            Self { python, model_path, engine_args, ipc_prefix, arena_path, wake_timeout, fatal, running: Mutex::new(None) };
        (Arc::new(launcher), fatal_rx)
    }

    /// Start the engine in its own process group and wait for its ready line,
    /// at most `timeout` if given. A previous engine still running is stopped
    /// first.
    pub async fn start(&self, timeout: Option<Duration>) -> Result<EngineReady> {
        let mut running = self.running.lock().await;
        if let Some(old) = running.take() {
            stop(old).await;
        }
        let mut command = Command::new(&self.python);
        command.args(["-m", "gllm.entrypoints.engine_server", "--ipc-prefix", &self.ipc_prefix]);
        if let Some(path) = &self.arena_path {
            command.arg("--media-arena").arg(path);
        }
        let mut child = command
            .args(["--model-path", &self.model_path])
            .args(&self.engine_args)
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .process_group(0)
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("starting the engine with `{}`", self.python))?;
        let pid = child.id().context("engine pid")? as i32;
        let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();
        let read_ready = async {
            loop {
                match lines.next_line().await? {
                    Some(line) => match line.strip_prefix("GLLM_READY ") {
                        Some(json) => return serde_json::from_str(json).context("parsing the engine ready line"),
                        None => println!("{line}"),
                    },
                    None => bail!("engine exited before it was ready: {:?}", child.wait().await?),
                }
            }
        };
        let ready: Result<EngineReady> = match timeout {
            None => read_ready.await,
            Some(limit) => tokio::time::timeout(limit, read_ready)
                .await
                .unwrap_or_else(|_| Err(anyhow::anyhow!("engine was not ready within {}s", limit.as_secs()))),
        };
        let ready = match ready {
            Ok(ready) => ready,
            Err(err) => {
                // Workers it already started must not keep their GPU memory.
                signal_group(pid, libc::SIGKILL);
                let _ = child.wait().await;
                wait_group_gone(pid).await;
                return Err(err);
            }
        };
        tokio::spawn(async move {
            while let Ok(Some(line)) = lines.next_line().await {
                println!("{line}");
            }
        });
        let stopping = Arc::new(AtomicBool::new(false));
        let (exited_tx, exited) = oneshot::channel();
        let (fatal, asked) = (self.fatal.clone(), stopping.clone());
        tokio::spawn(async move {
            let status = child.wait().await;
            if !asked.load(Ordering::SeqCst) {
                let _ = fatal.send(format!("{status:?}"));
            }
            let _ = exited_tx.send(());
        });
        *running = Some(Running { pid, stopping, exited });
        Ok(ready)
    }

    /// Stop the engine and wait until its workers are gone.
    pub async fn stop(&self) {
        if let Some(running) = self.running.lock().await.take() {
            stop(running).await;
        }
    }
}

async fn stop(running: Running) {
    let Running { pid, stopping, mut exited } = running;
    stopping.store(true, Ordering::SeqCst);
    // The engine's workers share its process group.
    signal_group(pid, libc::SIGTERM);
    if tokio::time::timeout(STOP_GRACE, &mut exited).await.is_err() {
        warn!("engine did not exit within {}s of SIGTERM; killing it", STOP_GRACE.as_secs());
        signal_group(pid, libc::SIGKILL);
        let _ = exited.await;
    }
    // The engine joins its workers on exit; make sure none is left holding
    // GPU memory before a new engine starts.
    wait_group_gone(pid).await;
}

/// Wait until no process of the engine's group is left, killing stragglers
/// after the grace period.
async fn wait_group_gone(pid: i32) {
    let deadline = tokio::time::Instant::now() + STOP_GRACE;
    while signal_group(pid, 0) {
        if tokio::time::Instant::now() > deadline {
            signal_group(pid, libc::SIGKILL);
            warn!("engine workers outlived the engine; killed them");
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An "interpreter" that starts a worker and never prints the ready line.
    fn hanging_engine(dir: &std::path::Path) -> String {
        use std::os::unix::fs::PermissionsExt as _;
        let path = dir.join("engine.sh");
        std::fs::write(&path, "#!/bin/sh\nsleep 600 &\necho $! > \"$(dirname \"$0\")/worker.pid\"\nwait\n").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path.to_string_lossy().into_owned()
    }

    #[tokio::test]
    async fn engine_that_never_gets_ready_times_out_and_is_killed() {
        let dir = std::env::temp_dir().join(format!("gllm_launcher_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (launcher, _fatal) =
            Launcher::new(hanging_engine(&dir), "m".into(), vec![], "t".into(), None, Duration::from_millis(500));
        let err = launcher.start(Some(launcher.wake_timeout)).await.unwrap_err();
        assert!(err.to_string().contains("not ready"), "{err}");
        let worker: i32 = std::fs::read_to_string(dir.join("worker.pid")).unwrap().trim().parse().unwrap();
        // SAFETY: probing a pid with signal 0.
        assert!(unsafe { libc::kill(worker, 0) } != 0, "the engine's worker is still running");
        assert!(launcher.running.lock().await.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
