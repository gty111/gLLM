//! Sleep mode: release / restore worker GPU memory.
//!
//! Level 1 releases memory by tag while the workers stay up. Level 2 stops the
//! engine and its workers (all their GPU memory goes) and starts them again on
//! wake-up. New requests are held from the start of a sleep until the engine is
//! fully awake again; a request that arrives while asleep wakes the engine.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use tokio::sync::{Mutex, watch};
use tracing::{error, info};

pub const ALL_TAGS: [&str; 2] = ["kv_cache", "weights"];

/// Start of the error a held request gets when the automatic wake-up fails.
pub const WAKE_FAILED: &str = "server is asleep and failed to wake up";

#[derive(Debug, thiserror::Error)]
pub enum SleepError {
    #[error("{0}")]
    BadRequest(String),
    #[error("{0}")]
    Failed(String),
    #[error("sleep mode is disabled (--no-sleep-mode)")]
    Disabled,
}

#[derive(Clone, Debug)]
struct Gate {
    holding: bool,
    /// Bumped whenever an automatic wake fails; held requests then give up.
    failures: u64,
    failure: Option<String>,
}

/// What the sleep controller needs from the engine.
#[async_trait::async_trait]
pub trait MemoryCommand: Send + Sync {
    /// Send a memory command to every replica and return their acks.
    async fn memory_command(&self, name: &str, tags: Vec<String>) -> anyhow::Result<Vec<Value>>;
    async fn abort_all(&self);
    /// Level 2: stop the engine and wait until its workers are gone.
    async fn stop_workers(&self) -> anyhow::Result<()>;
    /// Level 2 wake-up: start the engine again and reconnect to its workers.
    async fn start_workers(&self) -> anyhow::Result<()>;
}

pub struct SleepCtl {
    enabled: bool,
    /// Level 2 is available (the engine's workers are all local).
    full_sleep: bool,
    idle_level: u32,
    /// Level 2: the workers are stopped.
    stopped: AtomicBool,
    transition: Mutex<()>,
    sleeping: std::sync::Mutex<BTreeSet<String>>,
    gate: watch::Sender<Gate>,
    waiting: AtomicUsize,
    in_flight: Arc<AtomicUsize>,
    last_activity: std::sync::Mutex<Instant>,
    wake_running: std::sync::atomic::AtomicBool,
}

pub fn normalize_tags(tags: Option<Vec<String>>) -> Result<Vec<String>, SleepError> {
    let mut set: BTreeSet<String> = tags.unwrap_or_default().into_iter().collect();
    if set.is_empty() {
        set = ALL_TAGS.iter().map(|t| t.to_string()).collect();
    }
    let unknown: Vec<_> = set.iter().filter(|t| !ALL_TAGS.contains(&t.as_str())).cloned().collect();
    if !unknown.is_empty() {
        return Err(SleepError::BadRequest(format!(
            "unknown memory tags {unknown:?}; valid tags are {ALL_TAGS:?}"
        )));
    }
    Ok(set.into_iter().collect())
}

fn summarize(acks: &[Value], tags: &[String]) -> Value {
    let mut gib = serde_json::Map::new();
    for t in tags {
        let bytes: f64 = acks.iter().map(|a| a["bytes"][t].as_f64().unwrap_or(0.0)).sum();
        gib.insert(t.clone(), json!(((bytes / (1u64 << 30) as f64) * 1000.0).round() / 1000.0));
    }
    let seconds = acks.iter().map(|a| a["seconds"].as_f64().unwrap_or(0.0)).fold(0.0, f64::max);
    json!({"tags": tags, "gib": gib, "seconds": (seconds * 1000.0).round() / 1000.0})
}

fn check_acks(name: &str, acks: &[Value]) -> Result<(), SleepError> {
    let mut errors: Vec<String> = acks
        .iter()
        .filter(|a| !a["ok"].as_bool().unwrap_or(false))
        .map(|a| a["error"].as_str().unwrap_or("unknown error").to_owned())
        .collect();
    errors.sort();
    errors.dedup();
    if errors.is_empty() { Ok(()) } else { Err(SleepError::Failed(format!("{name} failed: {}", errors.join("; ")))) }
}

/// One held request, counted in `SleepCtl::waiting` while it lives.
struct Waiting<'a>(&'a AtomicUsize);

impl<'a> Waiting<'a> {
    fn new(count: &'a AtomicUsize) -> Self {
        count.fetch_add(1, Ordering::SeqCst);
        Self(count)
    }
}

impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

impl SleepCtl {
    pub fn new(enabled: bool, full_sleep: bool, idle_level: u32, in_flight: Arc<AtomicUsize>) -> Arc<Self> {
        Arc::new(Self {
            enabled,
            full_sleep,
            idle_level,
            stopped: AtomicBool::new(false),
            transition: Mutex::new(()),
            sleeping: std::sync::Mutex::new(BTreeSet::new()),
            gate: watch::channel(Gate { holding: false, failures: 0, failure: None }).0,
            waiting: AtomicUsize::new(0),
            in_flight,
            last_activity: std::sync::Mutex::new(Instant::now()),
            wake_running: std::sync::atomic::AtomicBool::new(false),
        })
    }

    pub fn sleeping_tags(&self) -> Vec<String> {
        self.sleeping.lock().unwrap().iter().cloned().collect()
    }

    /// 0 awake, 1 some tags released, 2 workers stopped.
    pub fn level(&self) -> u32 {
        if self.stopped.load(Ordering::SeqCst) {
            2
        } else {
            u32::from(!self.sleeping.lock().unwrap().is_empty())
        }
    }

    fn finish(&self, mut result: Value) -> Value {
        result["sleeping_tags"] = json!(self.sleeping_tags());
        result["level"] = json!(self.level());
        result
    }

    pub fn touch(&self) {
        *self.last_activity.lock().unwrap() = Instant::now();
    }

    fn set_holding(&self, holding: bool) {
        self.gate.send_modify(|g| g.holding = holding);
    }

    /// Wait until requests may run; wakes the engine if it is asleep.
    ///
    /// On success the request counts as in flight from here on (the engine
    /// retires it, or the caller gives it back with [`Self::unadmit`]), so a
    /// sleep that starts while it is still being prepared waits for it.
    pub async fn admit(self: &Arc<Self>, engine: Arc<dyn MemoryCommand>) -> Result<(), String> {
        let mut rx = self.gate.subscribe();
        let start = rx.borrow().failures;
        let mut waiting = None;
        let result = loop {
            // Count first, then look at the gate: a sleep raises the gate
            // first, then waits for the count, so one of them sees the other.
            self.in_flight.fetch_add(1, Ordering::SeqCst);
            let (holding, failures, failure) = {
                let g = rx.borrow_and_update();
                (g.holding, g.failures, g.failure.clone())
            };
            if !holding {
                break Ok(());
            }
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
            if failures != start {
                break Err(failure.unwrap_or_default());
            }
            if waiting.is_none() {
                // Dropped also when the caller goes away while held.
                waiting = Some(Waiting::new(&self.waiting));
                self.ensure_wake(engine.clone());
            }
            if rx.changed().await.is_err() {
                break Err("engine stopped".into());
            }
        };
        drop(waiting);
        self.touch();
        result
    }

    /// Give back an admission whose request never reached the engine.
    pub fn unadmit(&self) {
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
    }

    fn ensure_wake(self: &Arc<Self>, engine: Arc<dyn MemoryCommand>) {
        if self.wake_running.swap(true, Ordering::SeqCst) {
            return;
        }
        let this = self.clone();
        tokio::spawn(async move {
            // Takes the transition lock, so an in-progress sleep finishes first.
            if let Err(err) = this.wake_up(engine.as_ref(), None).await {
                error!("automatic wake-up failed: {err}");
                let msg = format!("{WAKE_FAILED}: {err}");
                this.gate.send_modify(|g| {
                    g.failures += 1;
                    g.failure = Some(msg);
                });
            }
            this.wake_running.store(false, Ordering::SeqCst);
        });
    }

    pub async fn sleep(
        self: &Arc<Self>,
        engine: &Arc<dyn MemoryCommand>,
        tags: Option<Vec<String>>,
        abort: bool,
        level: u32,
    ) -> Result<Value, SleepError> {
        if !self.enabled {
            return Err(SleepError::Disabled);
        }
        let tags = normalize_tags(tags)?;
        if level != 1 && level != 2 {
            return Err(SleepError::BadRequest(format!("invalid sleep level {level}; expected 1 or 2")));
        }
        if level == 2 {
            if !self.full_sleep {
                return Err(SleepError::BadRequest(
                    "sleep level 2 requires a local, non-disaggregated server".into(),
                ));
            }
            if tags.len() != ALL_TAGS.len() {
                return Err(SleepError::BadRequest("sleep level 2 requires all tags".into()));
            }
        }
        let _guard = self.transition.lock().await;
        if self.stopped.load(Ordering::SeqCst) {
            return Ok(self.finish(json!({"tags": [], "gib": {}, "seconds": 0.0})));
        }
        if level == 2 {
            let result = self.stop_workers(engine, abort).await;
            let waiting = self.waiting.load(Ordering::SeqCst) > 0;
            drop(_guard);
            if waiting {
                self.ensure_wake(engine.clone());
            }
            return result.map(|r| self.finish(r));
        }
        let todo: Vec<String> = {
            let sleeping = self.sleeping.lock().unwrap();
            tags.into_iter().filter(|t| !sleeping.contains(t)).collect()
        };
        let mut result = json!({"tags": [], "gib": {}, "seconds": 0.0});
        if !todo.is_empty() {
            self.set_holding(true);
            if abort {
                engine.abort_all().await;
            }
            while self.in_flight.load(Ordering::SeqCst) > 0 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            let acks = engine.memory_command("sleep", todo.clone()).await.map_err(|e| SleepError::Failed(e.to_string()));
            let acks = match acks.and_then(|acks| check_acks("sleep", &acks).map(|_| acks)) {
                Ok(acks) => acks,
                Err(err) => {
                    // Some ranks may have released memory: wake them all again.
                    if engine.memory_command("wake_up", todo.clone()).await.is_err() {
                        self.sleeping.lock().unwrap().extend(todo.iter().cloned());
                    }
                    let holding = !self.sleeping.lock().unwrap().is_empty();
                    self.set_holding(holding);
                    return Err(err);
                }
            };
            self.sleeping.lock().unwrap().extend(todo.iter().cloned());
            result = summarize(&acks, &todo);
            info!("engine asleep ({}): {}", self.sleeping_tags().join(","), result);
        }
        let waiting = self.waiting.load(Ordering::SeqCst) > 0;
        drop(_guard);
        if waiting {
            // Requests that arrived while falling asleep.
            self.ensure_wake(engine.clone());
        }
        Ok(self.finish(result))
    }

    /// Level 2: drain (or abort), then stop the engine. Called under the
    /// transition lock.
    async fn stop_workers(&self, engine: &Arc<dyn MemoryCommand>, abort: bool) -> Result<Value, SleepError> {
        self.set_holding(true);
        if abort {
            engine.abort_all().await;
        }
        while self.in_flight.load(Ordering::SeqCst) > 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let start = Instant::now();
        // From here on the workers may be gone, whatever the outcome; waking
        // up starts a fresh engine either way.
        self.stopped.store(true, Ordering::SeqCst);
        self.sleeping.lock().unwrap().extend(ALL_TAGS.iter().map(|t| t.to_string()));
        engine.stop_workers().await.map_err(|e| SleepError::Failed(format!("sleep level 2 failed: {e:#}")))?;
        let seconds = (start.elapsed().as_secs_f64() * 1000.0).round() / 1000.0;
        info!("engine asleep (level 2): workers stopped in {seconds:.2}s");
        Ok(json!({"tags": self.sleeping_tags(), "gib": {}, "seconds": seconds}))
    }

    pub async fn wake_up(&self, engine: &dyn MemoryCommand, tags: Option<Vec<String>>) -> Result<Value, SleepError> {
        if !self.enabled {
            return Err(SleepError::Disabled);
        }
        let tags = normalize_tags(tags)?;
        let _guard = self.transition.lock().await;
        let todo: Vec<String> = {
            let sleeping = self.sleeping.lock().unwrap();
            tags.iter().filter(|t| sleeping.contains(*t)).cloned().collect()
        };
        let mut result = json!({"tags": [], "gib": {}, "seconds": 0.0});
        if self.stopped.load(Ordering::SeqCst) {
            if tags.len() != ALL_TAGS.len() {
                return Err(SleepError::BadRequest("waking from sleep level 2 requires all tags".into()));
            }
            let start = Instant::now();
            engine
                .start_workers()
                .await
                .map_err(|e| SleepError::Failed(format!("wake-up from level 2 failed: {e:#}")))?;
            let woke = self.sleeping_tags();
            self.sleeping.lock().unwrap().clear();
            self.stopped.store(false, Ordering::SeqCst);
            let seconds = (start.elapsed().as_secs_f64() * 1000.0).round() / 1000.0;
            info!("engine woke from level 2: workers started in {seconds:.2}s");
            result = json!({"tags": woke, "gib": {}, "seconds": seconds});
        } else if !todo.is_empty() {
            let acks = engine.memory_command("wake_up", todo.clone()).await.map_err(|e| SleepError::Failed(e.to_string()))?;
            check_acks("wake_up", &acks)?;
            {
                let mut sleeping = self.sleeping.lock().unwrap();
                for t in &todo {
                    sleeping.remove(t);
                }
            }
            result = summarize(&acks, &todo);
            info!("engine woke {}: {}", todo.join(","), result);
        }
        if self.sleeping.lock().unwrap().is_empty() {
            self.set_holding(false);
            self.touch();
        }
        Ok(self.finish(result))
    }

    /// Sleep after `timeout` without requests in flight.
    pub fn spawn_idle_watcher(self: &Arc<Self>, engine: Arc<dyn MemoryCommand>, timeout: Duration) {
        if !self.enabled || timeout.is_zero() {
            return;
        }
        info!("idle sleep enabled: sleep after {:.1}s without requests", timeout.as_secs_f64());
        let this = self.clone();
        tokio::spawn(async move {
            let poll = (timeout / 4).clamp(Duration::from_millis(50), Duration::from_secs(1));
            loop {
                tokio::time::sleep(poll).await;
                let busy = !this.sleeping.lock().unwrap().is_empty()
                    || this.transition.try_lock().is_err()
                    || this.in_flight.load(Ordering::SeqCst) > 0
                    || this.waiting.load(Ordering::SeqCst) > 0;
                if busy {
                    this.touch();
                    continue;
                }
                if this.last_activity.lock().unwrap().elapsed() < timeout {
                    continue;
                }
                info!("no requests for {:.1}s; going to sleep", timeout.as_secs_f64());
                if let Err(err) = this.sleep(&engine, None, false, this.idle_level).await {
                    error!("idle sleep failed: {err}");
                }
                this.touch();
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Records the calls; `start_fails` makes a restart fail.
    #[derive(Default)]
    struct Fake {
        events: std::sync::Mutex<Vec<String>>,
        start_fails: AtomicBool,
    }

    #[async_trait::async_trait]
    impl MemoryCommand for Fake {
        async fn memory_command(&self, name: &str, tags: Vec<String>) -> anyhow::Result<Vec<Value>> {
            self.events.lock().unwrap().push(format!("{name}:{}", tags.join(",")));
            Ok(vec![json!({"ok": true, "bytes": {}, "seconds": 0.0})])
        }
        async fn abort_all(&self) {
            self.events.lock().unwrap().push("abort".into());
        }
        async fn stop_workers(&self) -> anyhow::Result<()> {
            self.events.lock().unwrap().push("stop".into());
            Ok(())
        }
        async fn start_workers(&self) -> anyhow::Result<()> {
            self.events.lock().unwrap().push("start".into());
            if self.start_fails.load(Ordering::SeqCst) { anyhow::bail!("worker died") } else { Ok(()) }
        }
    }

    fn setup(full_sleep: bool) -> (Arc<SleepCtl>, Arc<Fake>, Arc<dyn MemoryCommand>) {
        let ctl = SleepCtl::new(true, full_sleep, 1, Arc::new(AtomicUsize::new(0)));
        let fake = Arc::new(Fake::default());
        let engine: Arc<dyn MemoryCommand> = fake.clone();
        (ctl, fake, engine)
    }

    fn tags(t: &[&str]) -> Option<Vec<String>> {
        Some(t.iter().map(|s| s.to_string()).collect())
    }

    #[tokio::test]
    async fn level_two_stops_and_restarts_workers() {
        let (ctl, fake, engine) = setup(true);
        ctl.sleep(&engine, tags(&["weights"]), false, 1).await.unwrap();
        assert_eq!(ctl.level(), 1);
        let r = ctl.sleep(&engine, None, true, 2).await.unwrap();
        assert_eq!(r["level"], 2);
        assert_eq!(ctl.sleeping_tags(), ["kv_cache", "weights"]);
        // Another sleep while stopped is a no-op.
        assert_eq!(ctl.sleep(&engine, None, false, 1).await.unwrap()["level"], 2);
        assert!(matches!(ctl.wake_up(engine.as_ref(), tags(&["weights"])).await, Err(SleepError::BadRequest(_))));
        // A request wakes it.
        ctl.admit(engine.clone()).await.unwrap();
        assert_eq!(ctl.level(), 0);
        assert!(ctl.sleeping_tags().is_empty());
        assert_eq!(*fake.events.lock().unwrap(), ["sleep:weights", "abort", "stop", "start"]);
    }

    #[tokio::test]
    async fn failed_restart_stays_asleep_and_retries() {
        let (ctl, fake, engine) = setup(true);
        ctl.sleep(&engine, None, false, 2).await.unwrap();
        fake.start_fails.store(true, Ordering::SeqCst);
        assert!(matches!(ctl.wake_up(engine.as_ref(), None).await, Err(SleepError::Failed(_))));
        assert_eq!(ctl.level(), 2);
        assert!(ctl.admit(engine.clone()).await.is_err());
        fake.start_fails.store(false, Ordering::SeqCst);
        assert_eq!(ctl.wake_up(engine.as_ref(), None).await.unwrap()["level"], 0);
    }

    #[tokio::test]
    async fn sleep_waits_for_an_admitted_request() {
        let (ctl, fake, engine) = setup(true);
        ctl.admit(engine.clone()).await.unwrap();
        let sleeper = {
            let (ctl, engine) = (ctl.clone(), engine.clone());
            tokio::spawn(async move { ctl.sleep(&engine, None, false, 2).await })
        };
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(fake.events.lock().unwrap().is_empty(), "stopped workers under an admitted request");
        ctl.unadmit();
        assert_eq!(sleeper.await.unwrap().unwrap()["level"], 2);
        assert_eq!(*fake.events.lock().unwrap(), ["stop"]);
    }

    #[tokio::test]
    async fn a_held_request_that_goes_away_is_not_counted() {
        let (ctl, _, engine) = setup(true);
        ctl.sleep(&engine, None, false, 1).await.unwrap();
        ctl.wake_running.store(true, Ordering::SeqCst); // keep it asleep
        assert!(tokio::time::timeout(Duration::from_millis(50), ctl.admit(engine.clone())).await.is_err());
        assert_eq!(ctl.waiting.load(Ordering::SeqCst), 0);
        assert_eq!(ctl.in_flight.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn level_two_input_is_checked() {
        let (ctl, _, engine) = setup(true);
        for (t, level) in [(tags(&["weights"]), 2), (None, 3), (tags(&["bogus"]), 1)] {
            assert!(matches!(ctl.sleep(&engine, t, false, level).await, Err(SleepError::BadRequest(_))));
        }
        let (ctl, _, engine) = setup(false);
        assert!(matches!(ctl.sleep(&engine, None, false, 2).await, Err(SleepError::BadRequest(_))));
        assert_eq!(ctl.level(), 0);
    }
}
