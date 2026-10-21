//! HookRunner + HookHandler trait + Rust closure / shell script handlers.

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

#[cfg(unix)]
use rustix::process::{kill_process_group, test_kill_process_group, Pid, Signal};

use super::event::HookEvent;
use crate::abort::AbortController;
use crate::error::AgentError;

const DEFAULT_SCRIPT_TIMEOUT: Duration = Duration::from_secs(30);
const SCRIPT_KILL_GRACE: Duration = Duration::from_secs(2);
const SCRIPT_GROUP_POLL: Duration = Duration::from_millis(20);

/// Outcome of running a single hook against a single event.
///
/// Mirrors the Claude Code shell-script hook protocol:
/// - exit 0 → [`HookOutcome::Ok`] (proceed normally)
/// - exit 2 → [`HookOutcome::Block`] (abort the action that triggered the hook)
/// - other exit code → [`HookOutcome::Warn`] (log + proceed)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HookOutcome {
    Ok,
    Block,
    Warn(i32),
}

impl HookOutcome {
    pub fn from_exit_code(code: i32) -> Self {
        match code {
            0 => Self::Ok,
            2 => Self::Block,
            other => Self::Warn(other),
        }
    }
}

/// One hook implementation. Implementors come from either user Rust
/// closures or external shell scripts.
#[async_trait]
pub trait HookHandler: Send + Sync + std::fmt::Debug {
    async fn handle(&self, event: &HookEvent) -> HookOutcome;

    /// Run with the root turn's cancellation and activity state. Rust hooks
    /// that do not need either keep the legacy `handle` implementation.
    async fn handle_with_abort(&self, event: &HookEvent, _abort: &AbortController) -> HookOutcome {
        self.handle(event).await
    }
}

/// Adapter that wraps a synchronous Rust closure into a [`HookHandler`].
pub struct RustHookHandler {
    name: String,
    func: Box<dyn Fn(&HookEvent) -> HookOutcome + Send + Sync>,
}

impl RustHookHandler {
    pub fn new(
        name: impl Into<String>,
        f: impl Fn(&HookEvent) -> HookOutcome + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            func: Box::new(f),
        }
    }
}

impl std::fmt::Debug for RustHookHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RustHookHandler")
            .field("name", &self.name)
            .field("func", &"<closure>")
            .finish()
    }
}

#[async_trait]
impl HookHandler for RustHookHandler {
    async fn handle(&self, event: &HookEvent) -> HookOutcome {
        (self.func)(event)
    }
}

/// Adapter that runs an external shell script as a hook. The event is
/// serialized to JSON and written to the script's stdin; the script's
/// exit code drives the [`HookOutcome`].
#[derive(Debug, Clone)]
pub struct ScriptHookHandler {
    pub script: PathBuf,
    timeout: Duration,
}

impl ScriptHookHandler {
    pub fn new(script: impl Into<PathBuf>) -> Self {
        Self {
            script: script.into(),
            timeout: DEFAULT_SCRIPT_TIMEOUT,
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    async fn run(&self, event: &HookEvent, abort: &AbortController) -> Result<i32, AgentError> {
        use tokio::io::AsyncWriteExt;
        use tokio::process::Command;

        if abort.is_aborted() {
            return Err(AgentError::Aborted(
                abort.reason().unwrap_or_else(|| "aborted".into()),
            ));
        }

        // A user script is unclassified external code. Latch retry safety
        // before it can run, even if spawning or writing stdin later fails.
        abort.mark_side_effect_risk();
        let json = serde_json::to_vec(event)?;
        let mut command = Command::new(&self.script);
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.as_std_mut().process_group(0);
        }
        let mut child = command.spawn()?;
        let pid = child.id();
        let mut guard = ScriptProcessGuard::new(pid, abort.clone());

        enum Completion {
            Exited(std::process::ExitStatus),
            Aborted,
            TimedOut,
        }

        let completion = {
            let execution = async {
                if let Some(mut stdin) = child.stdin.take() {
                    stdin.write_all(&json).await?;
                    stdin.shutdown().await?;
                }
                child.wait().await
            };
            tokio::pin!(execution);
            tokio::select! {
                status = &mut execution => Completion::Exited(status?),
                _ = abort.cancelled() => Completion::Aborted,
                _ = tokio::time::sleep(self.timeout) => Completion::TimedOut,
            }
        };

        match completion {
            Completion::Exited(status) => {
                // A hook may leave group-bound descendants behind when its
                // direct process exits. Close the observable group first.
                guard.cleanup_after_leader_exit().await;
                Ok(status.code().unwrap_or(-1))
            }
            Completion::Aborted => {
                let proven = terminate_script_group(pid, &mut child).await;
                guard.complete_cleanup(proven);
                Err(AgentError::Aborted(
                    abort.reason().unwrap_or_else(|| "aborted".into()),
                ))
            }
            Completion::TimedOut => {
                let proven = terminate_script_group(pid, &mut child).await;
                guard.complete_cleanup(proven);
                Err(AgentError::other(format!(
                    "hook script timed out after {}ms",
                    self.timeout.as_millis()
                )))
            }
        }
    }
}

#[derive(Debug)]
struct ScriptProcessGuard {
    pid: Option<u32>,
    abort: AbortController,
    armed: bool,
}

impl ScriptProcessGuard {
    fn new(pid: Option<u32>, abort: AbortController) -> Self {
        Self {
            pid,
            abort,
            armed: true,
        }
    }

    // The `return`s are needed on the cfg branch that compiles alone;
    // without them the platform block would be a stray expression
    // statement when the sibling block is compiled away.
    #[allow(clippy::needless_return)]
    fn kill_group(&self) -> bool {
        #[cfg(unix)]
        {
            let Some(pid) = self.pid else {
                return false;
            };
            return signal_process_group(pid, Signal::KILL);
        }

        #[cfg(windows)]
        {
            let Some(pid) = self.pid else {
                return false;
            };
            return taskkill_process_tree(pid);
        }

        #[cfg(not(any(unix, windows)))]
        false
    }

    async fn cleanup_after_leader_exit(&mut self) {
        let signal_succeeded = self.kill_group();
        let proven = wait_for_script_group_exit(self.pid, signal_succeeded).await;
        self.complete_cleanup(proven);
    }

    fn complete_cleanup(&mut self, proven: bool) {
        if !proven {
            self.abort.mark_unresolved_external_work();
        }
        self.armed = false;
    }
}

impl Drop for ScriptProcessGuard {
    fn drop(&mut self) {
        if self.armed {
            let _ = self.kill_group();
            // Drop cannot await a process-group exit proof.
            self.abort.mark_unresolved_external_work();
        }
    }
}

async fn terminate_script_group(pid: Option<u32>, child: &mut tokio::process::Child) -> bool {
    #[cfg(unix)]
    if let Some(pid) = pid {
        let term_succeeded = signal_process_group(pid, Signal::TERM);
        let mut direct_child_reaped = matches!(
            tokio::time::timeout(SCRIPT_KILL_GRACE, child.wait()).await,
            Ok(Ok(_))
        );
        // The direct child exiting does not prove its descendants exited.
        let kill_succeeded = signal_process_group(pid, Signal::KILL);
        if !direct_child_reaped {
            direct_child_reaped = matches!(
                tokio::time::timeout(SCRIPT_KILL_GRACE, child.wait()).await,
                Ok(Ok(_))
            );
        }
        let termination_delivered = term_succeeded || kill_succeeded;
        let tree_exit_proven = wait_for_script_group_exit(Some(pid), termination_delivered).await;
        return direct_child_reaped && termination_delivered && tree_exit_proven;
    }

    #[cfg(windows)]
    if let Some(pid) = pid {
        let tree_exit_proven = taskkill_process_tree(pid);
        // Keep the direct-child kill as a fallback when taskkill is missing
        // or cannot inspect the process tree.
        let _ = child.start_kill();
        let direct_child_reaped = matches!(
            tokio::time::timeout(SCRIPT_KILL_GRACE, child.wait()).await,
            Ok(Ok(_))
        );
        return direct_child_reaped && tree_exit_proven;
    }

    let _ = pid;
    let _ = child.start_kill();
    let _ = tokio::time::timeout(SCRIPT_KILL_GRACE, child.wait()).await;
    false
}

/// Deliver `signal` to the whole process group rooted at `pid`.
///
/// A direct `killpg` syscall. Shelling out to `/bin/kill -s SIG -<pgid>`
/// is deliberately avoided: the negative pid-group spelling parses
/// inconsistently (or not at all) across kill implementations, so a
/// group kill that works on one CI image silently no-ops on another.
#[cfg(unix)]
fn signal_process_group(pid: u32, signal: Signal) -> bool {
    let Ok(pid) = i32::try_from(pid) else {
        return false;
    };
    let Some(pid) = Pid::from_raw(pid) else {
        return false;
    };
    kill_process_group(pid, signal).is_ok()
}

/// Whether the process group `pid` has observably exited.
///
/// `killpg(pid, 0)` still reports a group alive while it contains
/// zombies — and a container init that never reaps orphans can keep a
/// zombie around indefinitely. A zombie is dead (SIGKILL cannot be
/// blocked or ignored), so on Linux we additionally scan `/proc` and
/// treat a group whose remaining members are all zombies as gone.
#[cfg(unix)]
fn script_group_exit_observable(pid: u32) -> bool {
    let Ok(pid) = i32::try_from(pid) else {
        return false;
    };
    let Some(pid) = Pid::from_raw(pid) else {
        return false;
    };
    if test_kill_process_group(pid).is_err() {
        // ESRCH: the group no longer exists.
        return true;
    }
    #[cfg(target_os = "linux")]
    {
        !group_has_live_member(pid)
    }
    #[cfg(not(target_os = "linux"))]
    false
}

/// Linux-only: whether any non-zombie process remains in process group
/// `pid`. `/proc/<pid>/stat` records the group of every process; a `Z`
/// state means the process has exited and is only waiting to be reaped.
#[cfg(target_os = "linux")]
fn group_has_live_member(pid: Pid) -> bool {
    let target = pid.as_raw_nonzero().get();
    let Ok(entries) = std::fs::read_dir("/proc") else {
        // /proc unreadable — be conservative: assume someone is alive.
        return true;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if !name.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }
        let Ok(stat) = std::fs::read_to_string(entry.path().join("stat")) else {
            continue;
        };
        // stat layout: `pid (comm) state ppid pgrp ...`. `comm` may
        // contain spaces, so parse from after the closing paren.
        let Some((_, rest)) = stat.rsplit_once(')') else {
            continue;
        };
        let mut fields = rest.split_whitespace();
        let Some(state) = fields.next() else {
            continue;
        };
        let _ = fields.next(); // ppid
        let Some(pgrp) = fields.next() else {
            continue;
        };
        let Ok(pgrp) = pgrp.parse::<i32>() else {
            continue;
        };
        if pgrp != target {
            continue;
        }
        if state != "Z" {
            return true;
        }
    }
    false
}

/// Best-effort Windows descendant cleanup. `taskkill /T` walks and
/// terminates the tree; when the leader has already exited, taskkill
/// reports "could not be found" — nothing left to kill, which IS the
/// exit proof the guard wants.
#[cfg(windows)]
fn taskkill_process_tree(pid: u32) -> bool {
    let pid = pid.to_string();
    match std::process::Command::new("taskkill")
        .args(["/F", "/T", "/PID", &pid])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
    {
        Ok(out) if out.status.success() => true,
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr).to_ascii_lowercase();
            stderr.contains("could not be found") || stderr.contains("not found")
        }
        Err(_) => false,
    }
}

async fn wait_for_script_group_exit(pid: Option<u32>, signal_succeeded: bool) -> bool {
    #[cfg(unix)]
    {
        let _ = signal_succeeded;
        let Some(pid) = pid else {
            return false;
        };
        let deadline = tokio::time::Instant::now() + SCRIPT_KILL_GRACE;
        loop {
            if script_group_exit_observable(pid) {
                return true;
            }
            if tokio::time::Instant::now() >= deadline {
                return false;
            }
            tokio::time::sleep(SCRIPT_GROUP_POLL).await;
        }
    }

    #[cfg(windows)]
    return signal_succeeded;

    #[cfg(not(any(unix, windows)))]
    {
        let _ = (pid, signal_succeeded);
        false
    }
}

#[async_trait]
impl HookHandler for ScriptHookHandler {
    async fn handle(&self, event: &HookEvent) -> HookOutcome {
        self.handle_with_abort(event, &AbortController::new()).await
    }

    async fn handle_with_abort(&self, event: &HookEvent, abort: &AbortController) -> HookOutcome {
        match self.run(event, abort).await {
            Ok(code) => HookOutcome::from_exit_code(code),
            Err(err) => {
                tracing::warn!(
                    script = %self.script.display(),
                    event = event.name(),
                    error = %err,
                    "hook script failed to execute; treating as warn",
                );
                HookOutcome::Warn(-1)
            }
        }
    }
}

/// Registry of hooks. `run(event)` invokes every registered handler in
/// insertion order; the first `Block` outcome short-circuits and returns
/// `Block`. Non-zero non-block exits log a warning and continue.
#[derive(Debug, Default, Clone)]
pub struct HookRunner {
    handlers: Vec<Arc<dyn HookHandler>>,
}

impl HookRunner {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, handler: Arc<dyn HookHandler>) {
        self.handlers.push(handler);
    }

    pub fn with(mut self, handler: Arc<dyn HookHandler>) -> Self {
        self.handlers.push(handler);
        self
    }

    pub fn len(&self) -> usize {
        self.handlers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.handlers.is_empty()
    }

    /// Fire `event` through every registered handler in order. Returns
    /// `Block` as soon as any handler blocks; `Warn(code)` if any handler
    /// warned but none blocked; `Ok` otherwise.
    pub async fn run(&self, event: &HookEvent) -> HookOutcome {
        self.run_with_abort(event, &AbortController::new()).await
    }

    /// Fire an event using the root turn's cancellation and activity state.
    pub async fn run_with_abort(&self, event: &HookEvent, abort: &AbortController) -> HookOutcome {
        let mut last_warn: Option<i32> = None;
        for h in &self.handlers {
            // Hook handlers are arbitrary host/user code. Conservatively latch
            // retry safety before invoking any implementation, not just the
            // bundled script adapter.
            abort.mark_side_effect_risk();
            match h.handle_with_abort(event, abort).await {
                HookOutcome::Block => return HookOutcome::Block,
                HookOutcome::Warn(code) => {
                    tracing::warn!(
                        event = event.name(),
                        exit_code = code,
                        "hook returned non-zero non-block",
                    );
                    last_warn = Some(code);
                }
                HookOutcome::Ok => {}
            }
        }
        match last_warn {
            Some(code) => HookOutcome::Warn(code),
            None => HookOutcome::Ok,
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    #[cfg(unix)]
    use std::time::{Duration, Instant};

    use super::*;

    #[cfg(unix)]
    static SCRIPT_PROCESS_TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    fn rust_hook(name: &str, outcome: HookOutcome) -> Arc<RustHookHandler> {
        let outcome_clone = outcome;
        Arc::new(RustHookHandler::new(name, move |_| outcome_clone))
    }

    #[test]
    fn outcome_from_exit_code_mapping() {
        assert_eq!(HookOutcome::from_exit_code(0), HookOutcome::Ok);
        assert_eq!(HookOutcome::from_exit_code(2), HookOutcome::Block);
        assert_eq!(HookOutcome::from_exit_code(7), HookOutcome::Warn(7));
        assert_eq!(HookOutcome::from_exit_code(-1), HookOutcome::Warn(-1));
    }

    #[tokio::test]
    async fn empty_runner_returns_ok() {
        let r = HookRunner::new();
        let event = HookEvent::OnSessionStart;
        assert_eq!(r.run(&event).await, HookOutcome::Ok);
    }

    #[tokio::test]
    async fn block_short_circuits() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_clone = calls.clone();
        let counting = Arc::new(RustHookHandler::new("counter", move |_| {
            calls_clone.fetch_add(1, Ordering::SeqCst);
            HookOutcome::Ok
        }));

        let r = HookRunner::new()
            .with(rust_hook("blocker", HookOutcome::Block))
            .with(counting);

        let event = HookEvent::OnSessionStart;
        assert_eq!(r.run(&event).await, HookOutcome::Block);
        // The counting hook should NOT have run because the blocker came first.
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn warn_continues_and_returns_warn() {
        let r = HookRunner::new()
            .with(rust_hook("a", HookOutcome::Ok))
            .with(rust_hook("b", HookOutcome::Warn(7)))
            .with(rust_hook("c", HookOutcome::Ok));
        let event = HookEvent::OnSessionStart;
        assert_eq!(r.run(&event).await, HookOutcome::Warn(7));
    }

    #[tokio::test]
    async fn handler_receives_event_with_name() {
        let captured = Arc::new(std::sync::Mutex::new(String::new()));
        let captured_c = captured.clone();
        let h = Arc::new(RustHookHandler::new("capture", move |e| {
            *captured_c.lock().unwrap() = e.name().to_string();
            HookOutcome::Ok
        }));
        let r = HookRunner::new().with(h);
        let event = HookEvent::BeforeToolUse {
            tool: "Bash".into(),
            input: serde_json::json!({}),
        };
        r.run(&event).await;
        assert_eq!(*captured.lock().unwrap(), "before_tool_use");
    }

    #[tokio::test]
    async fn runner_latches_side_effect_risk_before_custom_handler() {
        let abort = AbortController::new();
        let activity = abort.activity();
        let observed = Arc::new(std::sync::Mutex::new(false));
        let observed_hook = observed.clone();
        let hook_activity = activity.clone();
        let runner =
            HookRunner::new().with(Arc::new(RustHookHandler::new("risk-observer", move |_| {
                *observed_hook.lock().unwrap() = hook_activity.side_effect_risk();
                HookOutcome::Ok
            })));

        runner
            .run_with_abort(&HookEvent::OnSessionStart, &abort)
            .await;

        assert!(*observed.lock().unwrap());
        assert!(activity.side_effect_risk());
        assert!(!activity.unresolved_external_work());
    }

    #[tokio::test]
    async fn missing_script_process_identity_marks_cleanup_unresolved() {
        let abort = AbortController::new();
        let mut guard = ScriptProcessGuard::new(None, abort.clone());

        guard.cleanup_after_leader_exit().await;

        assert!(abort.activity().unresolved_external_work());
    }

    #[test]
    fn dropped_script_guard_without_exit_proof_marks_cleanup_unresolved() {
        let abort = AbortController::new();
        drop(ScriptProcessGuard::new(None, abort.clone()));

        assert!(abort.activity().unresolved_external_work());
    }

    #[cfg(unix)]
    fn hanging_script() -> (tempfile::TempDir, PathBuf, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("hook.sh");
        let descendant_pid = dir.path().join("descendant.pid");
        std::fs::write(
            &script,
            format!(
                "#!/bin/sh\nsleep 30 &\necho $! > '{}'\nwait\n",
                descendant_pid.display()
            ),
        )
        .unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o700)).unwrap();
        (dir, script, descendant_pid)
    }

    #[cfg(unix)]
    async fn read_pid(path: &std::path::Path) -> i32 {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if let Ok(text) = std::fs::read_to_string(path) {
                    if let Ok(pid) = text.trim().parse() {
                        break pid;
                    }
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("hook did not start")
    }

    /// A process is gone when the kill-probe reports ESRCH — or, on
    /// Linux, when `/proc` shows it as a zombie (dead but not yet
    /// reaped by an init that may never reap in containers).
    #[cfg(unix)]
    fn process_observably_gone(pid: i32) -> bool {
        let Some(pid) = rustix::process::Pid::from_raw(pid) else {
            return true;
        };
        if rustix::process::test_kill_process(pid).is_err() {
            return true;
        }
        #[cfg(target_os = "linux")]
        {
            let stat = std::fs::read_to_string(format!("/proc/{}/stat", pid.as_raw_nonzero()))
                .unwrap_or_default();
            stat.rsplit_once(')')
                .and_then(|(_, rest)| rest.split_whitespace().next())
                == Some("Z")
        }
        #[cfg(not(target_os = "linux"))]
        false
    }

    #[cfg(unix)]
    async fn assert_process_gone(pid: i32) {
        tokio::time::timeout(Duration::from_secs(3), async move {
            while !process_observably_gone(pid) {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("hook descendant survived process-group cleanup");
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn script_hook_abort_latches_risk_and_kills_descendants() {
        let _serial = SCRIPT_PROCESS_TEST_LOCK.lock().await;
        let (_dir, script, pid_file) = hanging_script();
        let handler = ScriptHookHandler::new(script);
        let abort = AbortController::new();
        let task_abort = abort.clone();
        let task = tokio::spawn(async move {
            handler
                .handle_with_abort(&HookEvent::OnSessionStart, &task_abort)
                .await
        });
        let descendant = read_pid(&pid_file).await;
        assert!(abort.activity().side_effect_risk());

        abort.abort_with_reason("test cancellation");
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(5), task)
                .await
                .unwrap()
                .unwrap(),
            HookOutcome::Warn(-1)
        );
        assert_process_gone(descendant).await;
        assert!(!abort.activity().unresolved_external_work());
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn dropping_script_hook_future_kills_descendants() {
        let _serial = SCRIPT_PROCESS_TEST_LOCK.lock().await;
        let (_dir, script, pid_file) = hanging_script();
        let handler = ScriptHookHandler::new(script);
        let abort = AbortController::new();
        let task_abort = abort.clone();
        let task = tokio::spawn(async move {
            handler
                .handle_with_abort(&HookEvent::OnSessionStart, &task_abort)
                .await
        });
        let descendant = read_pid(&pid_file).await;

        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        assert_process_gone(descendant).await;
        assert!(abort.activity().unresolved_external_work());
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn script_hook_timeout_is_bounded_and_kills_descendants() {
        let _serial = SCRIPT_PROCESS_TEST_LOCK.lock().await;
        let (_dir, script, pid_file) = hanging_script();
        let handler = ScriptHookHandler::new(script).with_timeout(Duration::from_secs(1));
        let started = Instant::now();
        let abort = AbortController::new();
        let task_abort = abort.clone();
        let mut task =
            tokio::spawn(async move { handler.run(&HookEvent::OnSessionStart, &task_abort).await });
        let descendant = tokio::select! {
            pid = read_pid(&pid_file) => pid,
            early = &mut task => panic!("hook ended before starting descendant: {early:?}"),
        };
        let result = task.await.unwrap();

        assert!(result.unwrap_err().to_string().contains("timed out"));
        assert!(started.elapsed() < Duration::from_secs(5));
        assert_process_gone(descendant).await;
        assert!(!abort.activity().unresolved_external_work());
    }
}
