//! Full multi-turn query loop with phase machine (Phase 3 / Task 3.4).
//!
//! Walks the conversation through this state graph:
//!
//! ```text
//! Start → Streaming → ToolDispatch → ToolCollecting → YieldingResult → Streaming → ... → Done
//! ```
//!
//! - **Streaming** — call provider, forward TextDelta/Thinking/Usage,
//!   collect ToolUse blocks, track `stop_reason`.
//! - **ToolDispatch** — for each pending tool_use, run permission +
//!   hooks; failed checks emit a synthetic ToolResult (ok=false).
//! - **ToolCollecting** — drive [`ToolExecutor`] (batch I) so the
//!   surviving tool_uses run concurrently and yield in receipt order.
//! - **YieldingResult** — push a User message carrying every
//!   tool_result block, ready to feed back into the next streaming turn.
//! - **Done** — provider's stop_reason was a terminal value (end_turn,
//!   stop_sequence, max_tokens) and no tool_uses remain.

#![allow(clippy::result_large_err)]
// PermissionDecision and the assistant ContentBlock collection are
// intentionally large — they carry rule + reason diagnostics needed by
// the host UI. The multi-turn loop already crosses tokio task
// boundaries via mpsc, so a Box on the Ok variant doesn't materially
// change runtime cost; consistent with `permission/external_queue.rs`.

use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use futures::channel::mpsc;
use futures::{Stream, StreamExt};

use crate::abort::AbortController;
use crate::compact::{
    apply_compaction_to_store, apply_microcompact_to_store, compact_with_hooks,
    estimate_text_tokens, estimate_tokens, promote_to_store, AutoCompactReason, AutoCompactState,
    CompactError, CompactTrigger, CompactWithHooksRequest, MicrocompactConfig,
    PartialCompactDirection, SessionMemoryStore,
};
use crate::error::AgentError;
use crate::file_cache::FileStateCache;
use crate::hook::{HookEvent, HookOutcome, HookRunner};
use crate::message::{ContentBlock, Header, ImageSource, Message, MessageStore, ToolResultContent};
use crate::permission::{PermissionDecision, PermissionManager};
use crate::provider::{Provider, StreamRequest, ThinkingConfig, ToolDefinition};
use crate::stream::{Event, EventStream, RequestedToolUse, ResultData, ToolExecutor};
use crate::tool::{ToolRegistry, ToolUseContext};

/// Default declared context window when the caller has no model-specific
/// information. Matches Anthropic's `claude-3-5-sonnet` family. Callers
/// targeting smaller windows should override via the builder.
pub const DEFAULT_MODEL_MAX_TOKENS: u32 = 200_000;

/// Loop state at any point during a turn cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Start,
    Streaming,
    ToolDispatch,
    ToolCollecting,
    YieldingResult,
    Done,
}

/// Transitions the loop can fire. Phase 3 batch J ships `ToolUse` and
/// `StopHook`; the remaining variants are reserved for later phases —
/// they exist now (behind `#[non_exhaustive]`) so Phase 4+ work can
/// add behavior without a SemVer break.
///
/// **Reserved variants — not produced by the loop in Phase 3:**
/// - [`Self::CollapseDrain`] — Phase 4: context-window saturation
///   triggers sliding-window trim.
/// - [`Self::ReactiveCompact`] — Phase 4: token-budget-driven
///   transcript summarization.
/// - [`Self::MaxOutputEscalate`] — Phase 5+: provider's `max_tokens`
///   cap hit, split the assistant message.
/// - [`Self::MaxOutputMultiTurn`] — Phase 5+: continuation of
///   [`Self::MaxOutputEscalate`] across turns.
///
/// External code should pattern-match with a catch-all so the
/// reserved variants don't break when implemented.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Transition {
    /// Provider emitted tool_use blocks; dispatch them this turn.
    ToolUse(Vec<RequestedToolUse>),
    /// A hook returned [`HookOutcome::Block`]; abort the turn.
    StopHook { event: String, code: i32 },
    /// **Reserved (Phase 4)** — context window saturated; sliding-window trim.
    CollapseDrain,
    /// **Reserved (Phase 4)** — token budget hit; rewrite transcript to summary.
    ReactiveCompact,
    /// **Reserved (Phase 5+)** — provider's `max_tokens` cap reached; split.
    MaxOutputEscalate,
    /// **Reserved (Phase 5+)** — continuation of [`Self::MaxOutputEscalate`].
    MaxOutputMultiTurn,
}

/// Full multi-turn query driver. Construct with the builder, then call
/// [`run`](Self::run). Returns a stream that yields the same `Event`
/// vocabulary as a single-turn provider — TextDelta / Thinking /
/// ToolUse / ToolResult / Usage / Result — but covers as many provider
/// turns as the conversation requires before reaching `Done`.
#[derive(Debug)]
pub struct QueryLoop {
    pub provider: Arc<dyn Provider>,
    pub tools: Arc<ToolRegistry>,
    pub permissions: Arc<PermissionManager>,
    pub hooks: Arc<HookRunner>,
    /// Shared message store. **Single-driver invariant**: while
    /// [`Self::run`] is in progress, only the spawned drive task may
    /// mutate the store. External code may take a read snapshot via
    /// `store.lock()` but MUST NOT push, pop, or otherwise modify the
    /// store, because reactive auto-compaction relies on the snapshot
    /// taken inside `maybe_auto_compact` matching the live store when
    /// the compaction result is applied. Concurrent mutations would
    /// land before the boundary marker and silently escape the
    /// summary. If you need to drive multiple parallel turns over the
    /// same conversation history, construct a new `QueryLoop` per
    /// turn with the same `Arc` and serialize the `run()` calls.
    pub store: Arc<Mutex<MessageStore>>,
    pub model: String,
    pub system: Option<String>,
    pub max_output_tokens: u32,
    pub max_concurrent_tools: usize,
    /// Sub-agent nesting depth of THIS loop (0 = root). Threaded into
    /// every [`ToolUseContext`] so the Task tool's recursion guard holds
    /// across the child loop's `tokio::spawn`.
    pub task_depth: usize,
    /// Whether this loop owns the host-visible root turn boundary. Only the
    /// outermost loop waits for all shared TurnActivity workers before EOF;
    /// nested Task/team loops must set this to `false` to avoid waiting on
    /// their still-active parent tool task.
    pub root_turn_owner: bool,
    /// Optional mid-turn steering inbox: user messages injected while a
    /// multi-step turn runs. Drained at the top of each loop iteration and
    /// appended to the store before the next provider round-trip. Shared
    /// (Arc<Mutex<Receiver>>) because the loop is rebuilt per turn while the
    /// channel persists on the host engine.
    pub steer: Option<Arc<std::sync::Mutex<mpsc::UnboundedReceiver<Vec<ContentBlock>>>>>,
    /// Optional cap on assistant turns. Defaults to `usize::MAX` (UNBOUNDED) in
    /// [`Self::builder`] — the loop already stops naturally on a turn with no
    /// tool calls. Set a finite value only to force a runaway backstop.
    pub max_iterations: usize,
    /// How many times to retry a transient API failure (rate limit / 5xx /
    /// network) when opening the provider stream, with exponential backoff
    /// between attempts. Defaults to 10 in [`Self::builder`].
    pub max_api_retries: u32,
    /// Working directory threaded into every [`ToolUseContext`].
    pub cwd: PathBuf,
    /// Shared file state cache threaded into every [`ToolUseContext`].
    pub file_cache: Arc<FileStateCache>,
    /// Declared context window for the active model. Used by reactive
    /// compaction to compute thresholds. Defaults to
    /// [`DEFAULT_MODEL_MAX_TOKENS`].
    pub model_max_tokens: u32,
    /// Whether reactive auto-compaction is enabled. When `true`, every
    /// turn evaluates [`AutoCompactState`] before streaming and may
    /// rewrite the message store via [`compact_with_hooks`] +
    /// [`apply_compaction_to_store`]. Defaults to `true`.
    pub auto_compact_enabled: bool,
    /// Cross-run auto-compaction state. Latches like `no_progress` and
    /// the `consecutive_failures` circuit breaker survive across
    /// successive [`Self::run`] calls when the host wires the same
    /// `Arc` into every loop instance via
    /// [`QueryLoopBuilder::compact_state`]. Defaults to a fresh, owned
    /// state when not provided (state then resets per-run, matching
    /// pre-Q-δ behavior).
    pub compact_state: Arc<Mutex<AutoCompactState>>,
    /// Sampling temperature passed to the provider. `None` = provider default.
    pub temperature: Option<f32>,
    /// Extended-thinking / reasoning token budget passed to the provider.
    /// `None` = thinking disabled.
    pub thinking: Option<ThinkingConfig>,
    /// OpenAI-style reasoning effort ("low" | "medium" | "high") passed to
    /// providers that support it. `None` = provider default.
    pub reasoning_effort: Option<String>,
    /// Whether to request provider prompt caching (Anthropic `cache_control`).
    pub use_prompt_cache: bool,
    /// Whether the microcompact ladder step runs before LLM compaction:
    /// on threshold hit, old tool-result payloads are cleared first, and
    /// if that alone drops the total back under the threshold the LLM
    /// compaction is skipped for this turn. Defaults to `false`.
    pub microcompact_enabled: bool,
    /// Tool NAMES whose tool-result payloads microcompaction must never
    /// clear (interactive question tools return the user's actual input;
    /// clearing it makes the model re-ask forever). Empty by default.
    pub microcompact_protected_tools: Vec<String>,
    /// Tighter tool-loop-guard thresholds for weak models: nudge on the 2nd
    /// identical iteration and abort on the 4th (instead of 3rd/6th). Weak
    /// models enter the repetition attractor faster and recover from the
    /// nudge less often, so waiting the standard distance just burns calls.
    /// Defaults to `false`.
    pub strict_loop_guard: bool,
    /// Host-supplied extra instructions appended to the summarization
    /// prompt of every auto-compaction (see
    /// [`CompactWithHooksRequest::with_custom_instructions`]).
    pub compact_instructions: Option<String>,
    /// Optional durable sink: after a successful auto-compaction the
    /// analysis bullets are promoted here via
    /// [`crate::compact::promote_to_store`]. Errors surface as a Notice
    /// and never fail the compaction.
    pub session_memory: Option<Arc<dyn SessionMemoryStore>>,
}

impl QueryLoop {
    pub fn builder(provider: Arc<dyn Provider>, model: impl Into<String>) -> QueryLoopBuilder {
        QueryLoopBuilder::new(provider, model)
    }

    /// Run the loop. Returns a stream of `Event`s that completes after
    /// the conversation reaches `Phase::Done`.
    pub async fn run(
        self,
        user_msg: impl Into<String>,
        abort: AbortController,
    ) -> Result<Box<dyn EventStream>, AgentError> {
        self.run_blocks(
            vec![ContentBlock::Text {
                text: user_msg.into(),
            }],
            abort,
        )
        .await
    }

    /// Run the loop with pre-built user content blocks, such as text plus
    /// inline image attachments.
    pub async fn run_blocks(
        self,
        content: Vec<ContentBlock>,
        abort: AbortController,
    ) -> Result<Box<dyn EventStream>, AgentError> {
        // Push user message before spawning so callers see it in the
        // store synchronously.
        let user_message = Message::User {
            header: Header::new(),
            content,
        };
        {
            let mut store = self
                .store
                .lock()
                .map_err(|_| AgentError::other("query store lock poisoned"))?;
            store.push(user_message)?;
        }

        let store = self.store.clone();
        let await_quiescence = self.root_turn_owner;
        let repair_from = store.lock().map(|store| store.len()).unwrap_or(0);
        let repair_fence = Arc::new(Mutex::new(()));
        let (tx, rx) = mpsc::unbounded::<Result<Event, AgentError>>();
        let driver_abort = abort.clone();
        let driver_repair_fence = repair_fence.clone();
        let driver_work = abort.activity().track_worker();
        let driver = tokio::spawn(async move {
            let _work = driver_work;
            drive(self, tx, abort, driver_repair_fence).await;
        });
        Ok(Box::new(DriverStream {
            rx,
            driver: Some(driver),
            abort: driver_abort,
            store,
            repair_from,
            repair_fence,
            quiescence: None,
            terminal_error: None,
            await_quiescence,
            completed: false,
        }))
    }
}

/// Owns the real query driver. Dropping a host's event stream is a hard
/// cancellation boundary, not permission for a detached zombie driver to
/// keep mutating the shared store while a replacement turn starts.
struct DriverStream {
    rx: mpsc::UnboundedReceiver<Result<Event, AgentError>>,
    driver: Option<tokio::task::JoinHandle<()>>,
    abort: AbortController,
    store: Arc<Mutex<MessageStore>>,
    repair_from: usize,
    repair_fence: Arc<Mutex<()>>,
    quiescence: Option<Pin<Box<dyn Future<Output = ()> + Send>>>,
    terminal_error: Option<AgentError>,
    await_quiescence: bool,
    completed: bool,
}

impl Stream for DriverStream {
    type Item = Result<Event, AgentError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(quiescence) = self.quiescence.as_mut() {
            if quiescence.as_mut().poll(cx).is_pending() {
                return Poll::Pending;
            }
            self.quiescence.take();
            self.completed = true;
            return match self.terminal_error.take() {
                Some(error) => Poll::Ready(Some(Err(error))),
                None => Poll::Ready(None),
            };
        }
        match Pin::new(&mut self.rx).poll_next(cx) {
            Poll::Ready(Some(Err(error))) => {
                // Errors are terminal, but publishing one makes most hosts
                // drop the stream immediately. Hold it until the driver has
                // joined and root quiescence is acknowledged.
                if self.terminal_error.is_none() {
                    self.terminal_error = Some(error);
                }
                cx.waker().wake_by_ref();
                Poll::Pending
            }
            Poll::Ready(Some(Ok(_))) if self.terminal_error.is_some() => {
                // Discard any producer bug/cleanup event emitted after the
                // first terminal error while still draining toward EOF.
                cx.waker().wake_by_ref();
                Poll::Pending
            }
            Poll::Ready(Some(item)) => Poll::Ready(Some(item)),
            Poll::Pending => Poll::Pending,
            Poll::Ready(None) => {
                let Some(driver) = self.driver.as_mut() else {
                    self.completed = true;
                    return Poll::Ready(None);
                };
                match Pin::new(driver).poll(cx) {
                    Poll::Pending => Poll::Pending,
                    Poll::Ready(result) => {
                        self.driver.take();
                        if let Err(error) = result {
                            self.terminal_error = Some(AgentError::other(format!(
                                "query driver task failed: {error}"
                            )));
                        }
                        if !self.await_quiescence {
                            self.completed = true;
                            return match self.terminal_error.take() {
                                Some(error) => Poll::Ready(Some(Err(error))),
                                None => Poll::Ready(None),
                            };
                        }
                        let activity = self.abort.activity();
                        self.quiescence = Some(Box::pin(async move {
                            activity.wait_for_quiescence().await;
                        }));
                        cx.waker().wake_by_ref();
                        Poll::Pending
                    }
                }
            }
        }
    }
}

impl Drop for DriverStream {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        self.abort
            .abort_with_reason("query event stream dropped by host");
        if let Some(driver) = self.driver.as_ref() {
            driver.abort();
        }
        // Coordinate with the assistant-tool-use append, then synchronously
        // close any persisted gap before the outer worker JoinHandle can end.
        let _fence = self.repair_fence.lock().ok();
        repair_dangling_tool_uses(&self.store, self.repair_from, None);
    }
}

/// Repairs the message store if the query driver itself is hard-aborted after
/// it has persisted assistant tool-use blocks. The normal path disarms this
/// only after the matching user tool-result message is committed.
struct PendingToolRepair {
    store: Arc<Mutex<MessageStore>>,
    ids: Vec<String>,
    armed: bool,
}

impl PendingToolRepair {
    fn new(store: Arc<Mutex<MessageStore>>, ids: Vec<String>) -> Self {
        Self {
            store,
            ids,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for PendingToolRepair {
    fn drop(&mut self) {
        if !self.armed || self.ids.is_empty() {
            return;
        }
        repair_dangling_tool_uses(&self.store, 0, Some(&self.ids));
    }
}

fn repair_dangling_tool_uses(
    store: &Arc<Mutex<MessageStore>>,
    from: usize,
    candidates: Option<&[String]>,
) {
    let Ok(mut store) = store.lock() else {
        return;
    };
    let answered: std::collections::HashSet<&str> = store
        .iter()
        .filter_map(|message| match message {
            Message::User { content, .. } => Some(content),
            _ => None,
        })
        .flatten()
        .filter_map(|block| match block {
            ContentBlock::ToolResult { tool_use_id, .. } => Some(tool_use_id.as_str()),
            _ => None,
        })
        .collect();
    let requested: Vec<String> = match candidates {
        Some(ids) => ids.to_vec(),
        None => store
            .iter()
            .skip(from)
            .filter_map(|message| match message {
                Message::Assistant { content, .. } => Some(content),
                _ => None,
            })
            .flatten()
            .filter_map(|block| match block {
                ContentBlock::ToolUse { id, .. } => Some(id.clone()),
                _ => None,
            })
            .collect(),
    };
    let missing: Vec<String> = requested
        .into_iter()
        .filter(|id| !answered.contains(id.as_str()))
        .collect();
    if missing.is_empty() {
        return;
    }
    let header = store
        .last()
        .map(|message| Header::child_of(message.uuid()))
        .unwrap_or_default();
    let message = Message::User {
        header,
        content: missing
            .into_iter()
            .map(|id| ContentBlock::ToolResult {
                tool_use_id: id,
                content: tool_result_content(
                    false,
                    &serde_json::json!({ "error": "[interrupted]" }),
                    false,
                ),
                is_error: true,
            })
            .collect(),
    };
    let _ = store.push(message);
}

#[derive(Debug)]
pub struct QueryLoopBuilder {
    provider: Arc<dyn Provider>,
    model: String,
    tools: Option<Arc<ToolRegistry>>,
    permissions: Option<Arc<PermissionManager>>,
    hooks: Option<Arc<HookRunner>>,
    store: Option<Arc<Mutex<MessageStore>>>,
    system: Option<String>,
    max_output_tokens: u32,
    max_concurrent_tools: usize,
    task_depth: usize,
    root_turn_owner: bool,
    steer: Option<Arc<std::sync::Mutex<mpsc::UnboundedReceiver<Vec<ContentBlock>>>>>,
    max_iterations: usize,
    max_api_retries: u32,
    cwd: PathBuf,
    file_cache: Option<Arc<FileStateCache>>,
    model_max_tokens: u32,
    auto_compact_enabled: bool,
    compact_state: Option<Arc<Mutex<AutoCompactState>>>,
    temperature: Option<f32>,
    thinking: Option<ThinkingConfig>,
    reasoning_effort: Option<String>,
    use_prompt_cache: bool,
    microcompact_enabled: bool,
    microcompact_protected_tools: Vec<String>,
    strict_loop_guard: bool,
    compact_instructions: Option<String>,
    session_memory: Option<Arc<dyn SessionMemoryStore>>,
}

impl QueryLoopBuilder {
    pub fn new(provider: Arc<dyn Provider>, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
            tools: None,
            permissions: None,
            hooks: None,
            store: None,
            system: None,
            max_output_tokens: 4096,
            max_concurrent_tools: 8,
            task_depth: 0,
            root_turn_owner: true,
            steer: None,
            max_iterations: usize::MAX,
            max_api_retries: 10,
            cwd: PathBuf::from("."),
            file_cache: None,
            model_max_tokens: DEFAULT_MODEL_MAX_TOKENS,
            auto_compact_enabled: true,
            compact_state: None,
            temperature: None,
            thinking: None,
            reasoning_effort: None,
            use_prompt_cache: false,
            microcompact_enabled: false,
            microcompact_protected_tools: Vec::new(),
            strict_loop_guard: false,
            compact_instructions: None,
            session_memory: None,
        }
    }

    /// Sampling temperature. `None` (default) uses the provider's default;
    /// hosts set a low value (e.g. 0) for deterministic coding output.
    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }

    /// Extended-thinking / reasoning token budget. `None` (default) leaves
    /// thinking disabled; providers that don't support it ignore the field.
    pub fn thinking(mut self, cfg: ThinkingConfig) -> Self {
        self.thinking = Some(cfg);
        self
    }

    /// OpenAI-style reasoning effort ("low" | "medium" | "high"). `None`
    /// (default) uses the provider's default; providers that don't support
    /// it ignore the field.
    pub fn reasoning_effort(mut self, effort: impl Into<String>) -> Self {
        self.reasoning_effort = Some(effort.into());
        self
    }

    /// Request provider prompt caching. For Anthropic-family providers this
    /// adds `cache_control` breakpoints (system + last user message) so the
    /// stable prefix is cached across turns; other providers (OpenAI-compat)
    /// cache automatically and ignore the flag. Default off.
    pub fn use_prompt_cache(mut self, on: bool) -> Self {
        self.use_prompt_cache = on;
        self
    }

    pub fn tools(mut self, tools: Arc<ToolRegistry>) -> Self {
        self.tools = Some(tools);
        self
    }
    pub fn permissions(mut self, p: Arc<PermissionManager>) -> Self {
        self.permissions = Some(p);
        self
    }
    pub fn hooks(mut self, h: Arc<HookRunner>) -> Self {
        self.hooks = Some(h);
        self
    }
    pub fn store(mut self, s: Arc<Mutex<MessageStore>>) -> Self {
        self.store = Some(s);
        self
    }
    pub fn system(mut self, s: impl Into<String>) -> Self {
        self.system = Some(s.into());
        self
    }
    pub fn max_output_tokens(mut self, n: u32) -> Self {
        self.max_output_tokens = n;
        self
    }
    /// Sub-agent nesting depth for this loop (0 = root; Task children
    /// pass parent depth + 1). Feeds `ToolUseContext::task_depth`.
    pub fn task_depth(mut self, depth: usize) -> Self {
        self.task_depth = depth;
        self
    }

    /// Declare whether this loop owns host-visible root-turn completion.
    /// Nested loops must disable this even when they do not use Task depth.
    pub fn root_turn_owner(mut self, owns_root: bool) -> Self {
        self.root_turn_owner = owns_root;
        self
    }

    /// Attach a mid-turn steering inbox (see [`QueryLoop::steer`]).
    pub fn steer(
        mut self,
        rx: Arc<std::sync::Mutex<mpsc::UnboundedReceiver<Vec<ContentBlock>>>>,
    ) -> Self {
        self.steer = Some(rx);
        self
    }

    pub fn max_concurrent_tools(mut self, n: usize) -> Self {
        self.max_concurrent_tools = n;
        self
    }
    pub fn max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }
    /// How many times to retry a transient API failure with exponential
    /// backoff. Defaults to 10.
    pub fn max_api_retries(mut self, n: u32) -> Self {
        self.max_api_retries = n;
        self
    }
    pub fn cwd(mut self, p: impl Into<PathBuf>) -> Self {
        self.cwd = p.into();
        self
    }
    pub fn file_cache(mut self, c: Arc<FileStateCache>) -> Self {
        self.file_cache = Some(c);
        self
    }
    /// Override the declared context window. Defaults to
    /// [`DEFAULT_MODEL_MAX_TOKENS`].
    pub fn model_max_tokens(mut self, n: u32) -> Self {
        self.model_max_tokens = n;
        self
    }
    /// Enable or disable reactive auto-compaction. Defaults to `true`.
    pub fn auto_compact(mut self, on: bool) -> Self {
        self.auto_compact_enabled = on;
        self
    }
    /// Inject a shared [`AutoCompactState`] so latches and circuit
    /// breaker counters survive across successive [`QueryLoop::run`]
    /// calls. Without this, each run starts with a fresh state.
    pub fn compact_state(mut self, s: Arc<Mutex<AutoCompactState>>) -> Self {
        self.compact_state = Some(s);
        self
    }
    /// Enable the microcompact ladder step (clear old tool results before
    /// resorting to LLM compaction). Defaults to `false`.
    pub fn microcompact(mut self, on: bool) -> Self {
        self.microcompact_enabled = on;
        self
    }
    /// Tool NAMES whose tool-result payloads microcompaction must never
    /// clear (interactive question tools return the user's actual input;
    /// clearing it makes the model re-ask forever). Empty by default.
    pub fn microcompact_protected_tools(mut self, names: Vec<String>) -> Self {
        self.microcompact_protected_tools = names;
        self
    }
    /// Tighter tool-loop-guard thresholds for weak models (nudge on the 2nd
    /// identical iteration, abort on the 4th). Defaults to `false`.
    pub fn strict_loop_guard(mut self, on: bool) -> Self {
        self.strict_loop_guard = on;
        self
    }
    /// Extra instructions appended to the auto-compaction summarization
    /// prompt.
    pub fn compact_instructions(mut self, ci: impl Into<String>) -> Self {
        self.compact_instructions = Some(ci.into());
        self
    }
    /// Durable sink for compaction analysis bullets.
    pub fn session_memory(mut self, sm: Arc<dyn SessionMemoryStore>) -> Self {
        self.session_memory = Some(sm);
        self
    }

    pub fn build(self) -> QueryLoop {
        let file_cache = self.file_cache.unwrap_or_else(|| {
            Arc::new(FileStateCache::new(
                std::num::NonZeroUsize::new(64).unwrap(),
                8 * 1024 * 1024,
            ))
        });
        QueryLoop {
            provider: self.provider,
            tools: self.tools.unwrap_or_else(|| Arc::new(ToolRegistry::new())),
            permissions: self
                .permissions
                .unwrap_or_else(|| Arc::new(PermissionManager::new())),
            hooks: self.hooks.unwrap_or_else(|| Arc::new(HookRunner::new())),
            store: self
                .store
                .unwrap_or_else(|| Arc::new(Mutex::new(MessageStore::new()))),
            model: self.model,
            system: self.system,
            max_output_tokens: self.max_output_tokens,
            max_concurrent_tools: self.max_concurrent_tools,
            task_depth: self.task_depth,
            root_turn_owner: self.root_turn_owner,
            steer: self.steer,
            max_iterations: self.max_iterations,
            max_api_retries: self.max_api_retries,
            cwd: self.cwd,
            file_cache,
            model_max_tokens: self.model_max_tokens,
            auto_compact_enabled: self.auto_compact_enabled,
            compact_state: self
                .compact_state
                .unwrap_or_else(|| Arc::new(Mutex::new(AutoCompactState::new()))),
            temperature: self.temperature,
            thinking: self.thinking,
            reasoning_effort: self.reasoning_effort,
            use_prompt_cache: self.use_prompt_cache,
            microcompact_enabled: self.microcompact_enabled,
            microcompact_protected_tools: self.microcompact_protected_tools,
            strict_loop_guard: self.strict_loop_guard,
            compact_instructions: self.compact_instructions,
            session_memory: self.session_memory,
        }
    }
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

async fn drive(
    qloop: QueryLoop,
    tx: mpsc::UnboundedSender<Result<Event, AgentError>>,
    abort: AbortController,
    repair_fence: Arc<Mutex<()>>,
) {
    let mut iter = 0usize;
    let mut final_result = ResultData {
        stop_reason: None,
        model: None,
        metadata: Default::default(),
    };
    let compact_state = qloop.compact_state.clone();
    // The tool set is immutable for the loop's lifetime — serialize the
    // definitions once, not on every model round-trip (each `definitions()`
    // call rebuilds and re-sorts every tool's JSON schema).
    let tool_defs = qloop.tools.definitions();

    // Fixed per-request overhead the message-store token estimate omits: the
    // system prompt and the serialized tool schemas. Constant across iterations.
    let overhead_tokens = estimate_prompt_overhead(qloop.system.as_deref(), &tool_defs);
    // Delta calibration for the output clamp AND the reactive auto-compact
    // decision: `(actual_prompt, msg_estimate)` from the previous request.
    // While the store only grows, `msg_estimate_now + (actual_prev -
    // est_prev)` predicts the current prompt accurately — it folds in the
    // true system/tool overhead and estimator bias the provider counted
    // rather than trusting our estimate of them. A compaction rewrites the
    // store and invalidates the pairing (the per-message bias belonged to
    // messages that no longer exist), so it resets to `None` and the next
    // provider turn re-calibrates.
    let mut prompt_calibration: Option<(u32, u32)> = None;
    // Consecutive re-request nudges issued for tool calls the provider dropped
    // because the output-token cap truncated their input JSON. Bounded so a
    // model that overruns its budget every single time cannot grind forever.
    let mut truncation_retries: u32 = 0;
    // Consecutive empty responses (provider returned a bare completion
    // with no text, no thinking, no tool call). Bounded — DeepSeek
    // occasionally emits a bare `stop`; one retry almost always answers,
    // but a provider that is genuinely empty must not spin.
    let mut empty_response_retries: u32 = 0;
    let mut loop_guard = ToolLoopGuard::default();

    loop {
        if abort.is_aborted() {
            let _ = tx.unbounded_send(Err(AgentError::Aborted(
                abort.reason().unwrap_or_else(|| "aborted".into()),
            )));
            return;
        }

        if iter >= qloop.max_iterations {
            // Surface the partial run as a distinguishable Result first —
            // hosts can tell the runaway backstop from a generic failure
            // and show what the run accomplished before the cap.
            let mut data = final_result.clone();
            data.stop_reason = Some("max_iterations".into());
            let _ = tx.unbounded_send(Ok(Event::Result { data }));
            let _ = tx.unbounded_send(Err(AgentError::other(format!(
                "QueryLoop hit max_iterations ({})",
                qloop.max_iterations
            ))));
            return;
        }
        iter += 1;
        if let Ok(mut s) = compact_state.lock() {
            s.next_turn();
        }

        // ----- Mid-turn steering -----
        // Drain any user messages injected while this multi-step turn was in
        // flight and append them (as User turns) BEFORE the request snapshot,
        // so the model sees the new instruction on the very next round-trip.
        // The store ends with the prior user prompt / tool_results here, so a
        // fresh User message is coherent (providers coalesce adjacent
        // same-role turns). Announced as a Notice so the UI can show it.
        if let Some(steer) = &qloop.steer {
            if let Ok(mut rx) = steer.lock() {
                while let Ok(blocks) = rx.try_recv() {
                    let msg = Message::User {
                        header: child_header(&qloop.store),
                        content: blocks,
                    };
                    if push(&qloop.store, msg).is_ok() {
                        let _ = tx.unbounded_send(Ok(Event::Notice {
                            code: "agent.steer".into(),
                            message: "injected a mid-turn user message".into(),
                        }));
                    }
                }
            }
        }

        // ----- Reactive auto-compaction (Q-δ) -----
        // Before each streaming turn, evaluate whether the CALIBRATED
        // occupancy of the current MessageStore has crossed the auto-compact
        // threshold. The raw store estimate omits the system prompt, tool
        // schemas, and any estimator bias — exactly the numbers the output
        // clamp below folds in — so the decision uses the same calibration
        // offset; otherwise the clamp can squeeze `max_tokens` to the floor
        // (truncating tool calls mid-JSON) while compaction still believes
        // there is room. On hit, run a full hook-instrumented compaction
        // in-place and continue with the rewritten store.
        if qloop.auto_compact_enabled {
            let est_offset = match prompt_calibration {
                Some((actual_prev, est_prev)) => i64::from(actual_prev) - i64::from(est_prev),
                None => i64::from(overhead_tokens),
            };
            match maybe_auto_compact(&qloop, &compact_state, &abort, &tx, est_offset).await {
                Ok(rewrote) => {
                    // The pairing (actual, estimate) described the pre-compact
                    // store; applied to the rewritten one it would pin the
                    // prompt estimate at the old, huge value and clamp output
                    // to the minimum forever. Drop it and re-calibrate on the
                    // next provider-reported Usage.
                    if rewrote {
                        prompt_calibration = None;
                    }
                }
                Err(e) => {
                    let _ = tx.unbounded_send(Err(e));
                    return;
                }
            }
        }

        // ----- Streaming phase -----
        // The store is kept well-formed at the source (every dispatched
        // tool_use gets a tool_result before the turn can exit), so the raw
        // snapshot is safe to send — but a store damaged by an OLDER
        // compaction (a severed tool pair) carries an orphan half forever;
        // strict gateways 400 that shape, wedging the session. The
        // defensive pass downgrades such orphans to text on the request
        // copy (store untouched). Cross-turn same-role adjacency from an
        // interrupt (user tool_result followed by the next user prompt) is
        // folded by each provider's renderer.
        let messages = match snapshot(&qloop.store) {
            Ok(m) => sanitize_request_messages(m),
            Err(e) => {
                let _ = tx.unbounded_send(Err(e));
                return;
            }
        };
        // Clamp `max_tokens` so `prompt + max_tokens` cannot overflow the
        // context window. `max_output_tokens` is a static per-model ceiling
        // (e.g. 384k on a ~1M window); without this, a turn whose tool results
        // grow the store past `window - max_output` mid-flight hard-400s on the
        // very next request. Estimate the current prompt (calibrated against the
        // last request's provider-reported count) and cap the completion to the
        // room left. Recomputed every iteration because the store grows within a
        // turn.
        let msg_estimate: u32 = messages
            .iter()
            .map(estimate_tokens)
            .fold(0, u32::saturating_add);
        let prompt_estimate = match prompt_calibration {
            Some((actual_prev, est_prev)) => calibrated_prompt_estimate(
                msg_estimate,
                i64::from(actual_prev) - i64::from(est_prev),
            ),
            None => msg_estimate.saturating_add(overhead_tokens),
        };
        let effective_output = clamp_output_to_window(
            qloop.max_output_tokens,
            prompt_estimate,
            qloop.model_max_tokens,
        );
        let mut req = StreamRequest::new(qloop.model.clone(), messages)
            .with_max_output_tokens(effective_output);
        req = apply_temperature_and_thinking(req, qloop.temperature, qloop.thinking);
        if let Some(e) = qloop.reasoning_effort.clone() {
            req = req.with_reasoning_effort(e);
        }
        req = req.with_prompt_cache(qloop.use_prompt_cache);
        if let Some(s) = &qloop.system {
            req = req.with_system(s.clone());
        }
        if !tool_defs.is_empty() {
            req = req.with_tools(tool_defs.clone());
        }

        // Open the provider stream, retrying transient API failures (rate
        // limits, 5xx, transport drops) with exponential backoff. Each attempt
        // waits longer than the last, and every retry is announced as a Notice
        // so the UI can show it. Permanent errors (auth / bad request) and an
        // abort break out immediately.
        let upstream = {
            let mut attempt: u32 = 0;
            loop {
                match qloop.provider.stream(req.clone(), abort.clone()).await {
                    Ok(s) => break s,
                    Err(e) => {
                        if abort.is_aborted() {
                            return;
                        }
                        if attempt >= qloop.max_api_retries || !is_retryable_api_error(&e) {
                            let _ = tx.unbounded_send(Err(e));
                            return;
                        }
                        attempt += 1;
                        let delay = retry_delay(attempt, &e);
                        let _ = tx.unbounded_send(Ok(Event::Notice {
                            code: "api_retry".into(),
                            message: format!(
                                "API error (attempt {attempt}/{}), retrying in {}s — {e}",
                                qloop.max_api_retries,
                                delay.as_secs(),
                            ),
                        }));
                        // Back off, but wake immediately if the user aborts.
                        tokio::select! {
                            _ = tokio::time::sleep(delay) => {}
                            _ = abort.token().cancelled() => return,
                        }
                    }
                }
            }
        };

        let TurnSummary {
            assistant_blocks,
            pending_tool_uses,
            stop_reason,
            model,
            prompt_tokens,
            truncated_tool_use,
        } = match consume_turn(upstream, &tx, &abort).await {
            Ok(s) => s,
            Err(()) => return, // already-emitted error or aborted
        };
        // Calibrate the next iteration's clamp on the provider's actual prompt
        // count paired with our estimate of the messages we just sent.
        if let Some(actual) = prompt_tokens {
            prompt_calibration = Some((actual, msg_estimate));
        }
        if let Some(m) = model {
            final_result.model = Some(m);
        }
        if let Some(r) = stop_reason.clone() {
            final_result.stop_reason = Some(r);
        }

        let pending_ids: Vec<String> = pending_tool_uses
            .iter()
            .map(|tool| tool.id.clone())
            .collect();

        // Push the assistant tool-use blocks and arm their repair under the
        // same fence used by DriverStream::drop. Once host shutdown holds this
        // fence, the driver cannot create a new dangling tool-use afterward.
        // An EMPTY assistant message is never pushed (`content: []` fails
        // provider validation on every subsequent request) — it can occur when
        // the only content block was a tool call truncated by the token cap.
        let assistant_is_empty = assistant_blocks.is_empty();
        let assistant = Message::Assistant {
            header: child_header(&qloop.store),
            content: assistant_blocks,
        };
        let repair_guard = {
            let Ok(_fence) = repair_fence.lock() else {
                let _ =
                    tx.unbounded_send(Err(AgentError::other("query repair fence lock poisoned")));
                return;
            };
            if abort.is_aborted() {
                return;
            }
            if !assistant_is_empty {
                if let Err(e) = push(&qloop.store, assistant) {
                    let _ = tx.unbounded_send(Err(e));
                    return;
                }
            }
            (!pending_ids.is_empty())
                .then(|| PendingToolRepair::new(qloop.store.clone(), pending_ids))
        };

        // No tool_uses → we're done.
        let Some(mut repair_guard) = repair_guard else {
            // ... unless the provider dropped a tool call whose input JSON the
            // output-token cap cut mid-write: the model MEANT to act, so
            // ending the turn here would strand the user on a partial answer.
            // Nudge the model to re-issue the call in a fresh provider turn —
            // its reasoning is already in context, so the retry is cheap and
            // almost always completes. Bounded by MAX_TRUNCATION_RETRIES on
            // top of the overall `max_iterations` budget.
            if let Some(name) = truncated_tool_use.as_deref() {
                if stop_reason.as_deref() == Some("max_tokens")
                    && truncation_retries < MAX_TRUNCATION_RETRIES
                {
                    truncation_retries += 1;
                    let nudge = Message::User {
                        header: child_header(&qloop.store),
                        content: vec![ContentBlock::Text {
                            text: format!(
                                "<system-reminder>Your `{name}` tool call was cut off by the \
                                 output-token limit and discarded — the tool did NOT run. \
                                 Re-issue the call now, going straight to the tool call with \
                                 no preamble. If its input was large, split the work into \
                                 smaller calls.</system-reminder>"
                            ),
                        }],
                    };
                    if let Err(e) = push(&qloop.store, nudge) {
                        let _ = tx.unbounded_send(Err(e));
                        return;
                    }
                    continue;
                }
            }
            // Empty response: no text, no thinking, no tool call — the
            // provider completed with nothing. DeepSeek occasionally
            // emits a bare `stop`; a second sample almost always answers.
            // Retry once (bounded) before ending the turn silently.
            if assistant_is_empty && empty_response_retries < MAX_EMPTY_RESPONSE_RETRIES {
                empty_response_retries += 1;
                let _ = tx.unbounded_send(Ok(Event::Notice {
                    code: "agent.empty_response".into(),
                    message: format!(
                        "model returned an empty response (attempt {}) — retrying once",
                        empty_response_retries
                    ),
                }));
                continue;
            }
            break;
        };

        // Loop-guard signature, part 1: the requested calls (name + input;
        // ids are excluded — they differ every iteration by construction).
        // Finished with the results below, once they exist.
        let mut iteration_hasher = std::collections::hash_map::DefaultHasher::new();
        let iteration_tools: Vec<String> =
            pending_tool_uses.iter().map(|tu| tu.name.clone()).collect();
        {
            use std::hash::Hash;
            for tu in &pending_tool_uses {
                tu.name.hash(&mut iteration_hasher);
                tu.input.to_string().hash(&mut iteration_hasher);
            }
        }

        // ----- ToolDispatch phase: permission + BeforeToolUse hooks -----
        let mut surviving = Vec::with_capacity(pending_tool_uses.len());
        let mut tool_results: Vec<(String, bool, serde_json::Value)> = Vec::new();
        // Set on a consumer-gone / executor-error / abort path, so we repair
        // the store (push results) and then stop instead of re-looping. Lives
        // here (before the permission loop) so a denial whose result fails to
        // forward to a dropped consumer also trips it.
        let mut terminate = false;
        for tu in pending_tool_uses {
            // 1. Permission.
            let decision = qloop.permissions.evaluate(&tu.name, &tu.input, None);
            match &decision {
                PermissionDecision::Allow(_) => {}
                PermissionDecision::Ask(ask) => {
                    let _ = qloop
                        .hooks
                        .run_with_abort(
                            &HookEvent::OnPermissionRequest {
                                tool: tu.name.clone(),
                                input: tu.input.clone(),
                            },
                            &abort,
                        )
                        .await;
                    let synthetic = synthetic_tool_result(
                        &tu,
                        false,
                        format!(
                            "Tool '{}' requires manual approval and no external queue is wired (ask: {}).",
                            tu.name, ask.message_text
                        ),
                    );
                    if !forward_tool_result(&tx, synthetic.clone()) {
                        terminate = true;
                    }
                    tool_results.push((tu.id.clone(), synthetic.0.ok, synthetic.0.output));
                    let _ = qloop
                        .hooks
                        .run_with_abort(
                            &HookEvent::OnPermissionDenied {
                                tool: tu.name.clone(),
                                reason: ask.message_text.clone(),
                            },
                            &abort,
                        )
                        .await;
                    continue;
                }
                PermissionDecision::Deny(deny) => {
                    let synthetic = synthetic_tool_result(
                        &tu,
                        false,
                        format!("Tool '{}' denied: {}", tu.name, deny.message_text),
                    );
                    if !forward_tool_result(&tx, synthetic.clone()) {
                        terminate = true;
                    }
                    tool_results.push((tu.id.clone(), synthetic.0.ok, synthetic.0.output));
                    let _ = qloop
                        .hooks
                        .run_with_abort(
                            &HookEvent::OnPermissionDenied {
                                tool: tu.name.clone(),
                                reason: deny.message_text.clone(),
                            },
                            &abort,
                        )
                        .await;
                    continue;
                }
            }

            // 2. BeforeToolUse hook.
            let hook_outcome = qloop
                .hooks
                .run_with_abort(
                    &HookEvent::BeforeToolUse {
                        tool: tu.name.clone(),
                        input: tu.input.clone(),
                    },
                    &abort,
                )
                .await;
            if matches!(hook_outcome, HookOutcome::Block) {
                let synthetic = synthetic_tool_result(
                    &tu,
                    false,
                    format!("Tool '{}' blocked by BeforeToolUse hook", tu.name),
                );
                if !forward_tool_result(&tx, synthetic.clone()) {
                    terminate = true;
                }
                tool_results.push((tu.id.clone(), synthetic.0.ok, synthetic.0.output));
                continue;
            }

            let _ = qloop
                .hooks
                .run_with_abort(
                    &HookEvent::OnPermissionAllowed {
                        tool: tu.name.clone(),
                    },
                    &abort,
                )
                .await;
            surviving.push(tu);
        }

        // Ids dispatched to the executor, captured before it consumes them.
        // The abort path emits a single Err(Aborted) and drops the buffered
        // stream, so any survivor that never yields a result is reconciled
        // below into a synthetic `[interrupted]` result.
        let dispatched_ids: Vec<String> = surviving.iter().map(|tu| tu.id.clone()).collect();

        // Command heads for the repeat guard, taken from SURVIVORS only —
        // a call denied by permission or blocked by a hook never ran, and
        // counting it would nudge the model for work that didn't happen.
        // An explicit `cwd` keys separately: the same command in another
        // directory is different work.
        let iteration_bash_heads: Vec<String> = surviving
            .iter()
            .filter(|tu| matches!(tu.name.as_str(), "Bash" | "Shell" | "BashRun"))
            .filter_map(|tu| {
                let cmd = tu.input.get("command").and_then(|v| v.as_str())?;
                let head = bash_command_head(cmd);
                if head.is_empty() {
                    return None;
                }
                Some(match tu.input.get("cwd").and_then(|v| v.as_str()) {
                    Some(cwd) => format!("{cwd} :: {head}"),
                    None => head,
                })
            })
            .collect();
        // A mutating fs tool in this batch means the workspace changed:
        // re-running a build or test after an edit is fresh work, so all
        // repeat counts start over.
        if iteration_tools
            .iter()
            .any(|t| HEAD_RESET_TOOLS.contains(&t.as_str()))
        {
            loop_guard.reset_bash_heads();
        }

        // ----- ToolCollecting phase: dispatch survivors via executor -----
        // Skipped when the consumer already vanished mid-permission-phase
        // (`terminate`); the reconcile below then synthesizes results for every
        // survivor so the store is still repaired before we return.
        if !surviving.is_empty() && !terminate {
            let ctx = ToolUseContext {
                cwd: qloop.cwd.clone(),
                abort: abort.clone(),
                file_cache: qloop.file_cache.clone(),
                permissions: qloop.permissions.clone(),
                hooks: qloop.hooks.clone(),
                task_depth: qloop.task_depth,
            };
            let mut exec_stream = ToolExecutor::dispatch(
                surviving.clone(),
                qloop.tools.clone(),
                ctx,
                qloop.max_concurrent_tools,
            );
            while let Some(item) = exec_stream.next().await {
                match item {
                    Ok(event @ Event::ToolResult { .. }) => {
                        // Capture for the user-message block we'll push next.
                        if let Event::ToolResult { id, ok, output } = &event {
                            tool_results.push((id.clone(), *ok, output.clone()));
                        }
                        // Fire AfterToolUse / PostToolUseFailure hook.
                        if let Some(matching) = surviving.iter().find(|s| match &event {
                            Event::ToolResult { id, .. } => &s.id == id,
                            _ => false,
                        }) {
                            if let Event::ToolResult { ok, output, .. } = &event {
                                if *ok {
                                    let _ = qloop
                                        .hooks
                                        .run_with_abort(
                                            &HookEvent::AfterToolUse {
                                                tool: matching.name.clone(),
                                                input: matching.input.clone(),
                                                output: output.clone(),
                                                ok: *ok,
                                            },
                                            &abort,
                                        )
                                        .await;
                                } else {
                                    let _ = qloop
                                        .hooks
                                        .run_with_abort(
                                            &HookEvent::PostToolUseFailure {
                                                tool: matching.name.clone(),
                                                input: matching.input.clone(),
                                                error: output
                                                    .get("error")
                                                    .and_then(|v| v.as_str())
                                                    .unwrap_or("unknown")
                                                    .into(),
                                            },
                                            &abort,
                                        )
                                        .await;
                                }
                            }
                        }
                        if tx.unbounded_send(Ok(event)).is_err() {
                            // Consumer gone: stop streaming, but fall through to
                            // repair the store (push tool_results) so no
                            // dangling tool_use is left behind.
                            terminate = true;
                            break;
                        }
                    }
                    Ok(other) => {
                        if tx.unbounded_send(Ok(other)).is_err() {
                            terminate = true;
                            break;
                        }
                    }
                    Err(e) => {
                        // Executor error (incl. mid-dispatch abort, which emits
                        // one Aborted and drops remaining survivors). Forward
                        // best-effort, then break to repair the store.
                        let _ = tx.unbounded_send(Err(e));
                        terminate = true;
                        break;
                    }
                }
            }
        }

        // ----- Reconcile: every dispatched tool_use MUST get a result -----
        // The assistant message already pushed carries N tool_use blocks; the
        // permission loop + executor filled `tool_results`. Anything still
        // missing (mid-dispatch abort dropped survivors, executor error, or a
        // consumer-gone break above) gets a synthetic `[interrupted]` result,
        // so the store never holds a tool_use without a matching tool_result.
        let answered: std::collections::HashSet<&str> =
            tool_results.iter().map(|(id, _, _)| id.as_str()).collect();
        let missing: Vec<String> = dispatched_ids
            .iter()
            .filter(|id| !answered.contains(id.as_str()))
            .cloned()
            .collect();
        for id in missing {
            tool_results.push((id, false, serde_json::json!({ "error": "[interrupted]" })));
        }

        // Loop-guard signature, part 2: the results. Sorted before hashing so
        // completion-order jitter between concurrent tools of two otherwise
        // identical iterations cannot mask a repeat.
        let repeat_streak = {
            use std::hash::{Hash, Hasher as _};
            let mut result_keys: Vec<String> = tool_results
                .iter()
                .map(|(_, ok, output)| format!("{ok}:{output}"))
                .collect();
            result_keys.sort_unstable();
            result_keys.hash(&mut iteration_hasher);
            loop_guard.observe(iteration_hasher.finish())
        };
        // Nudge payloads are built BEFORE the moves below (iteration_tools
        // and tool_results are consumed by observe_tools / the result
        // feedback message): name the repeated tools and preview the
        // identical results so a weak model sees WHY its calls are going
        // nowhere instead of re-reading a generic warning.
        let loop_tools_summary = iteration_tools.join(", ");
        let loop_results_preview: Vec<String> = tool_results
            .iter()
            .take(2)
            .map(|(_, ok, output)| {
                let text = output.to_string();
                let clipped: String = text.chars().take(160).collect();
                format!(
                    "{ok}: {clipped}{}",
                    if text.chars().count() > 160 {
                        "…"
                    } else {
                        ""
                    }
                )
            })
            .collect();
        let tool_streak = loop_guard.observe_tools(iteration_tools);

        // ----- YieldingResult phase: feed tool_results back to provider -----
        let allow_images = qloop.provider.capabilities().tool_result_images;
        let next_user = Message::User {
            header: child_header(&qloop.store),
            content: tool_results
                .into_iter()
                .map(|(id, ok, output)| ContentBlock::ToolResult {
                    tool_use_id: id,
                    content: tool_result_content(ok, &output, allow_images),
                    is_error: !ok,
                })
                .collect(),
        };
        {
            let Ok(_fence) = repair_fence.lock() else {
                let _ =
                    tx.unbounded_send(Err(AgentError::other("query repair fence lock poisoned")));
                return;
            };
            // DriverStream::drop sets abort before taking this same fence. If
            // hard-drop won the race, its synthetic result is canonical and
            // this driver must not append a duplicate real result afterward.
            if abort.is_aborted() {
                return;
            }
            if let Err(e) = push(&qloop.store, next_user) {
                let _ = tx.unbounded_send(Err(e));
                return;
            }
            repair_guard.disarm();
        }

        // A consumer-gone / executor-error / abort break repaired the store
        // above; stop here instead of streaming another turn into the void.
        if terminate {
            return;
        }

        // ----- Loop guard -----
        // The store is well-formed here (every tool_use has its result), so
        // both actions below are safe. Abort outranks the warn: by the time
        // the hard streak is reached the model has already ignored the nudge.
        let (warn_streak, abort_streak) = if qloop.strict_loop_guard {
            (
                LOOP_GUARD_WARN_STREAK_STRICT,
                LOOP_GUARD_ABORT_STREAK_STRICT,
            )
        } else {
            (LOOP_GUARD_WARN_STREAK, LOOP_GUARD_ABORT_STREAK)
        };
        if repeat_streak >= abort_streak {
            let mut data = final_result.clone();
            data.stop_reason = Some("tool_loop".into());
            let _ = tx.unbounded_send(Ok(Event::Result { data }));
            let _ = tx.unbounded_send(Err(AgentError::other(format!(
                "tool-call loop detected: the model repeated the identical tool call(s) with \
                 identical results {} times in a row — stopping the turn",
                repeat_streak + 1
            ))));
            return;
        }
        if repeat_streak == warn_streak {
            let _ = tx.unbounded_send(Ok(Event::Notice {
                code: "agent.loop.repeat".into(),
                message: format!(
                    "identical tool call repeated {} times — nudging the model to change approach",
                    repeat_streak + 1
                ),
            }));
            // Concrete nudge: name the repeated tools and preview the
            // identical results, so a weak model can see WHY its calls
            // are going nowhere instead of re-reading a generic warning.
            let nudge = Message::User {
                header: child_header(&qloop.store),
                content: vec![ContentBlock::Text {
                    text: format!(
                        "<system-reminder>Loop guard: your last {} tool calls were IDENTICAL \
                         and returned IDENTICAL results.\n- tools: {}\n- results: {}\nRepeating \
                         the same call again will not yield new information. Change your \
                         approach — use a different tool or different arguments — or stop and \
                         report what you found (or what is blocking you) to the user.</system-reminder>",
                        repeat_streak + 1,
                        loop_tools_summary,
                        loop_results_preview.join("\n- "),
                    ),
                }],
            };
            if let Err(e) = push(&qloop.store, nudge) {
                let _ = tx.unbounded_send(Err(e));
                return;
            }
        }
        // Same-tool streak (args vary — the identical-call guard above is
        // blind to it): the "guess another keyword" spiral.
        let (tool_warn, tool_abort) = if qloop.strict_loop_guard {
            (TOOL_STREAK_WARN_STRICT, TOOL_STREAK_ABORT_STRICT)
        } else {
            (TOOL_STREAK_WARN, TOOL_STREAK_ABORT)
        };
        if tool_streak >= tool_abort {
            let mut data = final_result.clone();
            data.stop_reason = Some("tool_loop".into());
            let _ = tx.unbounded_send(Ok(Event::Result { data }));
            let _ = tx.unbounded_send(Err(AgentError::other(format!(
                "tool-call loop detected: the same tool was called {} iterations in a row \
                 with varying arguments and no other progress — stopping the turn",
                tool_streak + 1
            ))));
            return;
        }
        if tool_streak == tool_warn {
            let _ = tx.unbounded_send(Ok(Event::Notice {
                code: "agent.loop.repeat".into(),
                message: format!(
                    "the same tool has been called {} times in a row — nudging the model \
                     to change approach",
                    tool_streak + 1
                ),
            }));
            // Name the stuck tool so the nudge is actionable.
            let nudge = Message::User {
                header: child_header(&qloop.store),
                content: vec![ContentBlock::Text {
                    text: format!(
                        "<system-reminder>Loop guard: you have called the SAME tool \
                         {count} times in a row with varying arguments (tools: {tools}). If \
                         the answers so far have not produced what you need, more variations \
                         will not either. Step back and change approach — use a different \
                         tool, or if the capability you are hunting for does not seem to \
                         exist, stop and say so to the user.</system-reminder>",
                        count = tool_streak + 1,
                        tools = loop_tools_summary,
                    ),
                }],
            };
            if let Err(e) = push(&qloop.store, nudge) {
                let _ = tx.unbounded_send(Err(e));
                return;
            }
        }
        // Same-command re-runs (the same-tool guard deliberately exempts
        // Bash, and the identical-call guard is blind to a changed `| grep`
        // tail): observed 49 re-runs of one vitest invocation in a single
        // turn, each fetching a different slice of output the model had
        // already produced. Nudge on a doubling cadence; never abort.
        for head in iteration_bash_heads {
            let Some(count) = loop_guard.observe_bash_head(head.clone()) else {
                continue;
            };
            let _ = tx.unbounded_send(Ok(Event::Notice {
                code: "agent.loop.rerun".into(),
                message: format!("`{head}` has run {count} times this turn — nudging the model"),
            }));
            let nudge = Message::User {
                header: child_header(&qloop.store),
                content: vec![ContentBlock::Text {
                    text: format!(
                        "<system-reminder>Command-repeat guard: you have run `{head}` {count} \
                         times this turn. Its earlier output is still in the conversation \
                         above — re-running it to see a different slice repeats work you \
                         already did. If you need the full output, run it ONCE redirected to \
                         a file (e.g. `cmd > /tmp/out.txt 2>&1`) and read that file in \
                         slices; re-run a build or test only after changing something it \
                         depends on.</system-reminder>"
                    ),
                }],
            };
            if let Err(e) = push(&qloop.store, nudge) {
                let _ = tx.unbounded_send(Err(e));
                return;
            }
        }

        // After tool dispatch, the provider needs another turn to
        // observe the tool_results. So we re-loop on `tool_use`,
        // `max_tokens`, `stop_sequence`, or any unrecognized stop
        // reason — only `end_turn` is treated as terminal here.
        // Phase 5+ may add explicit handling for `max_tokens`
        // (MaxOutputEscalate transition) and `stop_sequence` (no
        // continuation expected); for now the conservative choice is
        // to give the provider another shot. Termination is still
        // bounded by `max_iterations`.
        if let Some(reason) = &stop_reason {
            if reason == "end_turn" {
                break;
            }
        }
    }

    // ----- Done phase -----
    let _ = tx.unbounded_send(Ok(Event::Result { data: final_result }));
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Reserve held back from the context window when clamping a request's
/// completion size, covering residual estimator bias. Scales with the window —
/// a 1M window's prompt estimate can be off by more in absolute terms than a
/// 200k one — with a floor so small windows stay safe.
const OUTPUT_CLAMP_RESERVE_DIVISOR: u32 = 32;
const OUTPUT_CLAMP_RESERVE_FLOOR: u32 = 8_192;

/// Never request fewer than this many output tokens: providers reject
/// `max_tokens=0`, and a handful is useless. If the prompt is so large that even
/// this floor overflows the window the request may still 400 — but reactive
/// auto-compaction runs first each iteration, so the store should already have
/// been trimmed before we get here.
const MIN_CLAMPED_OUTPUT: u32 = 512;

/// Maximum consecutive model re-request nudges after the provider dropped a
/// tool call truncated by the output-token cap. Two attempts forgive a
/// borderline budget; a model that overruns every time stops with the normal
/// `max_tokens` result instead of grinding.
const MAX_TRUNCATION_RETRIES: u32 = 2;

/// Empty responses (bare `stop` with no text/thinking/tool call) are
/// retried once. DeepSeek occasionally emits one; a second sample almost
/// always answers. One retry — a genuinely empty provider must not spin.
const MAX_EMPTY_RESPONSE_RETRIES: u32 = 1;

/// Estimate the fixed per-request overhead the message-store token estimate
/// omits: the system prompt plus the serialized tool schemas. Byte-rate (÷4)
/// matches [`estimate_text_tokens`]'s ASCII path — good enough for a reserve.
fn estimate_prompt_overhead(system: Option<&str>, tools: &[ToolDefinition]) -> u32 {
    let sys = system.map(estimate_text_tokens).unwrap_or(0);
    let tool_bytes = serde_json::to_string(tools).map(|s| s.len()).unwrap_or(0);
    sys.saturating_add((tool_bytes as u32) / 4)
}

/// Apply the delta-calibration `offset` to a store-only token estimate.
///
/// `offset` is the provider-reported actual prompt minus our estimate taken at
/// the same instant (so it captures the system prompt, tool schemas, and any
/// per-message estimator bias), or the estimated fixed overhead when no
/// provider count has been seen yet. Signed: an estimator that OVER-counts
/// (e.g. base64 images at bytes/4) yields a negative offset, which must pull
/// the estimate down, not saturate away.
fn calibrated_prompt_estimate(msg_estimate: u32, offset: i64) -> u32 {
    u32::try_from((i64::from(msg_estimate) + offset).max(0)).unwrap_or(u32::MAX)
}

/// Shrink a request's completion budget so `prompt + max_tokens` fits inside the
/// context `window`. `configured` is the static per-model output ceiling;
/// `prompt` is the (calibrated) current occupancy. Returns the room left after a
/// scaled reserve, never above `configured` nor below [`MIN_CLAMPED_OUTPUT`]. A
/// degenerate window (0/1) is left alone. Over-clamping only shortens a
/// completion (the loop continues); under-clamping hard-400s.
fn clamp_output_to_window(configured: u32, prompt: u32, window: u32) -> u32 {
    if window <= 1 {
        return configured;
    }
    let reserve = (window / OUTPUT_CLAMP_RESERVE_DIVISOR).max(OUTPUT_CLAMP_RESERVE_FLOOR);
    let headroom = window.saturating_sub(prompt).saturating_sub(reserve);
    configured.min(headroom).max(MIN_CLAMPED_OUTPUT)
}

/// Applies the caller's `temperature` and `thinking` settings to `req`.
///
/// Anthropic rejects any request that enables extended thinking with a
/// `temperature` other than the provider default (1.0) — thinking and a
/// pinned sampling temperature are mutually exclusive on that API, and this
/// holds for both the manual (`budget_tokens`) shape AND adaptive thinking:
/// adaptive-only models also reject a non-default temperature. Thinking is
/// attached to the request unconditionally whenever the caller configured
/// it — whether the manual `budget_tokens` value actually fits the
/// per-request output budget is a provider-shape concern (Anthropic requires
/// `budget_tokens < max_tokens`; adaptive-mode models never send
/// `budget_tokens` at all), not something the loop can decide without
/// knowing which wire shape the provider will emit. See
/// `provider::anthropic::build_request_body` for where that budget-fit rule
/// now lives. When thinking is attached, an explicit temperature is dropped
/// rather than sent, since an explicit effort/thinking setting is the
/// stronger intent signal; a `tracing::warn!` is emitted only when a
/// temperature was configured and actually got dropped, so silent overrides
/// remain observable. When thinking is absent, `temperature` is applied
/// exactly as before.
fn apply_temperature_and_thinking(
    mut req: StreamRequest,
    temperature: Option<f32>,
    thinking: Option<ThinkingConfig>,
) -> StreamRequest {
    let thinking_applies = thinking.is_some();

    if thinking_applies {
        if let Some(t) = temperature {
            tracing::warn!(
                temperature = t,
                "dropping caller-set temperature: extended thinking is active and \
                 Anthropic requires provider-default temperature when thinking is enabled"
            );
        }
        req = req.with_thinking(thinking.expect("thinking_applies implies Some"));
    } else if let Some(t) = temperature {
        req = req.with_temperature(t);
    }

    req
}

#[derive(Debug, Default)]
struct TurnSummary {
    assistant_blocks: Vec<ContentBlock>,
    pending_tool_uses: Vec<RequestedToolUse>,
    stop_reason: Option<String>,
    model: Option<String>,
    /// Full prompt occupancy the provider reported for this request (non-cached
    /// input + both cache tiers), if a Usage event arrived. Feeds the next
    /// iteration's output-clamp calibration.
    prompt_tokens: Option<u32>,
    /// Name of a tool call the provider dropped because the output-token cap
    /// cut its input JSON mid-write (`Result.metadata["truncated_tool_use"]`).
    /// The driver nudges the model to re-issue the call instead of ending the
    /// turn as if it had nothing left to do.
    truncated_tool_use: Option<String>,
}

async fn consume_turn(
    mut upstream: Box<dyn EventStream>,
    tx: &mpsc::UnboundedSender<Result<Event, AgentError>>,
    abort: &AbortController,
) -> Result<TurnSummary, ()> {
    let mut summary = TurnSummary::default();
    let mut accumulated_text = String::new();
    let mut accumulated_thinking: Option<String> = None;

    while let Some(item) = upstream.next().await {
        // Pulse before forwarding so a slow host/UI consumer cannot make a
        // healthy provider (including a nested Task loop) look inactive.
        abort.pulse();
        if abort.is_aborted() {
            let _ = tx.unbounded_send(Err(AgentError::Aborted(
                abort.reason().unwrap_or_else(|| "aborted".into()),
            )));
            return Err(());
        }
        match item {
            Ok(event) => match event {
                Event::TextDelta { delta } => {
                    accumulated_text.push_str(&delta);
                    if tx.unbounded_send(Ok(Event::TextDelta { delta })).is_err() {
                        return Err(());
                    }
                }
                Event::Thinking { delta } => {
                    accumulated_thinking
                        .get_or_insert_with(String::new)
                        .push_str(&delta);
                    if tx.unbounded_send(Ok(Event::Thinking { delta })).is_err() {
                        return Err(());
                    }
                }
                Event::ToolUse { id, name, input } => {
                    summary.pending_tool_uses.push(RequestedToolUse {
                        id: id.clone(),
                        name: name.clone(),
                        input: input.clone(),
                    });
                    if tx
                        .unbounded_send(Ok(Event::ToolUse { id, name, input }))
                        .is_err()
                    {
                        return Err(());
                    }
                }
                Event::Result { data } => {
                    summary.stop_reason = data.stop_reason.clone();
                    summary.model = data.model.clone();
                    summary.truncated_tool_use = data
                        .metadata
                        .get("truncated_tool_use")
                        .and_then(|t| t.get("name"))
                        .and_then(|n| n.as_str())
                        .map(String::from);
                    // Don't forward the per-turn Result yet — the driver
                    // emits a single final Result after the whole loop
                    // settles.
                }
                Event::Usage {
                    input_tokens,
                    cache_read,
                    cache_create,
                    ..
                } => {
                    // Record the provider's own prompt count (fields are Copy, so
                    // `event` stays intact to forward). The next iteration uses it
                    // to size `max_tokens` so `prompt + max_tokens` stays under the
                    // context window.
                    summary.prompt_tokens = Some(
                        input_tokens
                            .saturating_add(cache_read)
                            .saturating_add(cache_create),
                    );
                    if tx.unbounded_send(Ok(event)).is_err() {
                        return Err(());
                    }
                }
                Event::Error { .. }
                | Event::Notice { .. }
                | Event::ToolResult { .. }
                | Event::Unknown => {
                    if tx.unbounded_send(Ok(event)).is_err() {
                        return Err(());
                    }
                }
            },
            Err(e) => {
                let _ = tx.unbounded_send(Err(e));
                return Err(());
            }
        }
    }

    // Order blocks the way the model produced them: reasoning first, then the
    // answer text, then tool calls. Storing thinking BEFORE text keeps a resumed
    // session's transcript consistent with the live view (reasoning above the
    // answer) instead of surfacing the answer before its reasoning.
    if let Some(thinking) = accumulated_thinking {
        summary.assistant_blocks.push(ContentBlock::Thinking {
            thinking,
            signature: None,
        });
    }
    if !accumulated_text.is_empty() {
        summary.assistant_blocks.push(ContentBlock::Text {
            text: accumulated_text,
        });
    }
    for tu in &summary.pending_tool_uses {
        summary.assistant_blocks.push(ContentBlock::ToolUse {
            id: tu.id.clone(),
            name: tu.name.clone(),
            input: tu.input.clone(),
        });
    }
    Ok(summary)
}

fn snapshot(store: &Arc<Mutex<MessageStore>>) -> Result<Vec<Message>, AgentError> {
    let s = store
        .lock()
        .map_err(|_| AgentError::other("query store lock poisoned"))?;
    Ok(s.iter().cloned().collect())
}

/// Defensive tool-pairing repair for one request snapshot.
///
/// The QueryLoop keeps the store well-formed (every dispatched tool_use is
/// answered before the next request), and `pair_safe_split` keeps compaction
/// from severing a pair. But a store damaged by an OLDER compaction — whose
/// split tombstoned one half of an `assistant tool_use → user tool_result`
/// pair — carries the orphan half forever. Strict gateways reject that shape
/// with a 400 ("The response must be a response to a preceding message with
/// 'tool_calls'"), so EVERY subsequent turn fails and the session looks
/// wedged "right after compaction". Repair the REQUEST-ONLY copy (the store
/// is never touched): orphans are downgraded to text, and a trailing
/// dangling `tool_use` is kept — a legal "awaiting result" shape.
fn sanitize_request_messages(messages: Vec<Message>) -> Vec<Message> {
    let mut messages = messages;
    crate::compact::repair_pairing(&mut messages, true);
    messages
}

/// Whether an API failure is transient and worth retrying. Providers that report
/// `HTTP <code>: ...` (anthropic-compat) are classified precisely — rate limits,
/// overload, and 5xx retry; auth / bad-request do not. SDK-wrapped errors without
/// a clean status (openai/ollama) are treated as retryable transport failures
/// unless they name an obvious permanent condition.
fn is_retryable_api_error(e: &AgentError) -> bool {
    let AgentError::Provider { message, .. } = e else {
        return false;
    };
    if let Some(code) = parse_http_status(message) {
        // 429 (rate limit) + a few transient client statuses, and ANY 5xx server
        // error — 500/502/503/504, Anthropic 529 overload, and gateway/proxy 52x
        // (common from third-party endpoints) — except 501 Not Implemented, which
        // is permanent.
        return code == 408
            || code == 409
            || code == 425
            || code == 429
            || (code >= 500 && code != 501);
    }
    let m = message.to_ascii_lowercase();
    !(m.contains("invalid")
        || m.contains("unauthorized")
        || m.contains("401")
        || m.contains("403")
        || m.contains("400")
        || m.contains("404"))
}

/// Parse the leading numeric status from a `"HTTP <code> ...: ..."` message.
fn parse_http_status(message: &str) -> Option<u16> {
    let rest = message.strip_prefix("HTTP ")?;
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

/// Exponential backoff for a 1-based retry `attempt`: 1s, 2s, 4s, 8s, … capped at
/// 60s so a long outage doesn't stall each retry indefinitely.
fn retry_backoff(attempt: u32) -> std::time::Duration {
    let secs = 1u64
        .checked_shl(attempt.saturating_sub(1))
        .unwrap_or(u64::MAX)
        .min(60);
    std::time::Duration::from_secs(secs)
}

/// Actual delay before retry `attempt`. Server guidance wins: providers embed
/// the `Retry-After` header as a `retry-after=<secs>` token in the error text
/// (the response object doesn't survive the error path), honored up to 120s.
/// Otherwise [`retry_backoff`] with EQUAL JITTER — uniform in
/// `[base/2, base]` — so a fleet of agents retrying the same outage doesn't
/// stampede the recovering endpoint in lockstep.
fn retry_delay(attempt: u32, error: &AgentError) -> std::time::Duration {
    if let AgentError::Provider { message, .. } = error {
        if let Some(secs) = parse_retry_after(message) {
            return std::time::Duration::from_secs(secs.min(120));
        }
    }
    let base_ms = retry_backoff(attempt).as_millis() as u64;
    let half = base_ms / 2;
    // Cheap jitter without a rand dependency: subsecond clock noise.
    let noise = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| u64::from(d.subsec_nanos()))
        .unwrap_or(0);
    std::time::Duration::from_millis(half + noise % (half + 1))
}

/// Parse the numeric `retry-after=<secs>` token a provider embedded in an
/// error message. HTTP-date forms are ignored (backoff covers those).
fn parse_retry_after(message: &str) -> Option<u64> {
    let idx = message.find("retry-after=")?;
    let digits: String = message[idx + "retry-after=".len()..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse().ok()
}

fn push(store: &Arc<Mutex<MessageStore>>, msg: Message) -> Result<(), AgentError> {
    let mut s = store
        .lock()
        .map_err(|_| AgentError::other("query store lock poisoned"))?;
    s.push(msg)
}

fn child_header(store: &Arc<Mutex<MessageStore>>) -> Header {
    let parent = store.lock().ok().and_then(|s| s.last().map(|m| m.uuid()));
    match parent {
        Some(p) => Header::child_of(p),
        None => Header::new(),
    }
}

/// Reserved key a tool can put in its JSON output to return rich content
/// blocks. Exact shape required: a top-level object holding only this key
/// (an array of Text/Image blocks) plus an optional "text" fallback string.
/// Anything else falls back to plain text.
pub(crate) const CONTENT_BLOCKS_KEY: &str = "__agent_content_blocks__";

/// Convert one tool output into ToolResultContent, honoring the content-blocks
/// sentinel when the provider can preserve image tool results. Error results
/// never parse the sentinel.
fn tool_result_content(
    ok: bool,
    output: &serde_json::Value,
    allow_images: bool,
) -> ToolResultContent {
    let plain = || ToolResultContent::Text(output.to_string());
    if !ok {
        return plain();
    }
    let Some(obj) = output.as_object() else {
        return plain();
    };
    if !obj.contains_key(CONTENT_BLOCKS_KEY) {
        return plain();
    }
    if !obj.keys().all(|k| k == CONTENT_BLOCKS_KEY || k == "text") {
        return plain();
    }
    let text_fallback = || {
        ToolResultContent::Text(
            obj.get("text")
                .and_then(|t| t.as_str())
                .unwrap_or_default()
                .to_string(),
        )
    };
    let Ok(blocks) = serde_json::from_value::<Vec<ContentBlock>>(obj[CONTENT_BLOCKS_KEY].clone())
    else {
        return text_fallback();
    };
    let valid = !blocks.is_empty()
        && blocks.iter().all(|b| match b {
            ContentBlock::Text { .. } => true,
            ContentBlock::Image {
                source: ImageSource::Base64 { data, .. },
            } => data.len() / 4 * 3 <= crate::attachments::MAX_INLINE_IMAGE_BYTES,
            _ => false,
        });
    if !valid {
        return text_fallback();
    }
    if allow_images {
        ToolResultContent::Blocks(blocks)
    } else {
        text_fallback()
    }
}

#[derive(Debug, Clone)]
struct SyntheticToolResult(ToolResultEcho);

#[derive(Debug, Clone)]
struct ToolResultEcho {
    id: String,
    ok: bool,
    output: serde_json::Value,
}

fn synthetic_tool_result(
    tu: &RequestedToolUse,
    ok: bool,
    error_text: String,
) -> SyntheticToolResult {
    SyntheticToolResult(ToolResultEcho {
        id: tu.id.clone(),
        ok,
        output: serde_json::json!({"error": error_text}),
    })
}

/// Forward a synthetic tool result to the consumer. Returns `false` if the
/// consumer has gone away (receiver dropped), so the caller can stop the turn
/// instead of streaming another round into a dead channel.
fn forward_tool_result(
    tx: &mpsc::UnboundedSender<Result<Event, AgentError>>,
    synthetic: SyntheticToolResult,
) -> bool {
    tx.unbounded_send(Ok(Event::ToolResult {
        id: synthetic.0.id,
        ok: synthetic.0.ok,
        output: synthetic.0.output,
    }))
    .is_ok()
}

/// Reactive auto-compaction helper invoked once per turn.
///
/// Reads the live `MessageStore`, sums token estimates, evaluates
/// [`AutoCompactState`], and on `should_compact == true`:
///
/// 1. Calls [`compact_with_hooks`] (which fires PreCompact + PostCompact
///    hooks around [`super::super::compact::summarize::compact_conversation`]).
/// 2. Rewrites the store in-place via [`apply_compaction_to_store`].
/// 3. Forwards an `Event::Notice`-equivalent observability ping by
///    surfacing a synthetic `Event::Unknown`-like marker — for now we
///    rely on hooks for telemetry and only short-circuit on
///    [`CompactError::Aborted`].
///
/// **Failure handling**:
/// - `Aborted` due to outer abort → propagates as [`AgentError::Aborted`].
/// - `Aborted` from a `PreCompact` hook returning `Block` (outer abort
///   NOT fired) → recoverable: surfaces an `Event::Notice`, records a
///   failure for the breaker, and the turn continues.
/// - Any other error → records a failure (advances circuit breaker),
///   surfaces an `Event::Notice`, and the turn continues with the
///   un-compacted store. Three in a row opens the breaker.
///
/// **Direction**: uses [`PartialCompactDirection::EarliestHalf`] so the
/// most recent half of the transcript (including the user message that
/// just arrived via [`QueryLoop::run`]) is preserved verbatim. Without
/// this the current turn's user input would be tombstoned and the
/// provider would never see fresh input.
///
/// **No-progress detection**: after a successful compaction, the
/// post-compact token total is re-evaluated. If still ≥ threshold,
/// [`AutoCompactState::record_no_progress`] is latched so subsequent
/// turns skip auto-compaction entirely (preventing oscillation when
/// the summary itself is dense).
/// Minimum messages to compact in a partial direction. Below this we
/// skip — replacing 1 message with a boundary + summary pair is a net
/// loss in both message count and tokens.
const MIN_PARTIAL_COMPACT_TARGET: usize = 2;

/// After this many consecutive iterations whose tool calls AND results were
/// all byte-identical (3rd identical iteration), inject a user-visible nudge
/// telling the model to change approach — small models can degenerate into
/// replaying the same call verbatim, burning iterations without progress.
const LOOP_GUARD_WARN_STREAK: u32 = 2;

/// Hard stop: at this streak (6th identical iteration) the turn is aborted
/// with an explicit error instead of looping until `max_iterations` (which
/// defaults to unbounded) or the user interrupts.
const LOOP_GUARD_ABORT_STREAK: u32 = 5;

/// Strict-mode thresholds ([`QueryLoopBuilder::strict_loop_guard`]): nudge on
/// the 2nd identical iteration, abort on the 4th. Weak models enter the
/// repetition attractor faster and rarely recover from the nudge.
const LOOP_GUARD_WARN_STREAK_STRICT: u32 = 1;
const LOOP_GUARD_ABORT_STREAK_STRICT: u32 = 3;

/// Same-TOOL streak thresholds — applied ONLY to the static-answer discovery
/// tools in [`STATIC_ANSWER_TOOLS`]: consecutive iterations calling the same
/// such tool with varying args is the "guess another keyword" loop the
/// identical-call guard cannot see (observed: 49 ToolSearch calls hunting
/// for a tool that did not exist). Ordinary tools are exempt on purpose: a
/// long run of sequential FileReads or greps with different targets is real
/// work, and killing it broke legitimate 32-round chains in testing.
const TOOL_STREAK_WARN: u32 = 6;
const TOOL_STREAK_ABORT: u32 = 12;
const TOOL_STREAK_WARN_STRICT: u32 = 4;
const TOOL_STREAK_ABORT_STRICT: u32 = 8;

/// Tools whose answer set is STATIC within a turn — re-querying them with
/// endless keyword variants can never surface anything new, so a long
/// same-tool streak on these is pathological by construction.
const STATIC_ANSWER_TOOLS: &[&str] = &["ToolSearch"];

/// Age-blind microcompact used only on the pinned-context salvage path:
/// every tool result except the most recent few is fair game, current
/// turn included — the alternative is failing the turn outright, and tool
/// results are re-derivable in a way user/assistant text is not. Runs
/// even when the routine microcompact ladder is disabled by config: that
/// toggle governs the maintenance step, not last-resort salvage. The
/// preserve floor is raised at the call site to cover results the model
/// has not seen yet.
const EMERGENCY_MICROCOMPACT: MicrocompactConfig = MicrocompactConfig {
    min_age_turns: 0,
    preserve_last_n: 3,
    min_tokens_per_result: 50,
    protected_tool_names: Vec::new(),
    prune_head_chars: 800,
    prune_tail_chars: 400,
};

/// First same-command-head nudge fires at this many runs within one turn;
/// afterwards the cadence doubles (10th, 20th, …) so the model is nudged,
/// not nagged. Never aborts — re-running a build or test after a real edit
/// is legitimate work.
const BASH_REPEAT_WARN: u32 = 5;

/// Tool names whose success means the workspace changed. Any of these in
/// an iteration clears every command-head repeat count: post-edit re-runs
/// are fresh work, and only edit-free grinding should accumulate.
const HEAD_RESET_TOOLS: &[&str] = &[
    "FileEdit",
    "FileWrite",
    "NotebookEdit",
    "Edit",
    "Write",
    "MultiEdit",
];

/// Normalized "head" of a shell command: leading `cd <path> &&` hops are
/// stripped (location, not action), everything from the first pipe or
/// redirect on is cut (re-running `vitest x | grep A` and `vitest x |
/// grep B` is the same work re-done for a different slice of its output),
/// and the first four tokens identify the action.
fn bash_command_head(command: &str) -> String {
    let mut rest = command.trim();
    while let Some(stripped) = rest.strip_prefix("cd ") {
        let Some((_, after)) = stripped.split_once("&&") else {
            break;
        };
        rest = after.trim_start();
    }
    let action = rest.split(['|', '>']).next().unwrap_or(rest);
    action
        .split_whitespace()
        .take(4)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Detects consecutive iterations with identical (tool calls, results)
/// signatures. `observe` returns how many consecutive PRIOR iterations shared
/// the given signature (0 on first sight or any change).
#[derive(Default)]
struct ToolLoopGuard {
    last_sig: Option<u64>,
    streak: u32,
    /// Sorted, deduped tool-name set of the previous iteration.
    last_tools: Option<Vec<String>>,
    tool_streak: u32,
    /// Per-turn run counts and next nudge point for normalized shell
    /// command heads (see [`bash_command_head`]).
    bash_heads: std::collections::HashMap<String, (u32, u32)>,
}

impl ToolLoopGuard {
    fn observe(&mut self, sig: u64) -> u32 {
        if self.last_sig == Some(sig) {
            self.streak = self.streak.saturating_add(1);
        } else {
            self.last_sig = Some(sig);
            self.streak = 0;
        }
        self.streak
    }

    /// Consecutive iterations whose (deduped) tool-name set was the same
    /// single STATIC-ANSWER tool ([`STATIC_ANSWER_TOOLS`]). Mixed iterations
    /// and ordinary tools reset the streak — they are real work.
    fn observe_tools(&mut self, mut tools: Vec<String>) -> u32 {
        tools.sort_unstable();
        tools.dedup();
        if tools.len() != 1 || !STATIC_ANSWER_TOOLS.contains(&tools[0].as_str()) {
            self.last_tools = None;
            self.tool_streak = 0;
            return 0;
        }
        if self.last_tools.as_ref() == Some(&tools) {
            self.tool_streak = self.tool_streak.saturating_add(1);
        } else {
            self.last_tools = Some(tools);
            self.tool_streak = 0;
        }
        self.tool_streak
    }

    /// Count a run of a normalized shell-command head. Counts span the
    /// WHOLE turn — re-runs interleave with other calls, so a consecutive
    /// streak would never trip. Returns `Some(total)` when the count
    /// reaches the nudge point ([`BASH_REPEAT_WARN`], then doubling).
    fn observe_bash_head(&mut self, head: String) -> Option<u32> {
        if head.is_empty() {
            return None;
        }
        let entry = self.bash_heads.entry(head).or_insert((0, BASH_REPEAT_WARN));
        entry.0 = entry.0.saturating_add(1);
        if entry.0 >= entry.1 {
            entry.1 = entry.0.saturating_mul(2);
            Some(entry.0)
        } else {
            None
        }
    }

    /// The workspace changed (an edit tool ran) — every repeat count
    /// starts over.
    fn reset_bash_heads(&mut self) {
        self.bash_heads.clear();
    }
}

/// Returns whether the store was rewritten (microcompact or LLM compaction) so
/// the caller can invalidate any estimate calibration keyed to the old store.
///
/// `est_offset` is the caller's prompt-estimate calibration (see
/// [`calibrated_prompt_estimate`]): every threshold comparison here uses the
/// calibrated occupancy, keeping this decision consistent with the output
/// clamp — the raw store estimate omits the system prompt, tool schemas, and
/// estimator bias, and an under-count here lets the clamp squeeze completions
/// to the floor before compaction ever fires.
async fn maybe_auto_compact(
    qloop: &QueryLoop,
    state: &Arc<Mutex<AutoCompactState>>,
    abort: &AbortController,
    tx: &mpsc::UnboundedSender<Result<Event, AgentError>>,
    est_offset: i64,
) -> Result<bool, AgentError> {
    // Snapshot once — we want a stable view of the store for token
    // accounting AND for the compaction request itself. Store-size
    // minimums for the LLM partial compaction are checked just before
    // that request, NOT here: a short but overfull store must still
    // reach the pinned-context salvage / hard-stop below.
    let snapshot = snapshot(&qloop.store)?;
    if snapshot.is_empty() {
        return Ok(false);
    }

    let current_tokens: u32 = snapshot
        .iter()
        .map(estimate_tokens)
        .fold(0u32, u32::saturating_add);
    let occupancy = calibrated_prompt_estimate(current_tokens, est_offset);

    let decision = {
        let s = state
            .lock()
            .map_err(|_| AgentError::other("compact state lock poisoned"))?;
        s.evaluate(occupancy, qloop.model_max_tokens)
    };

    // ----- Ladder step ①: microcompact (no LLM call) -----
    // Clearing old tool-result payloads is free, so it runs from the
    // WARNING threshold up — well before LLM summarization is needed —
    // instead of waiting for the auto-compact threshold. When it alone
    // keeps the total under the auto threshold, LLM compaction is skipped
    // this turn. Deliberately NOT recorded as success/failure: the
    // circuit breaker tracks LLM compaction attempts only.
    let mut snapshot = snapshot;
    let mut rewrote = false;
    if qloop.microcompact_enabled
        && occupancy >= crate::compact::microcompact_threshold(qloop.model_max_tokens)
    {
        let threshold = crate::compact::auto_compact_threshold(qloop.model_max_tokens);
        let (mc, new_total, new_snapshot) = {
            let mut store = qloop
                .store
                .lock()
                .map_err(|_| AgentError::other("query store lock poisoned"))?;
            let mc = apply_microcompact_to_store(
                &mut store,
                &MicrocompactConfig {
                    protected_tool_names: qloop.microcompact_protected_tools.clone(),
                    ..MicrocompactConfig::default()
                },
            )?;
            let new_total = store
                .iter()
                .map(estimate_tokens)
                .fold(0u32, u32::saturating_add);
            let new_snapshot: Vec<Message> = store.iter().cloned().collect();
            (mc, new_total, new_snapshot)
        };
        if mc.cleared_count > 0 {
            let _ = tx.unbounded_send(Ok(Event::Notice {
                code: "agent.compact.micro".into(),
                message: format!(
                    "microcompact cleared {} tool result(s), freed ~{} tokens",
                    mc.cleared_count, mc.tokens_freed
                ),
            }));
            snapshot = new_snapshot;
            rewrote = true;
            if calibrated_prompt_estimate(new_total, est_offset) < threshold {
                return Ok(true);
            }
        }
    }

    // ----- Pinned context: emergency salvage, then hard stop -----
    // When compaction is masked (circuit breaker / no-progress latch) the
    // loop used to keep iterating anyway: the output clamp squeezes the
    // completion toward the floor and the provider eventually rejects
    // `prompt + max_tokens` — calls burn while the context sits pinned near
    // 100%. An LLM retry is exactly what the latches exist to prevent (and
    // a Full summarize request near the window overflows the provider input
    // limit itself), so the rescue is the free, deterministic one: an
    // age-blind microcompact that keeps only the last few tool results.
    // Only when even that leaves the store beyond the manual-compact line
    // does the turn fail, with an actionable message.
    // (Skipped when no window is declared: every threshold degenerates to
    // 0 there, and "pinned" is meaningless against an unknown window.)
    if qloop.model_max_tokens > 0
        && matches!(
            decision.reason,
            AutoCompactReason::CircuitBreakerOpen { .. } | AutoCompactReason::NoProgress
        )
    {
        let manual_threshold = crate::compact::manual_compact_threshold(qloop.model_max_tokens);
        let post_total: u32 = snapshot
            .iter()
            .map(estimate_tokens)
            .fold(0u32, u32::saturating_add);
        let mut post_occupancy = calibrated_prompt_estimate(post_total, est_offset);
        if post_occupancy >= manual_threshold {
            // Never clear results the model has not SEEN yet: everything
            // after the last Assistant message answers tool calls whose
            // output the model still awaits (this check runs before the
            // next streaming request). Handing it "[cleared]" for a call
            // it just made would re-trigger the exact re-run loop this
            // path exists to stop.
            let unseen_results: usize = snapshot
                .iter()
                .rev()
                .take_while(|m| !matches!(m, Message::Assistant { .. }))
                .filter_map(|m| match m {
                    Message::User { content, .. } => Some(
                        content
                            .iter()
                            .filter(|b| matches!(b, ContentBlock::ToolResult { .. }))
                            .count(),
                    ),
                    _ => None,
                })
                .sum();
            let emergency = MicrocompactConfig {
                preserve_last_n: EMERGENCY_MICROCOMPACT
                    .preserve_last_n
                    .max(unseen_results.min(u32::MAX as usize) as u32),
                protected_tool_names: qloop.microcompact_protected_tools.clone(),
                ..EMERGENCY_MICROCOMPACT
            };
            let (mc, new_total) = {
                let mut store = qloop
                    .store
                    .lock()
                    .map_err(|_| AgentError::other("query store lock poisoned"))?;
                let mc = apply_microcompact_to_store(&mut store, &emergency)?;
                let new_total = store
                    .iter()
                    .map(estimate_tokens)
                    .fold(0u32, u32::saturating_add);
                (mc, new_total)
            };
            if mc.cleared_count > 0 {
                let _ = tx.unbounded_send(Ok(Event::Notice {
                    code: "agent.compact.emergency".into(),
                    message: format!(
                        "context pinned with compaction masked — emergency-cleared {} old \
                         tool result(s), freed ~{} tokens",
                        mc.cleared_count, mc.tokens_freed
                    ),
                }));
                rewrote = true;
                post_occupancy = calibrated_prompt_estimate(new_total, est_offset);
                // The store shrank materially; the earlier no-progress
                // verdict no longer describes it, so let future auto
                // compaction try again once occupancy rebuilds. (The
                // failure circuit breaker stays: clearing results proves
                // nothing about the provider.)
                if post_occupancy < crate::compact::auto_compact_threshold(qloop.model_max_tokens) {
                    let mut s = state
                        .lock()
                        .map_err(|_| AgentError::other("compact state lock poisoned"))?;
                    s.reset_no_progress();
                }
            }
        }
        if post_occupancy >= manual_threshold {
            return Err(AgentError::other(
                "context window is effectively full and auto-compaction cannot make progress \
                 (compaction circuit breaker / no-progress latch is set); run a manual compact \
                 or clear the session before continuing",
            ));
        }
        return Ok(rewrote);
    }
    if !decision.should_compact {
        return Ok(rewrote);
    }

    // Auto-compaction uses EarliestHalf, which compacts `len/2` messages.
    // Skip when the store is too short to split usefully.
    if snapshot.len() < 2 || snapshot.len() / 2 < MIN_PARTIAL_COMPACT_TARGET {
        return Ok(rewrote);
    }

    // Run the compaction. `EarliestHalf` keeps the recent half — vital
    // because [`QueryLoop::run`] just pushed the user's prompt and we
    // need the model to see it on the next streaming turn.
    //
    // Use a child abort so an internal abort during summarization
    // doesn't poison the outer loop. Parent → child propagates, child
    // → parent does not.
    let mut request = CompactWithHooksRequest::new(&snapshot, qloop.model.clone())
        .with_trigger(CompactTrigger::Auto)
        .with_direction(PartialCompactDirection::EarliestHalf)
        .with_abort(abort.child());
    if let Some(ci) = qloop.compact_instructions.as_deref() {
        request = request.with_custom_instructions(ci);
    }
    let outcome = compact_with_hooks(&qloop.hooks, &*qloop.provider, request).await;

    match outcome {
        Ok(result) => {
            let auto_threshold = crate::compact::auto_compact_threshold(qloop.model_max_tokens);
            // Splice result into the live store under a short critical
            // section.
            let new_total = {
                let mut store = qloop
                    .store
                    .lock()
                    .map_err(|_| AgentError::other("query store lock poisoned"))?;
                apply_compaction_to_store(&mut store, &result)?;
                store
                    .iter()
                    .map(estimate_tokens)
                    .fold(0u32, u32::saturating_add)
            };
            {
                let mut s = state
                    .lock()
                    .map_err(|_| AgentError::other("compact state lock poisoned"))?;
                s.record_success();
                // No-progress detection: if the post-compact CALIBRATED
                // occupancy is still ≥ threshold, latch the flag so we
                // don't oscillate. The user can clear it via
                // [`AutoCompactState::reset_no_progress`]. The offset is
                // slightly stale for the rewritten store (its per-message
                // bias belonged to replaced messages), erring toward
                // latching — the safe direction, since the alternative is
                // looping useless compactions.
                if calibrated_prompt_estimate(new_total, est_offset) >= auto_threshold {
                    s.record_no_progress();
                }
            }
            if let Some(sm) = &qloop.session_memory {
                match promote_to_store(sm.as_ref(), &result).await {
                    Ok(n) if n > 0 => {
                        let _ = tx.unbounded_send(Ok(Event::Notice {
                            code: "agent.compact.memory".into(),
                            message: format!("promoted {n} analysis bullet(s) to session memory"),
                        }));
                    }
                    Ok(_) => {}
                    Err(e) => {
                        let _ = tx.unbounded_send(Ok(Event::Notice {
                            code: "agent.compact.memory_failed".into(),
                            message: e.to_string(),
                        }));
                    }
                }
            }
            let _ = tx.unbounded_send(Ok(Event::Notice {
                code: "agent.compact.ok".into(),
                message: format!(
                    "auto-compact {} → {} tokens ({} messages replaced)",
                    result.pre_compact_tokens,
                    result.post_compact_tokens,
                    result.replaced_uuids.len()
                ),
            }));
            Ok(true)
        }
        Err(CompactError::Aborted) => {
            // Distinguish outer abort vs. PreCompact hook block. The
            // hook path returns Aborted without setting the abort
            // token, so we can recover.
            if abort.is_aborted() {
                Err(AgentError::Aborted(
                    abort.reason().unwrap_or_else(|| "compact aborted".into()),
                ))
            } else {
                if let Ok(mut s) = state.lock() {
                    s.record_failure();
                }
                let _ = tx.unbounded_send(Ok(Event::Notice {
                    code: "agent.compact.blocked".into(),
                    message: "PreCompact hook blocked auto-compaction".into(),
                }));
                Ok(rewrote)
            }
        }
        Err(other) => {
            if let Ok(mut s) = state.lock() {
                s.record_failure();
            }
            let _ = tx.unbounded_send(Ok(Event::Notice {
                code: "agent.compact.failed".into(),
                message: other.to_string(),
            }));
            Ok(rewrote)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use futures::StreamExt;

    use super::*;
    use crate::permission::{PermissionMode, RuleSource};
    use crate::provider::ProviderCapabilities;
    use crate::testing::MockProvider;
    use crate::tool::Tool;

    #[test]
    fn builder_carries_thinking_and_reasoning_effort() {
        let provider: Arc<dyn Provider> = Arc::new(MockProvider::new(vec![]));
        let b = QueryLoop::builder(provider, "m")
            .thinking(ThinkingConfig::new(4096))
            .reasoning_effort("high");
        let ql = b.build();
        assert_eq!(ql.thinking.map(|t| t.max_tokens), Some(4096));
        assert_eq!(ql.reasoning_effort.as_deref(), Some("high"));
    }

    #[test]
    fn temperature_dropped_when_thinking_is_applied() {
        // Anthropic 400s if extended thinking and a non-default temperature
        // are both set. Thinking is attached unconditionally now — whether
        // the manual budget_tokens shape fits the request's output budget is
        // a provider-shape concern handled in anthropic::build_request_body,
        // not this loop-level guard — so an explicit temperature must still
        // be omitted (provider default) rather than sent alongside it.
        let req = StreamRequest::new("m", vec![]);
        let tc = ThinkingConfig::new(2048);
        let req = apply_temperature_and_thinking(req, Some(0.3), Some(tc));
        assert!(req.thinking.is_some());
        assert!(req.temperature.is_none());
    }

    #[test]
    fn temperature_passes_through_when_thinking_absent() {
        // No thinking configured: the caller's temperature must reach the
        // request unchanged, exactly as before this guard was added.
        let req = StreamRequest::new("m", vec![]);
        let req = apply_temperature_and_thinking(req, Some(0.3), None);
        assert!(req.thinking.is_none());
        assert_eq!(req.temperature, Some(0.3));
    }

    #[test]
    fn clamp_output_keeps_prompt_plus_completion_within_window() {
        // The reported failure: ~1M window, 384k output ceiling, a mid-turn
        // history of 900_839 tokens. 900_839 + 384_000 = 1_284_839 > 1_048_565.
        let window = 1_048_565;
        let prompt = 900_839;
        let out = clamp_output_to_window(384_000, prompt, window);
        assert!(out < 384_000, "must clamp below the static ceiling");
        assert!(
            prompt + out < window,
            "prompt {prompt} + completion {out} must fit window {window}"
        );
        // Reserve is window/32 here (32_767 > 8_192 floor).
        assert_eq!(out, window - prompt - window / OUTPUT_CLAMP_RESERVE_DIVISOR);
    }

    #[test]
    fn clamp_output_is_a_noop_with_headroom_and_never_zero() {
        // Small prompt on a large window → the full configured ceiling.
        assert_eq!(clamp_output_to_window(384_000, 40_000, 1_048_565), 384_000);
        // Prompt fills the window → floored, never max_tokens=0.
        assert_eq!(
            clamp_output_to_window(16_384, 1_048_000, 1_048_565),
            MIN_CLAMPED_OUTPUT
        );
        // Degenerate window is left untouched (no real model has one).
        assert_eq!(clamp_output_to_window(16_384, 0, 1), 16_384);
    }

    #[test]
    fn delta_calibration_predicts_grown_prompt() {
        // Iteration 1: 100k message estimate, provider counted 130k (30k of
        // system + tools). Iteration 2's store grew to 880k of messages.
        let (actual_prev, est_prev) = (130_000u32, 100_000u32);
        let msg_now = 880_000u32;
        let predicted = actual_prev.saturating_add(msg_now.saturating_sub(est_prev));
        // Additive overhead carries forward (not multiplied): ~910k, not 1.14M.
        assert_eq!(predicted, 910_000);
        let out = clamp_output_to_window(384_000, predicted, 1_048_565);
        assert!(predicted + out < 1_048_565, "calibrated request must fit");
    }

    #[test]
    fn overhead_estimate_counts_system_and_tools() {
        let tools = vec![ToolDefinition::new(
            "grep",
            "search files",
            serde_json::json!({"type": "object"}),
        )];
        let with = estimate_prompt_overhead(Some("you are a helpful agent"), &tools);
        let without = estimate_prompt_overhead(None, &[]);
        assert!(with > without, "system + tools must add to the estimate");
    }

    #[test]
    fn retryable_classifies_http_statuses() {
        let prov = |m: &str| AgentError::provider("anthropic", m);
        // Rate limit + ANY 5xx server/gateway/overload error → retry.
        assert!(is_retryable_api_error(&prov(
            "HTTP 429 Too Many Requests: slow down"
        )));
        assert!(is_retryable_api_error(&prov(
            "HTTP 500 Internal Server Error: x"
        )));
        assert!(is_retryable_api_error(&prov(
            "HTTP 503 Service Unavailable: x"
        )));
        assert!(is_retryable_api_error(&prov("HTTP 504 Gateway Timeout: x")));
        assert!(is_retryable_api_error(&prov(
            "HTTP 522 Connection Timed Out: cf"
        )));
        assert!(is_retryable_api_error(&prov("HTTP 529 Overloaded: x")));
        // Auth / bad request / not-implemented → do NOT retry.
        assert!(!is_retryable_api_error(&prov(
            "HTTP 401 Unauthorized: bad key"
        )));
        assert!(!is_retryable_api_error(&prov(
            "HTTP 400 Bad Request: schema"
        )));
        assert!(!is_retryable_api_error(&prov(
            "HTTP 501 Not Implemented: x"
        )));
        // No status = transport failure → retry; aborts/others → no.
        assert!(is_retryable_api_error(&prov(
            "error sending request: connection reset"
        )));
        assert!(!is_retryable_api_error(&AgentError::Aborted("user".into())));
        assert!(!is_retryable_api_error(&prov("invalid x-api-key header")));
    }

    #[test]
    fn backoff_grows_then_caps() {
        let s = |a| retry_backoff(a).as_secs();
        assert_eq!((s(1), s(2), s(3), s(4)), (1, 2, 4, 8));
        assert!(s(2) > s(1) && s(3) > s(2)); // each longer than the last…
        assert_eq!(s(7), 60); // …until the 60s cap
        assert_eq!(s(10), 60);
    }

    #[test]
    fn retry_delay_honors_server_retry_after() {
        // A provider that embedded Retry-After overrides the backoff curve.
        let e = AgentError::provider("anthropic", "HTTP 429 retry-after=7: slow down");
        assert_eq!(retry_delay(1, &e).as_secs(), 7);
        // Clamped to 120s so a hostile header can't park the loop forever.
        let e = AgentError::provider("anthropic", "HTTP 503 retry-after=9999: x");
        assert_eq!(retry_delay(1, &e).as_secs(), 120);
    }

    #[test]
    fn retry_delay_jitters_within_equal_jitter_band() {
        // No server hint → equal jitter in [base/2, base] for the attempt.
        let e = AgentError::provider("x", "HTTP 500: boom");
        let base = retry_backoff(4).as_millis() as u64; // 8000ms
        for _ in 0..50 {
            let d = retry_delay(4, &e).as_millis() as u64;
            assert!(d >= base / 2 && d <= base, "delay {d} out of band");
        }
    }

    #[test]
    fn parse_retry_after_extracts_seconds() {
        assert_eq!(parse_retry_after("HTTP 429 retry-after=12: x"), Some(12));
        assert_eq!(parse_retry_after("HTTP 500: no header"), None);
        // HTTP-date form is ignored (non-numeric) — backoff covers it.
        assert_eq!(parse_retry_after("retry-after=Wed, 21 Oct"), None);
    }

    fn img_sentinel(extra_key: bool) -> serde_json::Value {
        let mut o = serde_json::json!({
            CONTENT_BLOCKS_KEY: [
                { "type": "image",
                  "source": { "type": "base64", "media_type": "image/png",
                              "data": "iVBORw0KGgoAAAANSUhEUg==" } }
            ],
            "text": "screenshot saved: /tmp/s.png",
        });
        if extra_key {
            o.as_object_mut().unwrap().insert("rogue".into(), 1.into());
        }
        o
    }

    #[test]
    fn sentinel_converts_to_blocks_when_images_allowed() {
        let c = tool_result_content(true, &img_sentinel(false), true);
        match c {
            ToolResultContent::Blocks(b) => {
                assert_eq!(b.len(), 1);
                assert!(matches!(b[0], ContentBlock::Image { .. }));
            }
            other => panic!("expected Blocks, got {other:?}"),
        }
    }

    #[test]
    fn sentinel_strips_to_text_fallback_when_images_disallowed() {
        let c = tool_result_content(true, &img_sentinel(false), false);
        assert_eq!(
            c,
            ToolResultContent::Text("screenshot saved: /tmp/s.png".into())
        );
    }

    #[test]
    fn sentinel_requires_exact_shape() {
        let v = img_sentinel(true);
        assert_eq!(
            tool_result_content(true, &v, true),
            ToolResultContent::Text(v.to_string())
        );
    }

    #[test]
    fn sentinel_ignored_on_error_results() {
        let v = img_sentinel(false);
        assert_eq!(
            tool_result_content(false, &v, true),
            ToolResultContent::Text(v.to_string())
        );
    }

    #[test]
    fn sentinel_rejects_non_text_image_blocks() {
        let v = serde_json::json!({
            CONTENT_BLOCKS_KEY: [ { "type": "thinking", "thinking": "x", "signature": null } ],
            "text": "fallback",
        });
        assert_eq!(
            tool_result_content(true, &v, true),
            ToolResultContent::Text("fallback".into())
        );
    }

    #[test]
    fn sentinel_rejects_oversized_inline_image() {
        let big = "A".repeat((crate::attachments::MAX_INLINE_IMAGE_BYTES / 3 + 2) * 4);
        let v = serde_json::json!({
            CONTENT_BLOCKS_KEY: [ { "type": "image",
                "source": { "type": "base64", "media_type": "image/png", "data": big } } ],
            "text": "fallback",
        });
        assert_eq!(
            tool_result_content(true, &v, true),
            ToolResultContent::Text("fallback".into())
        );
    }

    #[test]
    fn plain_output_unchanged() {
        let v = serde_json::json!({"ok": true});
        assert_eq!(
            tool_result_content(true, &v, true),
            ToolResultContent::Text(v.to_string())
        );
    }

    #[derive(Debug)]
    struct EchoTool {
        name: String,
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            &self.name
        }
        fn description(&self) -> &str {
            "echo input as output"
        }
        fn input_schema(&self) -> serde_json::Value {
            serde_json::json!({})
        }
        async fn call(
            &self,
            _ctx: &ToolUseContext,
            input: serde_json::Value,
        ) -> Result<serde_json::Value, AgentError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(input)
        }
    }

    fn echo_registry(name: &str, calls: Arc<AtomicUsize>) -> Arc<ToolRegistry> {
        let mut r = ToolRegistry::new();
        r.register(Arc::new(EchoTool {
            name: name.into(),
            calls,
        }));
        Arc::new(r)
    }

    /// Tool that aborts the turn from inside its own call — simulates the user
    /// hitting Esc (or the consumer dropping the stream) while a tool runs.
    #[derive(Debug)]
    struct AbortingTool {
        abort: AbortController,
    }

    #[derive(Debug)]
    struct PendingDropTool {
        started: Arc<tokio::sync::Notify>,
        dropped: Arc<std::sync::atomic::AtomicBool>,
    }

    #[async_trait]
    impl Tool for PendingDropTool {
        fn name(&self) -> &str {
            "pending"
        }
        fn description(&self) -> &str {
            "waits until its task is hard-cancelled"
        }
        fn input_schema(&self) -> serde_json::Value {
            serde_json::json!({})
        }
        async fn call(
            &self,
            _ctx: &ToolUseContext,
            _input: serde_json::Value,
        ) -> Result<serde_json::Value, AgentError> {
            struct DropProbe(Arc<std::sync::atomic::AtomicBool>);
            impl Drop for DropProbe {
                fn drop(&mut self) {
                    self.0.store(true, Ordering::SeqCst);
                }
            }
            let _probe = DropProbe(self.dropped.clone());
            self.started.notify_one();
            futures::future::pending::<()>().await;
            unreachable!()
        }
    }

    #[async_trait]
    impl Tool for AbortingTool {
        fn name(&self) -> &str {
            "slow"
        }
        fn description(&self) -> &str {
            "aborts the turn mid-call"
        }
        fn input_schema(&self) -> serde_json::Value {
            serde_json::json!({})
        }
        async fn call(
            &self,
            ctx: &ToolUseContext,
            input: serde_json::Value,
        ) -> Result<serde_json::Value, AgentError> {
            self.abort.abort_with_reason("user interrupted");
            ctx.abort.cancelled().await;
            Ok(input)
        }
    }

    /// Provider that captures every `StreamRequest` it receives.
    /// Returns an empty stream so the loop terminates after a single
    /// turn when paired with a no-tool registry.
    #[derive(Debug)]
    struct CapturingProvider {
        captured: Arc<std::sync::Mutex<Vec<StreamRequest>>>,
    }

    #[async_trait]
    impl Provider for CapturingProvider {
        fn id(&self) -> &str {
            "capturing"
        }
        fn capabilities(&self) -> crate::provider::ProviderCapabilities {
            crate::provider::ProviderCapabilities {
                supports_tool_use: true,
                ..Default::default()
            }
        }
        async fn stream(
            &self,
            req: StreamRequest,
            _abort: AbortController,
        ) -> Result<Box<dyn EventStream>, AgentError> {
            if let Ok(mut g) = self.captured.lock() {
                g.push(req);
            }
            Ok(Box::new(futures::stream::empty()))
        }
    }

    #[derive(Debug)]
    struct ScriptedCapturingProvider {
        captured: Arc<std::sync::Mutex<Vec<StreamRequest>>>,
        capabilities: ProviderCapabilities,
        turns: std::sync::Mutex<Vec<Vec<Event>>>,
    }

    #[async_trait]
    impl Provider for ScriptedCapturingProvider {
        fn id(&self) -> &str {
            "scripted-capturing"
        }

        fn capabilities(&self) -> ProviderCapabilities {
            self.capabilities
        }

        async fn stream(
            &self,
            req: StreamRequest,
            _abort: AbortController,
        ) -> Result<Box<dyn EventStream>, AgentError> {
            self.captured.lock().unwrap().push(req);
            let turn = {
                let mut turns = self.turns.lock().unwrap();
                if turns.is_empty() {
                    Vec::new()
                } else {
                    turns.remove(0)
                }
            };
            Ok(Box::new(futures::stream::iter(turn.into_iter().map(Ok))))
        }
    }

    fn sentinel_capture_provider(
        captured: Arc<std::sync::Mutex<Vec<StreamRequest>>>,
        tool_result_images: bool,
    ) -> Arc<ScriptedCapturingProvider> {
        Arc::new(ScriptedCapturingProvider {
            captured,
            capabilities: ProviderCapabilities {
                supports_tool_use: true,
                tool_result_images,
                ..Default::default()
            },
            turns: std::sync::Mutex::new(vec![
                vec![
                    Event::ToolUse {
                        id: "tu_screenshot".into(),
                        name: "sentinel".into(),
                        input: img_sentinel(false),
                    },
                    Event::Result {
                        data: ResultData {
                            stop_reason: Some("tool_use".into()),
                            ..Default::default()
                        },
                    },
                ],
                vec![Event::Result {
                    data: ResultData {
                        stop_reason: Some("end_turn".into()),
                        ..Default::default()
                    },
                }],
            ]),
        })
    }

    fn latest_tool_result_content(captured: &[StreamRequest]) -> &ToolResultContent {
        let Some(StreamRequest { messages, .. }) = captured.last() else {
            panic!("expected a captured follow-up request");
        };
        let Some(Message::User { content, .. }) = messages.last() else {
            panic!("expected latest message to be user tool results");
        };
        let Some(ContentBlock::ToolResult { content, .. }) = content.first() else {
            panic!("expected a tool result block");
        };
        content
    }

    #[tokio::test]
    async fn loop_forwards_registered_tools_to_provider() {
        let captured = Arc::new(std::sync::Mutex::new(Vec::new()));
        let provider = Arc::new(CapturingProvider {
            captured: captured.clone(),
        });
        let registry = echo_registry("calc", Arc::new(AtomicUsize::new(0)));
        let qloop = QueryLoop::builder(provider, "m").tools(registry).build();
        let mut stream = qloop.run("hi", AbortController::new()).await.unwrap();
        while stream.next().await.is_some() {}
        let captured = captured.lock().unwrap();
        assert!(
            !captured.is_empty(),
            "should have captured at least one request"
        );
        assert_eq!(captured[0].tools.len(), 1);
        assert_eq!(captured[0].tools[0].name, "calc");
    }

    #[tokio::test]
    async fn truncated_tool_call_nudges_model_to_reissue() {
        // The provider dropped a tool call whose input the output-token cap
        // cut mid-write (Result tagged via metadata, stop_reason max_tokens,
        // no ToolUse event). The driver must NOT end the turn: it pushes a
        // nudge user message and re-requests, and the second (clean) turn
        // finishes normally.
        let captured = Arc::new(std::sync::Mutex::new(Vec::new()));
        let provider = Arc::new(ScriptedCapturingProvider {
            captured: captured.clone(),
            capabilities: ProviderCapabilities {
                supports_tool_use: true,
                ..Default::default()
            },
            turns: std::sync::Mutex::new(vec![
                vec![
                    Event::TextDelta {
                        delta: "let me ask".into(),
                    },
                    Event::Result {
                        data: ResultData {
                            stop_reason: Some("max_tokens".into()),
                            metadata: [(
                                "truncated_tool_use".to_string(),
                                serde_json::json!({"id": "t1", "name": "AskUserQuestion", "bytes": 396}),
                            )]
                            .into_iter()
                            .collect(),
                            ..Default::default()
                        },
                    },
                ],
                vec![
                    Event::TextDelta {
                        delta: "done".into(),
                    },
                    Event::Result {
                        data: ResultData {
                            stop_reason: Some("end_turn".into()),
                            ..Default::default()
                        },
                    },
                ],
            ]),
        });
        let qloop = QueryLoop::builder(provider, "m").build();
        let mut stream = qloop.run("hi", AbortController::new()).await.unwrap();
        let mut err = None;
        while let Some(item) = stream.next().await {
            if let Err(e) = item {
                err = Some(e);
            }
        }
        assert!(err.is_none(), "turn must complete cleanly: {err:?}");
        let captured = captured.lock().unwrap();
        assert_eq!(captured.len(), 2, "expected exactly one retry request");
        let Some(Message::User { content, .. }) = captured[1].messages.last() else {
            panic!("retry request must end with the nudge user message");
        };
        let Some(ContentBlock::Text { text }) = content.first() else {
            panic!("expected a text nudge block");
        };
        assert!(
            text.contains("cut off by the output-token limit"),
            "got: {text}"
        );
        assert!(text.contains("AskUserQuestion"), "got: {text}");
    }

    #[tokio::test]
    async fn run_blocks_pushes_rich_user_content() {
        let captured = Arc::new(std::sync::Mutex::new(Vec::new()));
        let provider = Arc::new(CapturingProvider {
            captured: captured.clone(),
        });
        let qloop = QueryLoop::builder(provider, "m").build();
        let content = vec![
            ContentBlock::Text {
                text: "describe".into(),
            },
            ContentBlock::Image {
                source: crate::message::ImageSource::Base64 {
                    media_type: "image/png".into(),
                    data: "abc123".into(),
                },
            },
        ];
        let mut stream = qloop
            .run_blocks(content.clone(), AbortController::new())
            .await
            .unwrap();
        while stream.next().await.is_some() {}

        let captured = captured.lock().unwrap();
        let Message::User {
            content: observed, ..
        } = &captured[0].messages[0]
        else {
            panic!("expected user message");
        };
        assert_eq!(observed, &content);
    }

    #[tokio::test]
    async fn tool_result_sentinel_reaches_provider_as_blocks_when_supported() {
        let captured = Arc::new(std::sync::Mutex::new(Vec::new()));
        let provider = sentinel_capture_provider(captured.clone(), true);
        let registry = echo_registry("sentinel", Arc::new(AtomicUsize::new(0)));
        let perms = Arc::new(PermissionManager::new().allow(RuleSource::User, "sentinel"));
        let qloop = QueryLoop::builder(provider, "mock")
            .tools(registry)
            .permissions(perms)
            .build();

        let mut stream = qloop
            .run("screenshot", AbortController::new())
            .await
            .unwrap();
        while stream.next().await.is_some() {}

        let captured = captured.lock().unwrap();
        match latest_tool_result_content(&captured) {
            ToolResultContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 1);
                assert!(matches!(blocks[0], ContentBlock::Image { .. }));
            }
            other => panic!("expected Blocks, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn tool_result_sentinel_falls_back_to_text_when_unsupported() {
        let captured = Arc::new(std::sync::Mutex::new(Vec::new()));
        let provider = sentinel_capture_provider(captured.clone(), false);
        let registry = echo_registry("sentinel", Arc::new(AtomicUsize::new(0)));
        let perms = Arc::new(PermissionManager::new().allow(RuleSource::User, "sentinel"));
        let qloop = QueryLoop::builder(provider, "mock")
            .tools(registry)
            .permissions(perms)
            .build();

        let mut stream = qloop
            .run("screenshot", AbortController::new())
            .await
            .unwrap();
        while stream.next().await.is_some() {}

        let captured = captured.lock().unwrap();
        assert_eq!(
            latest_tool_result_content(&captured),
            &ToolResultContent::Text("screenshot saved: /tmp/s.png".into())
        );
    }

    #[tokio::test]
    async fn single_turn_text_only() {
        let provider = Arc::new(MockProvider::with_turns(vec![vec![
            Event::TextDelta {
                delta: "hi ".into(),
            },
            Event::TextDelta {
                delta: "there".into(),
            },
            Event::Result {
                data: ResultData {
                    stop_reason: Some("end_turn".into()),
                    model: Some("mock-1".into()),
                    metadata: Default::default(),
                },
            },
        ]]));
        let qloop = QueryLoop::builder(provider, "mock-1")
            .permissions(Arc::new(
                PermissionManager::new().with_mode(PermissionMode::Bypass),
            ))
            .build();
        let mut stream = qloop.run("hello", AbortController::new()).await.unwrap();
        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            events.push(item.unwrap());
        }
        // 2 TextDelta + 1 final Result.
        assert_eq!(events.len(), 3);
        assert!(matches!(events.last(), Some(Event::Result { .. })));
    }

    #[tokio::test]
    async fn two_turn_tool_use_loop() {
        // Turn 1: assistant emits a tool_use + Result(stop_reason="tool_use").
        // Turn 2: after we feed tool_result back, assistant emits text +
        // Result(stop_reason="end_turn") → loop terminates.
        let provider = Arc::new(MockProvider::with_turns(vec![
            vec![
                Event::ToolUse {
                    id: "tu_1".into(),
                    name: "echo".into(),
                    input: serde_json::json!({"value": 42}),
                },
                Event::Result {
                    data: ResultData {
                        stop_reason: Some("tool_use".into()),
                        ..Default::default()
                    },
                },
            ],
            vec![
                Event::TextDelta {
                    delta: "got 42".into(),
                },
                Event::Result {
                    data: ResultData {
                        stop_reason: Some("end_turn".into()),
                        ..Default::default()
                    },
                },
            ],
        ]));

        let calls = Arc::new(AtomicUsize::new(0));
        let registry = echo_registry("echo", calls.clone());
        let perms = Arc::new(PermissionManager::new().allow(RuleSource::User, "echo"));

        let qloop = QueryLoop::builder(provider, "mock")
            .tools(registry)
            .permissions(perms)
            .build();

        let store = qloop.store.clone();
        let mut stream = qloop
            .run("call echo", AbortController::new())
            .await
            .unwrap();
        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            events.push(item.unwrap());
        }
        // Echo tool was called exactly once.
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        // Stream emitted ToolUse + ToolResult + TextDelta + Result.
        assert!(events.iter().any(|e| matches!(e, Event::ToolUse { .. })));
        assert!(events
            .iter()
            .any(|e| matches!(e, Event::ToolResult { ok: true, .. })));
        assert!(events.iter().any(|e| matches!(e, Event::TextDelta { .. })));
        assert!(matches!(events.last(), Some(Event::Result { .. })));

        // Store: User → Assistant(tool_use) → User(tool_result) → Assistant(text).
        let snap: Vec<_> = store.lock().unwrap().iter().cloned().collect();
        assert_eq!(snap.len(), 4);
        assert!(matches!(snap[0], Message::User { .. }));
        assert!(matches!(snap[1], Message::Assistant { .. }));
        assert!(matches!(snap[2], Message::User { .. }));
        assert!(matches!(snap[3], Message::Assistant { .. }));
    }

    #[tokio::test]
    async fn interrupted_tool_call_leaves_no_dangling_tool_use() {
        // The provider asks for a tool; the tool aborts the turn mid-call.
        // The store must still end with a tool_result for that tool_use, or
        // the next request would 400 ("tool_use without tool_result").
        let provider = Arc::new(MockProvider::with_turns(vec![vec![
            Event::ToolUse {
                id: "tu_1".into(),
                name: "slow".into(),
                input: serde_json::json!({}),
            },
            Event::Result {
                data: ResultData {
                    stop_reason: Some("tool_use".into()),
                    ..Default::default()
                },
            },
        ]]));
        let abort = AbortController::new();
        let mut r = ToolRegistry::new();
        r.register(Arc::new(AbortingTool {
            abort: abort.clone(),
        }));
        let perms = Arc::new(PermissionManager::new().allow(RuleSource::User, "slow"));
        let qloop = QueryLoop::builder(provider, "mock")
            .tools(Arc::new(r))
            .permissions(perms)
            .build();

        let store = qloop.store.clone();
        let mut stream = qloop.run("go", abort.clone()).await.unwrap();
        while stream.next().await.is_some() {}

        let snap: Vec<_> = store.lock().unwrap().iter().cloned().collect();
        let tool_use_ids: Vec<String> = snap
            .iter()
            .filter_map(|m| match m {
                Message::Assistant { content, .. } => Some(content),
                _ => None,
            })
            .flatten()
            .filter_map(|b| match b {
                ContentBlock::ToolUse { id, .. } => Some(id.clone()),
                _ => None,
            })
            .collect();
        let result_ids: std::collections::HashSet<String> = snap
            .iter()
            .filter_map(|m| match m {
                Message::User { content, .. } => Some(content),
                _ => None,
            })
            .flatten()
            .filter_map(|b| match b {
                ContentBlock::ToolResult { tool_use_id, .. } => Some(tool_use_id.clone()),
                _ => None,
            })
            .collect();
        assert!(
            !tool_use_ids.is_empty(),
            "a tool_use was recorded in the store"
        );
        for id in &tool_use_ids {
            assert!(
                result_ids.contains(id),
                "tool_use {id} must have a matching tool_result (no dangling)"
            );
        }
    }

    #[tokio::test]
    async fn hard_drop_reaches_nested_tool_quiescence_and_repairs_store() {
        let provider = Arc::new(MockProvider::with_turns(vec![vec![
            Event::ToolUse {
                id: "tu_pending".into(),
                name: "pending".into(),
                input: serde_json::json!({}),
            },
            Event::Result {
                data: ResultData {
                    stop_reason: Some("tool_use".into()),
                    ..Default::default()
                },
            },
        ]]));
        let started = Arc::new(tokio::sync::Notify::new());
        let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(PendingDropTool {
            started: started.clone(),
            dropped: dropped.clone(),
        }));
        let abort = AbortController::new();
        let activity = abort.activity();
        let qloop = QueryLoop::builder(provider, "mock")
            .tools(Arc::new(registry))
            .permissions(Arc::new(
                PermissionManager::new().allow(RuleSource::User, "pending"),
            ))
            .build();
        let store = qloop.store.clone();
        let stream = qloop.run("go", abort).await.unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(1), started.notified())
            .await
            .expect("pending tool started");

        drop(stream);

        let has_repair = store.lock().unwrap().iter().any(|message| match message {
            Message::User { content, .. } => content.iter().any(|block| {
                matches!(
                    block,
                    ContentBlock::ToolResult { tool_use_id, is_error: true, .. }
                        if tool_use_id == "tu_pending"
                )
            }),
            _ => false,
        });
        assert!(has_repair, "store repair must be synchronous with drop");
        tokio::time::timeout(
            std::time::Duration::from_secs(1),
            activity.wait_for_quiescence(),
        )
        .await
        .expect("nested driver/tool cleanup reached quiescence");
        assert!(dropped.load(Ordering::SeqCst));
        assert_eq!(activity.active_workers(), 0);
    }

    #[tokio::test]
    async fn partial_abort_fills_only_unanswered_tool_uses() {
        // Two tool_uses in one assistant turn; one tool aborts the turn. Both
        // ids must end with a result (the finished one real or synthetic, the
        // aborted one synthetic) — never a dangling tool_use.
        let provider = Arc::new(MockProvider::with_turns(vec![vec![
            Event::ToolUse {
                id: "tu_1".into(),
                name: "echo".into(),
                input: serde_json::json!({ "v": 1 }),
            },
            Event::ToolUse {
                id: "tu_2".into(),
                name: "slow".into(),
                input: serde_json::json!({}),
            },
            Event::Result {
                data: ResultData {
                    stop_reason: Some("tool_use".into()),
                    ..Default::default()
                },
            },
        ]]));
        let abort = AbortController::new();
        let mut r = ToolRegistry::new();
        r.register(Arc::new(EchoTool {
            name: "echo".into(),
            calls: Arc::new(AtomicUsize::new(0)),
        }));
        r.register(Arc::new(AbortingTool {
            abort: abort.clone(),
        }));
        let perms = Arc::new(
            PermissionManager::new()
                .allow(RuleSource::User, "echo")
                .allow(RuleSource::User, "slow"),
        );
        let qloop = QueryLoop::builder(provider, "mock")
            .tools(Arc::new(r))
            .permissions(perms)
            .build();

        let store = qloop.store.clone();
        let mut stream = qloop.run("go", abort.clone()).await.unwrap();
        while stream.next().await.is_some() {}

        let snap: Vec<_> = store.lock().unwrap().iter().cloned().collect();
        let result_ids: std::collections::HashSet<String> = snap
            .iter()
            .filter_map(|m| match m {
                Message::User { content, .. } => Some(content),
                _ => None,
            })
            .flatten()
            .filter_map(|b| match b {
                ContentBlock::ToolResult { tool_use_id, .. } => Some(tool_use_id.clone()),
                _ => None,
            })
            .collect();
        assert!(
            result_ids.contains("tu_1"),
            "tu_1 has a result: {result_ids:?}"
        );
        assert!(
            result_ids.contains("tu_2"),
            "tu_2 has a result: {result_ids:?}"
        );
    }

    #[tokio::test]
    async fn permission_deny_synthesizes_failed_tool_result_and_does_not_dispatch() {
        let provider = Arc::new(MockProvider::with_turns(vec![
            vec![
                Event::ToolUse {
                    id: "tu_1".into(),
                    name: "echo".into(),
                    input: serde_json::json!({}),
                },
                Event::Result {
                    data: ResultData {
                        stop_reason: Some("tool_use".into()),
                        ..Default::default()
                    },
                },
            ],
            vec![Event::Result {
                data: ResultData {
                    stop_reason: Some("end_turn".into()),
                    ..Default::default()
                },
            }],
        ]));
        let calls = Arc::new(AtomicUsize::new(0));
        let registry = echo_registry("echo", calls.clone());
        let perms = Arc::new(PermissionManager::new().deny(RuleSource::Policy, "echo"));

        let qloop = QueryLoop::builder(provider, "mock")
            .tools(registry)
            .permissions(perms)
            .build();

        let mut stream = qloop.run("hi", AbortController::new()).await.unwrap();
        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            events.push(item.unwrap());
        }
        // Echo tool was NOT called — denied at permission step.
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        // Stream still has a synthetic ToolResult ok=false.
        assert!(events
            .iter()
            .any(|e| matches!(e, Event::ToolResult { ok: false, .. })));
    }

    #[tokio::test]
    async fn before_tool_use_block_hook_skips_dispatch() {
        let provider = Arc::new(MockProvider::with_turns(vec![
            vec![
                Event::ToolUse {
                    id: "tu_1".into(),
                    name: "echo".into(),
                    input: serde_json::json!({}),
                },
                Event::Result {
                    data: ResultData {
                        stop_reason: Some("tool_use".into()),
                        ..Default::default()
                    },
                },
            ],
            vec![Event::Result {
                data: ResultData {
                    stop_reason: Some("end_turn".into()),
                    ..Default::default()
                },
            }],
        ]));
        let calls = Arc::new(AtomicUsize::new(0));
        let registry = echo_registry("echo", calls.clone());
        let perms = Arc::new(PermissionManager::new().allow(RuleSource::User, "echo"));

        let blocking_hook =
            Arc::new(crate::hook::RustHookHandler::new(
                "blocker",
                |event| match event {
                    HookEvent::BeforeToolUse { .. } => HookOutcome::Block,
                    _ => HookOutcome::Ok,
                },
            ));
        let mut hooks = HookRunner::new();
        hooks.register(blocking_hook);

        let qloop = QueryLoop::builder(provider, "mock")
            .tools(registry)
            .permissions(perms)
            .hooks(Arc::new(hooks))
            .build();

        let mut stream = qloop.run("hi", AbortController::new()).await.unwrap();
        while let Some(_item) = stream.next().await {}
        // Hook blocked dispatch; echo was not called.
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    fn compact_response() -> &'static str {
        "<analysis>- prior turns happened</analysis><summary>Prior context summarized.</summary>"
    }

    /// Build a builder pre-populated with `extra` messages plus aggressive
    /// compact thresholds so any 2+ messages trigger immediately.
    ///
    /// Takes `Arc<dyn Provider>` (not `Arc<MockProvider>`) so tests that
    /// need to inspect the outgoing `StreamRequest` (e.g. asserting on
    /// `compact_instructions`) can plug in a custom `Provider` impl.
    fn compact_loop_builder(
        provider: Arc<dyn Provider>,
        store: Arc<Mutex<MessageStore>>,
    ) -> QueryLoopBuilder {
        QueryLoop::builder(provider, "mock")
            .permissions(Arc::new(
                PermissionManager::new().with_mode(PermissionMode::Bypass),
            ))
            .store(store)
            // Saturates every threshold to 0: any non-empty snapshot fires.
            .model_max_tokens(20_001)
    }

    fn preload_store(messages: Vec<Message>) -> Arc<Mutex<MessageStore>> {
        let mut s = MessageStore::new();
        for m in messages {
            s.push(m).unwrap();
        }
        Arc::new(Mutex::new(s))
    }

    fn user_text(text: &str) -> Message {
        Message::User {
            header: Header::new(),
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    fn assistant_text(text: &str) -> Message {
        Message::Assistant {
            header: Header::new(),
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn steer_injects_a_user_message_before_the_next_round_trip() {
        // A message sent into the steer inbox is drained at the top of the
        // loop and appended as a User turn before the request snapshot, so
        // the model sees it. Sending it before run() means iteration 1 picks
        // it up deterministically.
        let provider = Arc::new(MockProvider::with_turns(vec![vec![
            Event::TextDelta { delta: "ok".into() },
            Event::Result {
                data: ResultData {
                    stop_reason: Some("end_turn".into()),
                    ..Default::default()
                },
            },
        ]]));
        let store = preload_store(vec![]);
        let (tx_steer, rx_steer) = mpsc::unbounded::<Vec<ContentBlock>>();
        tx_steer
            .unbounded_send(vec![ContentBlock::Text {
                text: "actually, also do X".into(),
            }])
            .unwrap();
        let qloop = QueryLoop::builder(provider, "mock")
            .permissions(Arc::new(
                PermissionManager::new().with_mode(PermissionMode::Bypass),
            ))
            .store(store.clone())
            .steer(Arc::new(std::sync::Mutex::new(rx_steer)))
            .build();

        let mut stream = qloop.run("hi", AbortController::new()).await.unwrap();
        let mut saw_steer_notice = false;
        while let Some(item) = stream.next().await {
            if let Ok(Event::Notice { code, .. }) = &item {
                if code == "agent.steer" {
                    saw_steer_notice = true;
                }
            }
        }
        assert!(saw_steer_notice, "a steer notice should be emitted");
        // The injected message is in the store as a User turn.
        let snap: Vec<_> = store.lock().unwrap().iter().cloned().collect();
        let injected = snap.iter().any(|m| match m {
            Message::User { content, .. } => content
                .iter()
                .any(|b| matches!(b, ContentBlock::Text { text } if text.contains("also do X"))),
            _ => false,
        });
        assert!(injected, "steered message must land in the store");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn auto_compact_fires_on_threshold_hit_and_rewrites_store() {
        // Provider scripts: turn 0 = compaction summary (consumed by
        // compact_with_hooks), turn 1 = the regular assistant streaming
        // reply that follows, ending with end_turn.
        let provider = Arc::new(MockProvider::with_turns(vec![
            vec![
                Event::TextDelta {
                    delta: compact_response().into(),
                },
                Event::Result {
                    data: Default::default(),
                },
            ],
            vec![
                Event::TextDelta { delta: "ok".into() },
                Event::Result {
                    data: ResultData {
                        stop_reason: Some("end_turn".into()),
                        ..Default::default()
                    },
                },
            ],
        ]));
        // Preload 4 messages whose token mass sits in the FIRST two, so
        // EarliestHalf (token-midpoint split) compacts exactly those and
        // preserves the small trailing 2 verbatim. After run() pushes
        // the user prompt, snapshot length is 5, replaced = 2.
        let store = preload_store(vec![
            user_text(&"u1 ".repeat(200)),
            assistant_text(&"a1 ".repeat(200)),
            user_text("u2"),
            assistant_text("a2"),
        ]);
        let qloop = compact_loop_builder(provider.clone(), store.clone()).build();

        let mut stream = qloop.run("trigger", AbortController::new()).await.unwrap();
        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            events.push(item.unwrap());
        }

        // Notice event (code = "agent.compact.ok") proves auto-compact fired.
        let notice = events.iter().find_map(|e| match e {
            Event::Notice { code, message } if code == "agent.compact.ok" => Some(message.clone()),
            _ => None,
        });
        assert!(
            notice.is_some(),
            "expected agent.compact.ok notice, got events: {events:?}"
        );

        // Both scripted turns were consumed (compact + assistant).
        assert_eq!(provider.remaining_turns(), 0);

        // EarliestHalf snapshot of 5 (4 preload + run-pushed user)
        // tombstones the first 2 in place AND inserts boundary+summary
        // AT the boundary (after the tombstones, before preserved).
        // Final layout (8 messages):
        //   0: Tombstone(u1)
        //   1: Tombstone(a1)
        //   2: System boundary
        //   3: User summary "[Context summary]…"
        //   4: User u2 (preserved)
        //   5: Assistant a2 (preserved)
        //   6: User "trigger" (preserved fresh prompt)
        //   7: Assistant fallback reply
        let snap: Vec<_> = store.lock().unwrap().iter().cloned().collect();
        assert_eq!(snap.len(), 8, "got {snap:?}");
        assert!(matches!(snap[0], Message::Tombstone { .. }));
        assert!(matches!(snap[1], Message::Tombstone { .. }));
        assert!(matches!(
            snap[2],
            Message::System { ref text, .. } if text == crate::compact::COMPACT_BOUNDARY_TEXT
        ));
        match &snap[3] {
            Message::User { content, .. } => {
                let text = content
                    .iter()
                    .find_map(|b| match b {
                        ContentBlock::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .expect("summary user msg has text block");
                assert!(text.starts_with("[Context summary]"));
            }
            other => panic!("expected User summary at index 3, got {other:?}"),
        }
        // Critical: the fresh "trigger" prompt sits AFTER the summary
        // (at idx 6), so Anthropic's view (skipping Tombstone+System) is
        // [summary-as-user, u2, a2, trigger, assistant_reply] —
        // chronologically coherent.
        let trigger_idx = snap.iter().position(|m| match m {
            Message::User { content, .. } => content
                .iter()
                .any(|b| matches!(b, ContentBlock::Text { text } if text == "trigger")),
            _ => false,
        });
        assert!(trigger_idx.is_some(), "fresh user turn was tombstoned away");
        assert!(
            trigger_idx.unwrap() > 3,
            "fresh user prompt should sit after summary at idx 3, got {:?}",
            trigger_idx
        );
        // Final assistant reply.
        assert!(matches!(snap[7], Message::Assistant { .. }));
    }

    #[test]
    fn calibrated_prompt_estimate_applies_signed_offset() {
        // Positive offset (estimator under-counts): pulled up.
        assert_eq!(calibrated_prompt_estimate(1_000, 500), 1_500);
        // Negative offset (estimator over-counts, e.g. base64 images):
        // pulled down instead of saturating away.
        assert_eq!(calibrated_prompt_estimate(1_000, -400), 600);
        // Never below zero.
        assert_eq!(calibrated_prompt_estimate(100, -1_000), 0);
        // Never above u32::MAX.
        assert_eq!(calibrated_prompt_estimate(u32::MAX, 10), u32::MAX);
    }

    /// The regression this guards: the raw store estimate omits the system
    /// prompt / tool schemas / estimator bias, so real occupancy could sit
    /// past the auto-compact threshold while the raw estimate stayed under
    /// it — the output clamp (which IS calibrated) then squeezed
    /// `max_tokens` to the floor and the model truncated tool JSON mid-call.
    /// The decision must use the same calibrated occupancy as the clamp.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn auto_compact_decision_uses_calibrated_occupancy() {
        // model_max 40k → auto threshold = 40k - 20k (summary reserve)
        // - 13k (buffer) = 7k. Raw store estimate ≈ 1.5k.
        let messages = vec![
            user_text(&"u1 ".repeat(500)),
            assistant_text(&"a1 ".repeat(500)),
            user_text("u2"),
            assistant_text("a2"),
        ];

        // Raw estimate under threshold + zero offset → no compaction, no
        // provider turn consumed, store untouched.
        let provider = Arc::new(MockProvider::with_turns(vec![vec![
            Event::TextDelta {
                delta: compact_response().into(),
            },
            Event::Result {
                data: Default::default(),
            },
        ]]));
        let store = preload_store(messages.clone());
        let qloop = QueryLoop::builder(provider.clone() as Arc<dyn Provider>, "mock")
            .permissions(Arc::new(
                PermissionManager::new().with_mode(PermissionMode::Bypass),
            ))
            .store(store.clone())
            .model_max_tokens(40_000)
            .build();
        let (tx, _rx) = mpsc::unbounded();
        let rewrote = maybe_auto_compact(
            &qloop,
            &qloop.compact_state.clone(),
            &AbortController::new(),
            &tx,
            0,
        )
        .await
        .unwrap();
        assert!(!rewrote, "raw estimate is under threshold; must not fire");
        assert_eq!(provider.remaining_turns(), 1);
        assert_eq!(store.lock().unwrap().iter().count(), 4);

        // Same store, but a calibration offset (true prompt minus estimate,
        // as the provider reported it) pushes occupancy past the threshold
        // → compaction fires and rewrites the store.
        let store = preload_store(messages);
        let qloop = QueryLoop::builder(provider.clone() as Arc<dyn Provider>, "mock")
            .permissions(Arc::new(
                PermissionManager::new().with_mode(PermissionMode::Bypass),
            ))
            .store(store.clone())
            .model_max_tokens(40_000)
            .build();
        let (tx, _rx) = mpsc::unbounded();
        let rewrote = maybe_auto_compact(
            &qloop,
            &qloop.compact_state.clone(),
            &AbortController::new(),
            &tx,
            10_000,
        )
        .await
        .unwrap();
        assert!(rewrote, "calibrated occupancy crossed the threshold");
        assert_eq!(provider.remaining_turns(), 0);
        let has_boundary = store.lock().unwrap().iter().any(|m| {
            matches!(
                m,
                Message::System { text, .. } if text == crate::compact::COMPACT_BOUNDARY_TEXT
            )
        });
        assert!(has_boundary, "store should carry the compact boundary");
    }

    #[tokio::test]
    async fn a_pinned_context_with_masked_compaction_fails_the_turn() {
        // The no-progress latch masks compaction; with the store past the
        // manual-compact line the loop must error out with instructions
        // instead of iterating toward a provider 400. Below that line the
        // masked decision still just skips compaction.
        let provider = Arc::new(MockProvider::with_turns(vec![]));
        // ~50K estimated tokens on a 40K window: past every threshold.
        // (Four messages, so the partial-compact minimum-target early
        // return doesn't short-circuit before the hard-stop check.)
        let messages = (0..4)
            .map(|i| user_tool_result(&format!("tu_{i}"), &"x".repeat(50_000)))
            .collect::<Vec<_>>();
        let store = preload_store(messages);
        let qloop = QueryLoop::builder(provider.clone() as Arc<dyn Provider>, "mock")
            .permissions(Arc::new(
                PermissionManager::new().with_mode(PermissionMode::Bypass),
            ))
            .store(store.clone())
            .model_max_tokens(40_000)
            .microcompact(false)
            .build();
        qloop.compact_state.lock().unwrap().record_no_progress();
        let (tx, _rx) = mpsc::unbounded();
        let err = maybe_auto_compact(
            &qloop,
            &qloop.compact_state.clone(),
            &AbortController::new(),
            &tx,
            0,
        )
        .await
        .expect_err("pinned context must fail the turn");
        assert!(
            err.to_string().contains("manual compact"),
            "actionable message: {err}"
        );

        // A small store with the same latch: no error, no compaction.
        let store = preload_store(
            (0..4)
                .map(|i| user_tool_result(&format!("tu_{i}"), "small"))
                .collect::<Vec<_>>(),
        );
        let qloop = QueryLoop::builder(provider as Arc<dyn Provider>, "mock")
            .permissions(Arc::new(
                PermissionManager::new().with_mode(PermissionMode::Bypass),
            ))
            .store(store)
            .model_max_tokens(40_000)
            .microcompact(false)
            .build();
        qloop.compact_state.lock().unwrap().record_no_progress();
        let (tx, _rx) = mpsc::unbounded();
        let rewrote = maybe_auto_compact(
            &qloop,
            &qloop.compact_state.clone(),
            &AbortController::new(),
            &tx,
            0,
        )
        .await
        .unwrap();
        assert!(!rewrote);
    }

    #[tokio::test]
    async fn a_pinned_context_rescued_by_emergency_clearing_continues_the_turn() {
        // Mostly tool-result bulk: the age-blind emergency microcompact
        // keeps the last 3 results, dropping occupancy below the
        // manual-compact line — the turn continues instead of erroring.
        // Each result follows its assistant tool_use turn (as in a real
        // store), so only the final result counts as unseen.
        let provider = Arc::new(MockProvider::with_turns(vec![]));
        let mut messages = Vec::new();
        for i in 0..10 {
            messages.push(assistant_text(&format!("calling tool {i}")));
            messages.push(user_tool_result(&format!("tu_{i}"), &"x".repeat(20_000)));
        }
        let store = preload_store(messages);
        let qloop = QueryLoop::builder(provider as Arc<dyn Provider>, "mock")
            .permissions(Arc::new(
                PermissionManager::new().with_mode(PermissionMode::Bypass),
            ))
            .store(store.clone())
            // 50K: occupancy (50K) still pins the context, but the manual
            // line (27K) sits well below the post-salvage total (~17K with
            // head/tail pruning) — the rescue must not depend on the exact
            // marker budget.
            .model_max_tokens(50_000)
            .microcompact(false)
            .build();
        qloop.compact_state.lock().unwrap().record_no_progress();
        let (tx, mut rx) = mpsc::unbounded();
        let rewrote = maybe_auto_compact(
            &qloop,
            &qloop.compact_state.clone(),
            &AbortController::new(),
            &tx,
            0,
        )
        .await
        .expect("emergency clearing rescues the turn");
        assert!(rewrote, "the store was rewritten by the salvage pass");
        let mut saw_emergency = false;
        while let Ok(ev) = rx.try_recv() {
            if matches!(&ev, Ok(Event::Notice { code, .. }) if code == "agent.compact.emergency") {
                saw_emergency = true;
            }
        }
        assert!(saw_emergency, "the salvage pass announces itself");
        let placeholders = store
            .lock()
            .unwrap()
            .iter()
            .filter(|m| {
                matches!(m, Message::User { content, .. } if matches!(
                    &content[0],
                    ContentBlock::ToolResult { content, .. }
                        if tool_result_text_contains(content, crate::compact::CLEARED_PLACEHOLDER)
                ))
            })
            .count();
        assert_eq!(placeholders, 7, "everything but the last 3 results cleared");
        // Post-salvage occupancy is still above the auto threshold, so the
        // no-progress latch stays latched.
        assert!(qloop.compact_state.lock().unwrap().no_progress);
    }

    #[test]
    fn bash_command_head_normalizes_location_and_output_filters() {
        // `cd` hops and pipe/redirect tails don't change what work runs.
        assert_eq!(
            bash_command_head(
                "cd /w/agent-frontend && npx vitest run src/a.test.tsx 2>&1 | grep -A22 x"
            ),
            "npx vitest run src/a.test.tsx"
        );
        assert_eq!(
            bash_command_head("npx vitest run src/a.test.tsx --reporter=verbose | head -30"),
            "npx vitest run src/a.test.tsx"
        );
        // Different grep patterns stay distinct heads (real search work).
        assert_ne!(
            bash_command_head("grep -rn alpha src"),
            bash_command_head("grep -rn beta src")
        );
        assert_eq!(bash_command_head(""), "");
    }

    #[test]
    fn bash_head_guard_nudges_at_five_then_doubles() {
        let mut g = ToolLoopGuard::default();
        let run = |g: &mut ToolLoopGuard| g.observe_bash_head("npx vitest run a".into());
        for _ in 0..4 {
            assert_eq!(run(&mut g), None);
        }
        assert_eq!(run(&mut g), Some(5));
        for _ in 0..4 {
            assert_eq!(run(&mut g), None);
        }
        assert_eq!(run(&mut g), Some(10));
        // A different head keeps its own independent count.
        assert_eq!(g.observe_bash_head("cargo test".into()), None);
        // Empty heads (no command) never nudge.
        assert_eq!(g.observe_bash_head(String::new()), None);
        // An edit resets every count: post-edit re-runs are fresh work.
        g.reset_bash_heads();
        for _ in 0..4 {
            assert_eq!(run(&mut g), None);
        }
        assert_eq!(run(&mut g), Some(5));
    }

    #[test]
    fn tool_loop_guard_counts_consecutive_identical_signatures() {
        let mut g = ToolLoopGuard::default();
        assert_eq!(g.observe(1), 0);
        assert_eq!(g.observe(1), 1);
        assert_eq!(g.observe(1), 2);
        // A different signature resets the streak.
        assert_eq!(g.observe(2), 0);
        assert_eq!(g.observe(1), 0);
        assert_eq!(g.observe(1), 1);
    }

    /// Scripts the provider to request the SAME tool call (same name + input,
    /// fresh ids) every turn. The echo tool returns the identical result each
    /// time, so the guard must nudge at the 3rd identical iteration and abort
    /// the turn at the 6th instead of looping to `max_iterations` (unbounded
    /// by default) — the observed deepseek failure mode: 137 identical greps
    /// until the user pressed Esc.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn tool_loop_guard_nudges_then_aborts_identical_iterations() {
        let same_call = |id: &str| {
            vec![
                Event::ToolUse {
                    id: id.into(),
                    name: "calc".into(),
                    input: serde_json::json!({"q": "same"}),
                },
                Event::Result {
                    data: ResultData {
                        stop_reason: Some("tool_use".into()),
                        ..Default::default()
                    },
                },
            ]
        };
        // More scripted turns than the guard allows — the abort must fire
        // before the script runs dry.
        let turns: Vec<Vec<Event>> = (0..10).map(|i| same_call(&format!("t{i}"))).collect();
        let provider = Arc::new(MockProvider::with_turns(turns));
        let calls = Arc::new(AtomicUsize::new(0));
        let qloop = QueryLoop::builder(provider, "m")
            .permissions(Arc::new(
                PermissionManager::new().with_mode(PermissionMode::Bypass),
            ))
            .tools(echo_registry("calc", calls.clone()))
            .build();
        let store = qloop.store.clone();

        let mut stream = qloop.run("go", AbortController::new()).await.unwrap();
        let mut events = Vec::new();
        let mut err = None;
        while let Some(item) = stream.next().await {
            match item {
                Ok(e) => events.push(e),
                Err(e) => err = Some(e),
            }
        }

        // 6th identical iteration → hard abort with a distinct error.
        let err = err.expect("loop guard must terminate the turn with an error");
        assert!(
            err.to_string().contains("tool-call loop detected"),
            "unexpected error: {err}"
        );
        assert_eq!(
            calls.load(Ordering::SeqCst),
            6,
            "the guard stops after the 6th identical call, not at the script's end"
        );
        // The 3rd identical iteration emitted the nudge notice…
        assert!(
            events.iter().any(|e| matches!(
                e,
                Event::Notice { code, .. } if code == "agent.loop.repeat"
            )),
            "expected the agent.loop.repeat notice, got {events:?}"
        );
        // …and injected the loop-guard reminder into the transcript, where
        // the model can actually see it.
        let has_nudge = store.lock().unwrap().iter().any(|m| {
            matches!(
                m,
                Message::User { content, .. } if content.iter().any(|b| matches!(
                    b,
                    ContentBlock::Text { text } if text.contains("Loop guard")
                ))
            )
        });
        assert!(has_nudge, "loop-guard nudge must be in the store");
        // The terminal Result carries the distinguishing stop reason.
        let stop = events.iter().rev().find_map(|e| match e {
            Event::Result { data } => data.stop_reason.clone(),
            _ => None,
        });
        assert_eq!(stop.as_deref(), Some("tool_loop"));
    }

    /// The "guess another keyword" spiral: the SAME tool called every
    /// iteration with DIFFERENT args (and different results) — invisible to
    /// the identical-call guard — must nudge at the 7th call and abort at
    /// the 13th. Observed in the wild as 49 ToolSearch calls hunting for a
    /// tool that did not exist.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn same_tool_varying_args_trips_the_tool_streak_guard() {
        let call = |i: usize| {
            vec![
                Event::ToolUse {
                    id: format!("t{i}"),
                    name: "ToolSearch".into(),
                    input: serde_json::json!({ "q": format!("variant {i}") }),
                },
                Event::Result {
                    data: ResultData {
                        stop_reason: Some("tool_use".into()),
                        ..Default::default()
                    },
                },
            ]
        };
        let turns: Vec<Vec<Event>> = (0..30).map(call).collect();
        let provider = Arc::new(MockProvider::with_turns(turns));
        let calls = Arc::new(AtomicUsize::new(0));
        let qloop = QueryLoop::builder(provider, "m")
            .permissions(Arc::new(
                PermissionManager::new().with_mode(PermissionMode::Bypass),
            ))
            .tools(echo_registry("ToolSearch", calls.clone()))
            .build();

        let mut stream = qloop.run("go", AbortController::new()).await.unwrap();
        let mut events = Vec::new();
        let mut err = None;
        while let Some(item) = stream.next().await {
            match item {
                Ok(e) => events.push(e),
                Err(e) => err = Some(e),
            }
        }
        let err = err.expect("tool-streak guard terminates the turn");
        assert!(
            err.to_string().contains("varying arguments"),
            "unexpected error: {err}"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 13, "abort on the 13th call");
        assert!(
            events.iter().any(|e| matches!(
                e,
                Event::Notice { code, message } if code == "agent.loop.repeat"
                    && message.contains("same tool")
            )),
            "expected the same-tool nudge notice"
        );
    }

    /// Strict mode (weak models): abort lands on the 4th identical iteration
    /// instead of the 6th.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn strict_loop_guard_aborts_on_the_fourth_identical_iteration() {
        let same_call = |id: &str| {
            vec![
                Event::ToolUse {
                    id: id.into(),
                    name: "calc".into(),
                    input: serde_json::json!({"q": "same"}),
                },
                Event::Result {
                    data: ResultData {
                        stop_reason: Some("tool_use".into()),
                        ..Default::default()
                    },
                },
            ]
        };
        let turns: Vec<Vec<Event>> = (0..10).map(|i| same_call(&format!("t{i}"))).collect();
        let provider = Arc::new(MockProvider::with_turns(turns));
        let calls = Arc::new(AtomicUsize::new(0));
        let qloop = QueryLoop::builder(provider, "m")
            .permissions(Arc::new(
                PermissionManager::new().with_mode(PermissionMode::Bypass),
            ))
            .tools(echo_registry("calc", calls.clone()))
            .strict_loop_guard(true)
            .build();

        let mut stream = qloop.run("go", AbortController::new()).await.unwrap();
        let mut err = None;
        while let Some(item) = stream.next().await {
            if let Err(e) = item {
                err = Some(e);
            }
        }
        assert!(err
            .expect("strict guard terminates the turn")
            .to_string()
            .contains("tool-call loop detected"));
        assert_eq!(calls.load(Ordering::SeqCst), 4, "strict abort on the 4th");
    }

    /// Alternating between two different TOOLS must never trip either guard
    /// tier — the exact guard sees differing calls and the same-tool streak
    /// resets on every tool change.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn tool_loop_guard_ignores_alternating_calls() {
        let call = |id: &str, tool: &str, last: bool| {
            vec![
                Event::ToolUse {
                    id: id.into(),
                    name: tool.into(),
                    input: serde_json::json!({ "q": "work" }),
                },
                Event::Result {
                    data: ResultData {
                        stop_reason: Some(if last { "end_turn" } else { "tool_use" }.into()),
                        ..Default::default()
                    },
                },
            ]
        };
        let mut turns: Vec<Vec<Event>> = Vec::new();
        for i in 0..8 {
            let tool = if i % 2 == 0 { "calc" } else { "calc2" };
            turns.push(call(&format!("t{i}"), tool, false));
        }
        turns.push(vec![
            Event::TextDelta { delta: "ok".into() },
            Event::Result {
                data: ResultData {
                    stop_reason: Some("end_turn".into()),
                    ..Default::default()
                },
            },
        ]);
        let provider = Arc::new(MockProvider::with_turns(turns));
        let calls = Arc::new(AtomicUsize::new(0));
        let registry = {
            let mut r = ToolRegistry::new();
            r.register(Arc::new(EchoTool {
                name: "calc".into(),
                calls: calls.clone(),
            }));
            r.register(Arc::new(EchoTool {
                name: "calc2".into(),
                calls: calls.clone(),
            }));
            Arc::new(r)
        };
        let qloop = QueryLoop::builder(provider, "m")
            .permissions(Arc::new(
                PermissionManager::new().with_mode(PermissionMode::Bypass),
            ))
            .tools(registry)
            .build();

        let mut stream = qloop.run("go", AbortController::new()).await.unwrap();
        let mut saw_err = false;
        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            match item {
                Ok(e) => events.push(e),
                Err(_) => saw_err = true,
            }
        }
        assert!(!saw_err, "alternating calls must not trip the loop guard");
        assert_eq!(calls.load(Ordering::SeqCst), 8, "all scripted calls ran");
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, Event::Notice { code, .. } if code == "agent.loop.repeat")),
            "no nudge for non-identical iterations"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn auto_compact_disabled_does_not_consume_provider_turn() {
        // Only script the assistant turn. If auto-compact tried to run
        // it would fail because the second turn's events don't parse as
        // <analysis>/<summary>.
        let provider = Arc::new(MockProvider::with_turns(vec![vec![
            Event::TextDelta {
                delta: "hello".into(),
            },
            Event::Result {
                data: ResultData {
                    stop_reason: Some("end_turn".into()),
                    ..Default::default()
                },
            },
        ]]));
        let store = preload_store(vec![user_text("u1"), assistant_text("a1")]);
        let qloop = compact_loop_builder(provider.clone(), store.clone())
            .auto_compact(false)
            .build();

        let mut stream = qloop.run("trigger", AbortController::new()).await.unwrap();
        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            events.push(item.unwrap());
        }
        // No compact notice, and the single scripted turn was consumed.
        assert_eq!(provider.remaining_turns(), 0);
        assert!(!events.iter().any(
            |e| matches!(e, Event::Notice { code, .. } if code.starts_with("agent.compact."))
        ));
        // Store: u1, a1, trigger, assistant — 4, no tombstones.
        let snap: Vec<_> = store.lock().unwrap().iter().cloned().collect();
        assert_eq!(snap.len(), 4);
        assert!(snap.iter().all(|m| !matches!(m, Message::Tombstone { .. })));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn auto_compact_failure_surfaces_notice_and_loop_continues() {
        // Turn 0 = empty stream (compaction returns EmptyResponse).
        // Turn 1 = regular assistant reply ending the loop.
        let provider = Arc::new(MockProvider::with_turns(vec![
            vec![Event::Result {
                data: Default::default(),
            }],
            vec![
                Event::TextDelta {
                    delta: "fallback".into(),
                },
                Event::Result {
                    data: ResultData {
                        stop_reason: Some("end_turn".into()),
                        ..Default::default()
                    },
                },
            ],
        ]));
        // Preload 4 messages so EarliestHalf compacts 2 (≥ MIN_PARTIAL_COMPACT_TARGET).
        let store = preload_store(vec![
            user_text("u1"),
            assistant_text("a1"),
            user_text("u2"),
            assistant_text("a2"),
        ]);
        let qloop = compact_loop_builder(provider.clone(), store.clone()).build();

        let mut stream = qloop.run("trigger", AbortController::new()).await.unwrap();
        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            events.push(item.unwrap());
        }
        // agent.compact.failed Notice event surfaced.
        assert!(events.iter().any(|e| matches!(
            e,
            Event::Notice { code, .. } if code == "agent.compact.failed"
        )));
        // Loop still completed with a final Result.
        assert!(matches!(events.last(), Some(Event::Result { .. })));
        // Both turns consumed (failed compact + assistant).
        assert_eq!(provider.remaining_turns(), 0);
        // Store unchanged by failed compaction (no tombstones).
        let snap: Vec<_> = store.lock().unwrap().iter().cloned().collect();
        assert!(snap.iter().all(|m| !matches!(m, Message::Tombstone { .. })));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn pre_compact_hook_block_is_recoverable_not_fatal() {
        // The blocking hook returns Block on PreCompact. compact_with_hooks
        // converts that to CompactError::Aborted, but our integration
        // distinguishes it from a real outer abort and continues.
        let provider = Arc::new(MockProvider::with_turns(vec![vec![
            Event::TextDelta {
                delta: "still alive".into(),
            },
            Event::Result {
                data: ResultData {
                    stop_reason: Some("end_turn".into()),
                    ..Default::default()
                },
            },
        ]]));
        let blocker = Arc::new(crate::hook::RustHookHandler::new(
            "pre-compact-blocker",
            |event| match event {
                HookEvent::PreCompact { .. } => HookOutcome::Block,
                _ => HookOutcome::Ok,
            },
        ));
        let mut hooks = HookRunner::new();
        hooks.register(blocker);

        // Preload 4 so EarliestHalf target_count ≥ MIN_PARTIAL_COMPACT_TARGET.
        let store = preload_store(vec![
            user_text("u1"),
            assistant_text("a1"),
            user_text("u2"),
            assistant_text("a2"),
        ]);
        let qloop = compact_loop_builder(provider.clone(), store.clone())
            .hooks(Arc::new(hooks))
            .build();

        let mut stream = qloop.run("trigger", AbortController::new()).await.unwrap();
        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            events.push(item.unwrap());
        }
        // agent.compact.blocked notice (not a fatal Aborted error).
        assert!(events.iter().any(|e| matches!(
            e,
            Event::Notice { code, .. } if code == "agent.compact.blocked"
        )));
        // Loop continued and completed normally.
        assert!(matches!(events.last(), Some(Event::Result { .. })));
        assert_eq!(provider.remaining_turns(), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn shared_compact_state_persists_failures_across_runs() {
        // Each run() spawns a fresh drive(), but a shared
        // Arc<Mutex<AutoCompactState>> survives across calls so that
        // circuit-breaker counts and the no_progress latch persist.
        // Verify by failing two compactions in run #1 (provider returns
        // empty), then making another call: the breaker advances to 2
        // failures rather than resetting to 0 each run.
        let provider = Arc::new(MockProvider::with_turns(vec![
            // run #1 turn 0 — failed compact (empty)
            vec![Event::Result {
                data: Default::default(),
            }],
            // run #1 turn 1 — assistant reply
            vec![
                Event::TextDelta { delta: "a".into() },
                Event::Result {
                    data: ResultData {
                        stop_reason: Some("end_turn".into()),
                        ..Default::default()
                    },
                },
            ],
            // run #2 turn 0 — failed compact (empty)
            vec![Event::Result {
                data: Default::default(),
            }],
            // run #2 turn 1 — assistant reply
            vec![
                Event::TextDelta { delta: "b".into() },
                Event::Result {
                    data: ResultData {
                        stop_reason: Some("end_turn".into()),
                        ..Default::default()
                    },
                },
            ],
        ]));
        let store = preload_store(vec![
            user_text("u1"),
            assistant_text("a1"),
            user_text("u2"),
            assistant_text("a2"),
        ]);
        let shared_state = Arc::new(Mutex::new(AutoCompactState::new()));

        let qloop = compact_loop_builder(provider.clone(), store.clone())
            .compact_state(shared_state.clone())
            .build();
        let mut stream = qloop.run("first", AbortController::new()).await.unwrap();
        while stream.next().await.is_some() {}

        // After run #1: 1 compact failure recorded.
        assert_eq!(shared_state.lock().unwrap().consecutive_failures, 1);

        // Round 2: rebuild the loop sharing the same state + store.
        let qloop2 = compact_loop_builder(provider.clone(), store.clone())
            .compact_state(shared_state.clone())
            .build();
        let mut stream2 = qloop2.run("second", AbortController::new()).await.unwrap();
        while stream2.next().await.is_some() {}

        // After run #2: counter should be 2 (latched across runs).
        assert_eq!(shared_state.lock().unwrap().consecutive_failures, 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn auto_compact_skipped_when_target_below_minimum() {
        // 3 messages total → EarliestHalf target = 1 < MIN_PARTIAL_COMPACT_TARGET.
        // Provider scripts only the assistant turn; auto-compaction
        // must NOT consume an extra turn.
        let provider = Arc::new(MockProvider::with_turns(vec![vec![
            Event::TextDelta {
                delta: "small".into(),
            },
            Event::Result {
                data: ResultData {
                    stop_reason: Some("end_turn".into()),
                    ..Default::default()
                },
            },
        ]]));
        let store = preload_store(vec![user_text("u1"), assistant_text("a1")]);
        let qloop = compact_loop_builder(provider.clone(), store.clone()).build();
        let mut stream = qloop.run("trigger", AbortController::new()).await.unwrap();
        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            events.push(item.unwrap());
        }
        // Only the assistant turn was consumed (not a compaction turn).
        assert_eq!(provider.remaining_turns(), 0);
        // No compact notice fired.
        assert!(!events.iter().any(
            |e| matches!(e, Event::Notice { code, .. } if code.starts_with("agent.compact."))
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn auto_compact_fires_at_minimum_target_boundary() {
        // 4 messages total → EarliestHalf target = 2 == MIN_PARTIAL_COMPACT_TARGET
        // (the equality boundary). Should fire, not skip.
        let provider = Arc::new(MockProvider::with_turns(vec![
            vec![
                Event::TextDelta {
                    delta: compact_response().into(),
                },
                Event::Result {
                    data: Default::default(),
                },
            ],
            vec![
                Event::TextDelta { delta: "ok".into() },
                Event::Result {
                    data: ResultData {
                        stop_reason: Some("end_turn".into()),
                        ..Default::default()
                    },
                },
            ],
        ]));
        // Preload 3 → run() pushes 1 → snapshot len = 4, mid = 2 == MIN.
        let store = preload_store(vec![user_text("u1"), assistant_text("a1"), user_text("u2")]);
        let qloop = compact_loop_builder(provider.clone(), store.clone()).build();
        let mut stream = qloop.run("trigger", AbortController::new()).await.unwrap();
        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            events.push(item.unwrap());
        }
        // Both turns consumed (compact + assistant).
        assert_eq!(provider.remaining_turns(), 0);
        // agent.compact.ok notice fired.
        assert!(events
            .iter()
            .any(|e| matches!(e, Event::Notice { code, .. } if code == "agent.compact.ok")));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn empty_response_is_retried_once_then_succeeds() {
        // DeepSeek occasionally emits a bare `stop` with no content, no
        // thinking, no tool call. The loop must retry once (a second
        // sample usually answers) instead of silently ending the turn.
        let provider = Arc::new(MockProvider::with_turns(vec![
            vec![Event::Result {
                data: ResultData {
                    stop_reason: Some("stop".into()),
                    ..Default::default()
                },
            }],
            vec![
                Event::TextDelta { delta: "ok".into() },
                Event::Result {
                    data: ResultData {
                        stop_reason: Some("end_turn".into()),
                        ..Default::default()
                    },
                },
            ],
        ]));
        let store = preload_store(vec![]);
        let qloop = QueryLoop::builder(provider.clone(), "mock")
            .permissions(Arc::new(
                PermissionManager::new().with_mode(PermissionMode::Bypass),
            ))
            .store(store.clone())
            .build();
        let mut stream = qloop.run("hi", AbortController::new()).await.unwrap();
        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            events.push(item.unwrap());
        }
        // Both turns consumed: the empty one and the successful retry.
        assert_eq!(provider.remaining_turns(), 0);
        assert!(events
            .iter()
            .any(|e| matches!(e, Event::Notice { code, .. } if code == "agent.empty_response")));
        assert!(events
            .iter()
            .any(|e| matches!(e, Event::TextDelta { delta } if delta == "ok")));
        // The empty first turn produced no assistant message in the store.
        let snap: Vec<_> = store.lock().unwrap().iter().cloned().collect();
        let assistant_texts: usize = snap
            .iter()
            .filter(|m| matches!(m, Message::Assistant { content, .. } if !content.is_empty()))
            .count();
        assert_eq!(assistant_texts, 1, "only the retried answer is stored");
    }

    #[tokio::test]
    async fn max_iterations_terminates_loop() {
        // Provider scripts an infinite tool_use loop (every turn requests
        // a tool). max_iterations cap should halt the loop with an
        // error.
        let mut turns = Vec::new();
        for i in 0..32 {
            turns.push(vec![
                Event::ToolUse {
                    id: format!("tu_{i}"),
                    name: "echo".into(),
                    input: serde_json::json!({"i": i}),
                },
                Event::Result {
                    data: ResultData {
                        stop_reason: Some("tool_use".into()),
                        ..Default::default()
                    },
                },
            ]);
        }
        let provider = Arc::new(MockProvider::with_turns(turns));
        let calls = Arc::new(AtomicUsize::new(0));
        let registry = echo_registry("echo", calls.clone());
        let perms = Arc::new(PermissionManager::new().allow(RuleSource::User, "echo"));

        let qloop = QueryLoop::builder(provider, "mock")
            .tools(registry)
            .permissions(perms)
            .max_iterations(3)
            .build();

        let mut stream = qloop.run("hi", AbortController::new()).await.unwrap();
        let mut got_max_err = false;
        while let Some(item) = stream.next().await {
            if let Err(AgentError::Other(msg)) = item {
                if msg.contains("max_iterations") {
                    got_max_err = true;
                }
            }
        }
        assert!(got_max_err, "expected max_iterations error");
    }

    fn user_tool_result(tu_id: &str, payload: &str) -> Message {
        Message::User {
            header: Header::new(),
            content: vec![ContentBlock::ToolResult {
                tool_use_id: tu_id.into(),
                content: agent_tool_result_text(payload),
                is_error: false,
            }],
        }
    }

    fn assistant_with_tool_use(tu_id: &str) -> Message {
        Message::Assistant {
            header: Header::new(),
            content: vec![ContentBlock::ToolUse {
                id: tu_id.into(),
                name: "Bash".into(),
                input: serde_json::json!({"command": "ls"}),
            }],
        }
    }

    #[test]
    fn sanitize_downgrades_orphan_tool_result_keeps_content() {
        // A store damaged by an OLDER compaction severed the pair: the
        // tool_use half is gone, the result half survives. Without the
        // defensive pass the provider 400s every request; the pass must
        // downgrade the orphan to text (content preserved).
        let out = sanitize_request_messages(vec![user_tool_result("t1", "payload here")]);
        let Message::User { content, .. } = &out[0] else {
            panic!("expected user message");
        };
        let ContentBlock::Text { text } = &content[0] else {
            panic!("expected downgraded text, got {:?}", content[0]);
        };
        assert!(text.contains("payload here"), "{text}");
    }

    #[test]
    fn sanitize_downgrades_mid_transcript_dangling_tool_use() {
        // tool_use without any result, followed by more conversation — not
        // the legal trailing shape, so it must become text.
        let out = sanitize_request_messages(vec![
            user_text("go"),
            assistant_with_tool_use("t1"),
            user_text("next"),
        ]);
        let Message::Assistant { content, .. } = &out[1] else {
            panic!("expected assistant message");
        };
        let ContentBlock::Text { text } = &content[0] else {
            panic!("expected downgraded text, got {:?}", content[0]);
        };
        assert!(text.contains("tool call"), "{text}");
    }

    #[test]
    fn sanitize_keeps_trailing_tool_use_and_paired_sequences() {
        // A trailing assistant tool_use is a legal "awaiting result" shape;
        // a fully paired sequence must pass through untouched.
        let trailing =
            sanitize_request_messages(vec![user_text("go"), assistant_with_tool_use("t1")]);
        assert!(matches!(
            &trailing[1],
            Message::Assistant { content, .. } if matches!(content[0], ContentBlock::ToolUse { .. })
        ));
        let paired = vec![
            user_text("go"),
            assistant_with_tool_use("t1"),
            user_tool_result("t1", "ok"),
            assistant_text("done"),
        ];
        let out = sanitize_request_messages(paired.clone());
        assert!(matches!(
            &out[1],
            Message::Assistant { content, .. } if matches!(content[0], ContentBlock::ToolUse { .. })
        ));
        assert!(matches!(
            &out[2],
            Message::User { content, .. } if matches!(content[0], ContentBlock::ToolResult { .. })
        ));
    }

    #[test]
    fn sanitize_ignores_tombstoned_halves() {
        // The severed halves of an old pair: the tool_use is tombstoned and
        // the result is live — Tombstones carry no tool_use, so the result
        // is an orphan and must be downgraded (providers skip Tombstones).
        let tombstone = Message::Tombstone {
            header: Header::new(),
            reason: "compacted".into(),
        };
        let out = sanitize_request_messages(vec![tombstone, user_tool_result("t1", "data")]);
        let Message::User { content, .. } = &out[1] else {
            panic!("expected user message");
        };
        assert!(matches!(content[0], ContentBlock::Text { .. }));
    }

    fn agent_tool_result_text(payload: &str) -> crate::message::ToolResultContent {
        crate::message::ToolResultContent::Text(payload.into())
    }

    /// Whether a tool-result payload (text or blocks) contains `needle`.
    /// Pruned payloads keep the head/tail around the marker, so tests
    /// assert containment instead of whole-payload equality.
    fn tool_result_text_contains(
        content: &crate::message::ToolResultContent,
        needle: &str,
    ) -> bool {
        match content {
            crate::message::ToolResultContent::Text(t) => t.contains(needle),
            crate::message::ToolResultContent::Blocks(bs) => bs.iter().any(|b| match b {
                ContentBlock::Text { text } => text.contains(needle),
                _ => false,
            }),
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn microcompact_ladder_skips_llm_compaction_when_enough() {
        // One huge OLD tool result (~100K tokens) three human turns back
        // + 5 small current-turn ones. model_max_tokens = 50_000 → auto
        // threshold = 17_000. Pre-micro total ≈ 100K ≥ 17K → fires;
        // clearing tu_0 alone drops the total below 17K → the LLM
        // compaction turn must NOT run. (Ages count human-anchored turns
        // only, so tu_0 needs real prompts after it to be old enough.)
        let provider = Arc::new(MockProvider::with_turns(vec![vec![
            Event::TextDelta { delta: "ok".into() },
            Event::Result {
                data: ResultData {
                    stop_reason: Some("end_turn".into()),
                    ..Default::default()
                },
            },
        ]]));
        let mut preload = vec![
            user_tool_result("tu_0", &"x".repeat(400_000)),
            user_text("earlier prompt one"),
            user_text("earlier prompt two"),
            user_text("earlier prompt three"),
        ];
        for i in 1..6 {
            preload.push(user_tool_result(&format!("tu_{i}"), &"y".repeat(600)));
        }
        let store = preload_store(preload);
        let qloop = QueryLoop::builder(provider.clone(), "mock")
            .permissions(Arc::new(
                PermissionManager::new().with_mode(PermissionMode::Bypass),
            ))
            .store(store.clone())
            .model_max_tokens(50_000)
            .microcompact(true)
            .build();
        let mut stream = qloop.run("trigger", AbortController::new()).await.unwrap();
        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            events.push(item.unwrap());
        }
        // Only the assistant turn was consumed — no LLM compaction.
        assert_eq!(provider.remaining_turns(), 0);
        assert!(events
            .iter()
            .any(|e| matches!(e, Event::Notice { code, .. } if code == "agent.compact.micro")));
        assert!(!events
            .iter()
            .any(|e| matches!(e, Event::Notice { code, .. } if code == "agent.compact.ok")));
        // tu_0's payload was pruned in the store (head + marker + tail);
        // no tombstones exist.
        let snap: Vec<_> = store.lock().unwrap().iter().cloned().collect();
        assert!(!snap.iter().any(|m| matches!(m, Message::Tombstone { .. })));
        match &snap[0] {
            Message::User { content, .. } => match &content[0] {
                ContentBlock::ToolResult { content, .. } => {
                    assert!(
                        tool_result_text_contains(content, crate::compact::CLEARED_PLACEHOLDER),
                        "pruned payload must carry the marker"
                    );
                }
                other => panic!("expected tool result, got {other:?}"),
            },
            other => panic!("expected user message, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn microcompact_ladder_spares_protected_tool_results() {
        // AskUserQuestion results carry the user's ACTUAL input; the
        // auto-compact ladder must never clear them even when old and
        // huge. One 400K-char protected result, three human turns back,
        // followed by 5 recent results: unprotected it is the only
        // eligible target; protected it must survive untouched.
        let provider = Arc::new(MockProvider::with_turns(vec![
            vec![
                Event::TextDelta {
                    delta: compact_response().into(),
                },
                Event::Result {
                    data: Default::default(),
                },
            ],
            vec![
                Event::TextDelta { delta: "ok".into() },
                Event::Result {
                    data: ResultData {
                        stop_reason: Some("end_turn".into()),
                        ..Default::default()
                    },
                },
            ],
        ]));
        let mut preload = vec![
            Message::Assistant {
                header: Header::new(),
                content: vec![ContentBlock::ToolUse {
                    id: "tu_ask".into(),
                    name: "AskUserQuestion".into(),
                    input: serde_json::json!({"questions": []}),
                }],
            },
            user_tool_result("tu_ask", &"x".repeat(400_000)),
            user_text("earlier prompt one"),
            user_text("earlier prompt two"),
            user_text("earlier prompt three"),
        ];
        for i in 1..6 {
            preload.push(user_tool_result(&format!("tu_{i}"), &"y".repeat(600)));
        }
        let store = preload_store(preload);
        let qloop = QueryLoop::builder(provider.clone(), "mock")
            .permissions(Arc::new(
                PermissionManager::new().with_mode(PermissionMode::Bypass),
            ))
            .store(store.clone())
            .model_max_tokens(50_000)
            .microcompact(true)
            .microcompact_protected_tools(vec!["AskUserQuestion".into()])
            .build();
        let mut stream = qloop.run("trigger", AbortController::new()).await.unwrap();
        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            events.push(item.unwrap());
        }
        // The only eligible (old, huge) result is protected, so the ladder
        // cleared NOTHING (no micro notice) and the LLM compaction took
        // over (compact.ok fired, both provider turns consumed). Without
        // the protection the ladder would clear tu_ask, drop below the
        // threshold, and skip the LLM compaction — the compact.ok notice
        // and the consumed second turn prove the protection reached the
        // ladder rather than the threshold never firing.
        assert!(!events
            .iter()
            .any(|e| matches!(e, Event::Notice { code, .. } if code == "agent.compact.micro")));
        assert!(events
            .iter()
            .any(|e| matches!(e, Event::Notice { code, .. } if code == "agent.compact.ok")));
        assert_eq!(provider.remaining_turns(), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn auto_compact_promotes_analysis_to_session_memory() {
        let tagged = "<analysis>\n\
            - DECISION: rebuild MessageStore on microcompact.\n\
            - REQUIREMENT: never include co-author lines in commits.\n\
            - OPEN QUESTION: split loop_.rs into submodules?\n\
            </analysis>\n\
            <summary>Working on the compact ladder; decisions recorded.</summary>";
        let provider = Arc::new(MockProvider::with_turns(vec![
            vec![
                Event::TextDelta {
                    delta: tagged.into(),
                },
                Event::Result {
                    data: Default::default(),
                },
            ],
            vec![
                Event::TextDelta { delta: "ok".into() },
                Event::Result {
                    data: ResultData {
                        stop_reason: Some("end_turn".into()),
                        ..Default::default()
                    },
                },
            ],
        ]));
        let store = preload_store(vec![
            user_text("u1"),
            assistant_text("a1"),
            user_text("u2"),
            assistant_text("a2"),
        ]);
        let mem = Arc::new(crate::compact::InMemoryStore::new());
        let qloop = compact_loop_builder(provider.clone(), store)
            .session_memory(mem.clone())
            .build();
        let mut stream = qloop.run("trigger", AbortController::new()).await.unwrap();
        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            events.push(item.unwrap());
        }
        assert!(events
            .iter()
            .any(|e| matches!(e, Event::Notice { code, .. } if code == "agent.compact.memory")));
        let entries = mem.list().await.unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].kind, crate::compact::SessionMemoryKind::Decision);
        assert_eq!(
            entries[1].kind,
            crate::compact::SessionMemoryKind::Requirement
        );
        assert_eq!(
            entries[2].kind,
            crate::compact::SessionMemoryKind::OpenQuestion
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn compact_instructions_reach_the_summarization_prompt() {
        // Captures every StreamRequest's system prompt.
        #[derive(Debug)]
        struct CapturingProvider {
            systems: Arc<std::sync::Mutex<Vec<Option<String>>>>,
            turns: std::sync::Mutex<Vec<Vec<Event>>>,
        }
        #[async_trait]
        impl Provider for CapturingProvider {
            fn id(&self) -> &str {
                "capturing"
            }
            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities::default()
            }
            async fn stream(
                &self,
                req: StreamRequest,
                _abort: AbortController,
            ) -> Result<Box<dyn EventStream>, AgentError> {
                self.systems.lock().unwrap().push(req.system.clone());
                let turn = self.turns.lock().unwrap().remove(0);
                Ok(Box::new(futures::stream::iter(turn.into_iter().map(Ok))))
            }
        }
        let systems = Arc::new(std::sync::Mutex::new(Vec::new()));
        let provider = Arc::new(CapturingProvider {
            systems: systems.clone(),
            turns: std::sync::Mutex::new(vec![
                vec![
                    Event::TextDelta {
                        delta: "<analysis>- did X</analysis><summary>Did X.</summary>".into(),
                    },
                    Event::Result {
                        data: Default::default(),
                    },
                ],
                vec![
                    Event::TextDelta { delta: "ok".into() },
                    Event::Result {
                        data: ResultData {
                            stop_reason: Some("end_turn".into()),
                            ..Default::default()
                        },
                    },
                ],
            ]),
        });
        let store = preload_store(vec![
            user_text("u1"),
            assistant_text("a1"),
            user_text("u2"),
            assistant_text("a2"),
        ]);
        let qloop = compact_loop_builder(provider, store)
            .compact_instructions("ALWAYS TAG REQUIREMENT BULLETS")
            .build();
        let mut stream = qloop.run("trigger", AbortController::new()).await.unwrap();
        while let Some(item) = stream.next().await {
            item.unwrap();
        }
        // First request is the compaction call; its system prompt must
        // carry the custom instructions appended to the vendor prompt.
        let captured = systems.lock().unwrap();
        let first = captured[0].as_deref().expect("compact call has a system");
        assert!(first.contains("ALWAYS TAG REQUIREMENT BULLETS"));
    }

    #[tokio::test]
    async fn driver_stream_surfaces_driver_panics() {
        let (tx, rx) = mpsc::unbounded();
        let driver = tokio::spawn(async move {
            drop(tx);
            panic!("driver panic probe");
        });
        let mut stream = DriverStream {
            rx,
            driver: Some(driver),
            abort: AbortController::new(),
            store: Arc::new(Mutex::new(MessageStore::new())),
            repair_from: 0,
            repair_fence: Arc::new(Mutex::new(())),
            quiescence: None,
            terminal_error: None,
            await_quiescence: true,
            completed: false,
        };

        let error = stream.next().await.unwrap().unwrap_err();
        assert!(error.to_string().contains("query driver task failed"));
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn driver_stream_holds_terminal_error_until_quiescent() {
        let abort = AbortController::new();
        let work = abort.activity().track_worker();
        let release = Arc::new(tokio::sync::Notify::new());
        let worker_release = release.clone();
        tokio::spawn(async move {
            worker_release.notified().await;
            drop(work);
        });
        let (tx, rx) = mpsc::unbounded();
        tx.unbounded_send(Err(AgentError::other("terminal probe")))
            .unwrap();
        drop(tx);
        let driver = tokio::spawn(async {});
        let mut stream = DriverStream {
            rx,
            driver: Some(driver),
            abort,
            store: Arc::new(Mutex::new(MessageStore::new())),
            repair_from: 0,
            repair_fence: Arc::new(Mutex::new(())),
            quiescence: None,
            terminal_error: None,
            await_quiescence: true,
            completed: false,
        };

        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(20), stream.next())
                .await
                .is_err(),
            "terminal error must wait for nested cleanup"
        );
        release.notify_one();
        let error = tokio::time::timeout(std::time::Duration::from_secs(1), stream.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap_err();
        assert!(error.to_string().contains("terminal probe"));
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn dropping_driver_stream_repairs_store_synchronously() {
        let store = Arc::new(Mutex::new(MessageStore::new()));
        store
            .lock()
            .unwrap()
            .push(Message::Assistant {
                header: Header::new(),
                content: vec![ContentBlock::ToolUse {
                    id: "hard-drop-tool".into(),
                    name: "slow".into(),
                    input: serde_json::json!({}),
                }],
            })
            .unwrap();
        let (tx, rx) = mpsc::unbounded();
        let driver = tokio::spawn(async move {
            let _tx = tx;
            futures::future::pending::<()>().await;
        });
        let stream = DriverStream {
            rx,
            driver: Some(driver),
            abort: AbortController::new(),
            store: store.clone(),
            repair_from: 0,
            repair_fence: Arc::new(Mutex::new(())),
            quiescence: None,
            terminal_error: None,
            await_quiescence: true,
            completed: false,
        };

        drop(stream);
        drop(PendingToolRepair::new(
            store.clone(),
            vec!["hard-drop-tool".into()],
        ));

        let repair_count = store
            .lock()
            .unwrap()
            .iter()
            .filter_map(|message| match message {
                Message::User { content, .. } => Some(content),
                _ => None,
            })
            .flatten()
            .filter(|block| {
                matches!(
                    block,
                    ContentBlock::ToolResult { tool_use_id, is_error: true, .. }
                        if tool_use_id == "hard-drop-tool"
                )
            })
            .count();
        assert_eq!(repair_count, 1, "hard-drop repair must be idempotent");
    }
}
