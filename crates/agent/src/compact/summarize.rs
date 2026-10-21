//! LLM-driven conversation summarization (claude-code parity, Tier 1).
//!
//! Calls the configured [`Provider`] with a special compaction system
//! prompt, collects the streamed text, parses
//! `<analysis>...</analysis><summary>...</summary>`, and builds a
//! [`CompactionResult`] that the caller can splice back into the
//! [`MessageStore`].

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use thiserror::Error;
use uuid::Uuid;

use super::{
    estimate_tokens,
    prompt::{
        parse_summary_response, summarization_prompt, ParseSummaryError, ParsedSummary,
        PartialCompactDirection, SUMMARIZE_NOW_NUDGE,
    },
};
use crate::abort::AbortController;
use crate::error::AgentError;
use crate::message::{ContentBlock, Header, Message, MessageStore, ToolResultContent};
use crate::provider::{Provider, StreamRequest};
use crate::stream::Event;

/// Sentinel text the boundary marker carries — same string as
/// [`super::insert_boundary_marker`] uses.
pub const COMPACT_BOUNDARY_TEXT: &str = "CONTEXT SUMMARY BELOW";

/// Reserved-tokens budget Claude Code keeps for the compaction
/// response itself (p99.99 of compact summary output is ~17K tokens).
pub const MAX_OUTPUT_TOKENS_FOR_SUMMARY: u32 = 20_000;

#[derive(Debug, Error)]
pub enum CompactError {
    #[error("compact: not enough messages to compact")]
    NotEnoughMessages,
    #[error("compact: provider error: {0}")]
    Provider(#[from] AgentError),
    #[error("compact: parse error: {0}")]
    Parse(#[from] ParseSummaryError),
    #[error("compact: aborted")]
    Aborted,
    #[error("compact: provider stream produced no text")]
    EmptyResponse,
}

/// Outcome of a successful [`compact_conversation`] call.
#[derive(Debug, Clone)]
pub struct CompactionResult {
    /// The boundary `Message::System` carrying [`COMPACT_BOUNDARY_TEXT`].
    pub boundary_message: Message,
    /// The `Message::System` carrying the LLM-produced summary.
    pub summary_message: Message,
    /// `<analysis>` block from the model — typically NOT pushed back
    /// into the message store; useful for diagnostics, telemetry,
    /// and the post-cleanup file-restoration step.
    pub analysis: String,
    /// Full `<summary>` text. Equal to `summary_message` text.
    pub summary: String,
    /// UUIDs of the messages this compaction is meant to replace. The
    /// caller decides whether to delete + tombstone them, or simply
    /// surface the summary alongside.
    pub replaced_uuids: Vec<Uuid>,
    /// Estimated token total of the REPLACED messages before
    /// compaction (the whole transcript for `Full`, the replaced
    /// slice for partial directions).
    pub pre_compact_tokens: u32,
    /// Estimated token total of `(boundary + summary)` AFTER.
    pub post_compact_tokens: u32,
    /// Direction of compaction performed.
    pub direction: PartialCompactDirection,
}

/// Index splitting `messages` so `[..idx]` holds roughly half of the
/// estimated token mass, clamped to `1..=len-1` so both sides stay
/// non-empty. Halving by message COUNT is wrong when the mass is
/// uneven: a head full of tiny tombstones from a previous compaction
/// (or one huge tool dump late in the transcript) makes a count-based
/// "half" reclaim almost nothing, and occupancy-driven callers then
/// re-fire compaction forever without the context ever shrinking.
/// Callers must ensure `messages.len() >= 2`.
fn token_split_index(messages: &[Message]) -> usize {
    let total: u64 = messages.iter().map(|m| u64::from(estimate_tokens(m))).sum();
    let half = total.div_ceil(2);
    let mut acc = 0u64;
    for (i, msg) in messages.iter().enumerate() {
        acc = acc.saturating_add(u64::from(estimate_tokens(msg)));
        if acc >= half {
            return (i + 1).clamp(1, messages.len() - 1);
        }
    }
    // Unreachable (acc == total >= half at the last message); keep the
    // count-based midpoint as a safe fallback.
    (messages.len() / 2).max(1)
}

/// Snap a raw token-midpoint split to a tool-pair-safe boundary before it
/// becomes the STORE-mutating replaced set.
///
/// `repair_request_slice` protects only the request-only summarize slice;
/// the tombstone set used to be the raw midpoint, so a split landing inside
/// an `assistant tool_use → user tool_result` pair tombstoned one half and
/// kept the other. Every later provider render then contained an orphaned
/// `role:"tool"` message (OpenAI dialects) or a dangling `tool_use`
/// (Anthropic) and the provider rejected EVERY subsequent request with a
/// 400 — the session was permanently wedged "right after compaction".
///
/// A cut is legal iff NO unanswered tool_use crosses it. That balance is
/// computed in one incremental sweep (DSH-style `inProgressToolCalls`):
/// each assistant `tool_use` raises it, each user `tool_result` lowers it
/// (clamped at zero so an orphaned result — damage from an even older
/// compaction — cannot poison the sweep), and a cut whose balance is zero
/// never severs a pair, adjacent or not.
///
/// Snap in the direction that keeps compaction PROGRESS (the whole pair
/// joins the replaced set): forward past the pair for `EarliestHalf`
/// (replaced = prefix), backward to the pair start for `LatestHalf`
/// (replaced = suffix). When that direction is impossible, fall back to
/// the other side — losing a little progress is fine; splitting the pair
/// never is. A transcript with no severable boundary at all degrades to
/// `0` (EarliestHalf) or `len` (LatestHalf), i.e. an empty replaced set —
/// no progress, but never a severed pair. Both `0` and `len` are legal:
/// callers must tolerate an empty replaced set (the boundary/summary
/// messages still get spliced in).
fn pair_safe_split(
    messages: &[Message],
    target: usize,
    direction: PartialCompactDirection,
) -> usize {
    let len = messages.len();
    if len < 2 {
        return target;
    }
    // balance[i] = unanswered tool_use count BEFORE position i (the cut
    // between message i-1 and i). balance[0] is trivially 0.
    let mut balance = vec![0i32; len + 1];
    let mut acc = 0i32;
    for (i, m) in messages.iter().enumerate() {
        balance[i] = acc;
        match m {
            Message::Assistant { content, .. } => {
                let uses = content
                    .iter()
                    .filter(|b| matches!(b, ContentBlock::ToolUse { .. }))
                    .count() as i32;
                acc = acc.saturating_add(uses);
            }
            Message::User { content, .. } => {
                let results = content
                    .iter()
                    .filter(|b| matches!(b, ContentBlock::ToolResult { .. }))
                    .count() as i32;
                // Clamp at zero: an orphaned result (damage from an even
                // older compaction) must not make every later cut look
                // "unbalanced" — it is itself a broken pair half, not a
                // barrier.
                acc = acc.saturating_sub(results).max(0);
            }
            _ => {}
        }
    }
    balance[len] = acc;

    let legal = |i: usize| balance[i] == 0;
    // Direction of progress: EarliestHalf prefers cuts AFTER the target
    // (more of the prefix joins the replaced set), LatestHalf BEFORE it.
    match direction {
        PartialCompactDirection::EarliestHalf => {
            for i in target.min(len - 1)..len {
                if legal(i) {
                    return i;
                }
            }
            // No legal cut at or after the target: back off toward the
            // transcript head (0 = empty replaced set, always legal).
            for i in (0..target.min(len)).rev() {
                if legal(i) {
                    return i;
                }
            }
            0
        }
        _ => {
            // LatestHalf: prefer cuts at/before the target.
            for i in (1..=target.min(len)).rev() {
                if legal(i) {
                    return i;
                }
            }
            // Fall back forward (len = empty replaced set, always legal).
            for i in (target + 1)..=len {
                if legal(i) {
                    return i;
                }
            }
            len
        }
    }
}

/// Make the request-only compact slice self-consistent for strict providers.
///
/// The token-midpoint split (and reactive compaction running MID-turn, when
/// the newest tool_use legitimately has no result yet) can produce a slice
/// where an assistant `tool_use` has no following `tool_result`, or where a
/// leading `tool_result` lost its `tool_use` to the other half. Gateways that
/// enforce OpenAI-style pairing reject either shape with a 400 ("tool_calls
/// must be followed by tool messages"), which made auto-compaction fail
/// repeatedly and trip its circuit breaker. Repair both shapes here:
///
/// - a `tool_use` with no result in the slice gets a synthetic error
///   `tool_result` inserted directly after its assistant message;
/// - a `tool_result` with no `tool_use` in the slice is downgraded to a
///   plain-text block (its content still feeds the summary).
///
/// This never touches the store — the slice is request-only.
fn sanitize_slice_for_request(messages: Vec<Message>) -> Vec<Message> {
    use std::collections::HashSet;
    let use_ids: HashSet<String> = messages
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
    let result_ids: HashSet<String> = messages
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

    let mut out = Vec::with_capacity(messages.len());
    for msg in messages {
        match msg {
            Message::User { header, content } => {
                let content = content
                    .into_iter()
                    .map(|b| match b {
                        ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            ..
                        } if !use_ids.contains(&tool_use_id) => {
                            let text = match content {
                                ToolResultContent::Text(t) => t,
                                ToolResultContent::Blocks(blocks) => blocks
                                    .iter()
                                    .filter_map(|b| match b {
                                        ContentBlock::Text { text } => Some(text.as_str()),
                                        _ => None,
                                    })
                                    .collect::<Vec<_>>()
                                    .join("\n"),
                            };
                            ContentBlock::Text {
                                text: format!("[tool result]\n{text}"),
                            }
                        }
                        other => other,
                    })
                    .collect();
                out.push(Message::User { header, content });
            }
            Message::Assistant { header, content } => {
                let missing: Vec<String> = content
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::ToolUse { id, .. } if !result_ids.contains(id) => {
                            Some(id.clone())
                        }
                        _ => None,
                    })
                    .collect();
                out.push(Message::Assistant { header, content });
                if !missing.is_empty() {
                    out.push(Message::User {
                        header: Header::new(),
                        content: missing
                            .into_iter()
                            .map(|id| ContentBlock::ToolResult {
                                tool_use_id: id,
                                content: ToolResultContent::Text(
                                    "[result not available in this slice]".into(),
                                ),
                                is_error: true,
                            })
                            .collect(),
                    });
                }
            }
            other => out.push(other),
        }
    }
    out
}

/// Compact `messages` into a single summary System message.
///
/// **Behavior**:
/// 1. Validate at least 2 messages exist (single message is not
///    compactable).
/// 2. Compute pre_compact tokens via [`estimate_tokens`] over the
///    slice the chosen direction replaces.
/// 3. Build a [`StreamRequest`] with `messages` as the body and
///    [`summarization_prompt`] as the system. `max_output_tokens` is
///    capped by [`MAX_OUTPUT_TOKENS_FOR_SUMMARY`].
/// 4. Call `provider.stream` and accumulate every `TextDelta` into a
///    single response string. Aborted streams short-circuit to
///    [`CompactError::Aborted`]. Stream-level provider errors bubble
///    up as [`CompactError::Provider`].
/// 5. Parse the response via
///    [`super::prompt::parse_summary_response`].
/// 6. Build the [`CompactionResult`] with boundary +  summary
///    Message::System nodes and the parsed analysis text.
///
/// The caller is responsible for splicing the result back into a
/// [`MessageStore`] (e.g., via [`apply_compaction_to_store`]).
pub async fn compact_conversation(
    messages: &[Message],
    provider: &dyn Provider,
    model: impl Into<String>,
    custom_instructions: Option<&str>,
    direction: PartialCompactDirection,
    abort: AbortController,
) -> Result<CompactionResult, CompactError> {
    if messages.len() < 2 {
        return Err(CompactError::NotEnoughMessages);
    }

    let system = summarization_prompt(direction, custom_instructions);

    // Slice into compact-target depending on direction. For Full, send
    // every message. For the partial directions, split at the TOKEN
    // midpoint (not the message-count midpoint): EarliestHalf compacts
    // the prefix holding ~half the estimated mass, LatestHalf the
    // matching suffix.
    let (target_slice, replaced_uuids): (Vec<Message>, Vec<Uuid>) = match direction {
        PartialCompactDirection::Full => (
            messages.to_vec(),
            messages.iter().map(|m| m.uuid()).collect(),
        ),
        PartialCompactDirection::EarliestHalf => {
            let mid = pair_safe_split(messages, token_split_index(messages), direction);
            (
                messages[..mid].to_vec(),
                messages[..mid].iter().map(|m| m.uuid()).collect(),
            )
        }
        PartialCompactDirection::LatestHalf => {
            let mid = pair_safe_split(messages, token_split_index(messages), direction);
            (
                messages[mid..].to_vec(),
                messages[mid..].iter().map(|m| m.uuid()).collect(),
            )
        }
    };

    // The "before" figure describes what this compaction replaces (the
    // target slice), so partial compactions report an honest reduction
    // instead of the whole-transcript total.
    let pre_compact_tokens: u32 = target_slice
        .iter()
        .map(estimate_tokens)
        .fold(0u32, u32::saturating_add);

    // The transcript usually ENDS mid-task (tool call → tool result), which
    // invites the model to continue that task instead of obeying the system
    // prompt — DeepSeek was seen emitting raw DSML tool markup as "summary"
    // text this way. A closing synthetic user turn re-anchors the request on
    // the summarize instruction. It is request-only: never stored, and not
    // part of `replaced_uuids`.
    let mut request_messages = sanitize_slice_for_request(target_slice);
    request_messages.push(Message::User {
        header: Header::new(),
        content: vec![ContentBlock::Text {
            text: SUMMARIZE_NOW_NUDGE.into(),
        }],
    });
    let model: String = model.into();

    // A failed compaction is recoverable (the transcript is untouched); a
    // garbage summary spliced into the store is not. So parse strictly and
    // give the model ONE retry when it ignored the response format (parse
    // errors are stochastic — a second sample usually complies). Provider,
    // stream, and empty-response failures are NOT retried: those are
    // transport-level and belong to the provider layer / auto-compact
    // failure handling.
    let mut last_err = CompactError::EmptyResponse;
    let mut parsed: Option<ParsedSummary> = None;
    for _attempt in 0..2 {
        let req = StreamRequest::new(model.clone(), request_messages.clone())
            .with_system(system.clone())
            .with_max_output_tokens(MAX_OUTPUT_TOKENS_FOR_SUMMARY);

        let mut stream = provider
            .stream(req, abort.clone())
            .await
            .map_err(CompactError::Provider)?;

        let collected = Arc::new(Mutex::new(String::new()));
        let mut emitted_error: Option<AgentError> = None;
        while let Some(item) = stream.next().await {
            if abort.is_aborted() {
                return Err(CompactError::Aborted);
            }
            match item {
                Ok(Event::TextDelta { delta }) => {
                    if let Ok(mut buf) = collected.lock() {
                        buf.push_str(&delta);
                    }
                }
                Ok(Event::Result { .. }) => {
                    // Stream completed; loop will end at next None.
                }
                Ok(Event::Error { code, message }) => {
                    emitted_error = Some(AgentError::provider(
                        "compact",
                        format!("stream error code={code}: {message}"),
                    ));
                    break;
                }
                Ok(_) => {
                    // Other events (Usage, Thinking, ToolUse) — ignore for
                    // compaction. NO_TOOLS_PREAMBLE should prevent
                    // ToolUse; if one slips through we drop it.
                }
                Err(e) => {
                    emitted_error = Some(e);
                    break;
                }
            }
        }

        if let Some(err) = emitted_error {
            return Err(CompactError::Provider(err));
        }

        let response_text = collected.lock().map(|b| b.clone()).unwrap_or_default();
        if response_text.trim().is_empty() {
            return Err(CompactError::EmptyResponse);
        }

        match parse_summary_response(&response_text) {
            Ok(p) => {
                parsed = Some(p);
                break;
            }
            Err(e) => {
                last_err = e.into();
                continue;
            }
        }
    }

    let Some(ParsedSummary { analysis, summary }) = parsed else {
        return Err(last_err);
    };

    let boundary_message = Message::System {
        header: Header::new(),
        text: COMPACT_BOUNDARY_TEXT.into(),
    };
    // Render the summary as a User message so providers that drop
    // System messages from the body (notably Anthropic, which renders
    // System content only via the request-level `system` parameter)
    // still see the compaction summary in the conversation. Prefixed
    // with a marker so consumers and the model can recognize it as
    // synthesized context rather than literal user input.
    let summary_message = Message::User {
        header: Header::new(),
        content: vec![ContentBlock::Text {
            text: format!("[Context summary]\n{summary}"),
        }],
    };
    let post_compact_tokens =
        estimate_tokens(&boundary_message).saturating_add(estimate_tokens(&summary_message));

    Ok(CompactionResult {
        boundary_message,
        summary_message,
        analysis,
        summary,
        replaced_uuids,
        pre_compact_tokens,
        post_compact_tokens,
        direction,
    })
}

/// Apply a [`CompactionResult`] to a [`MessageStore`]: tombstone every
/// `replaced_uuids` (so child messages still resolve through the DAG)
/// and append boundary + summary system messages at the end.
///
/// Tombstones are non-destructive — the original UUIDs are preserved
/// so `parent_uuid` references in unrelated branches remain valid.
/// Future renderers / replay tools can recognize tombstoned messages
/// via [`Message::Tombstone`].
pub fn apply_compaction_to_store(
    store: &mut MessageStore,
    result: &CompactionResult,
) -> Result<(), AgentError> {
    // Snapshot existing messages, tombstone replaced UUIDs in place,
    // and insert boundary + summary AT the compaction boundary so the
    // chronological order makes sense to the model:
    //
    // - Full          → append at end (everything is replaced).
    // - EarliestHalf  → insert AFTER the last replaced message
    //                   (i.e., right BEFORE the first preserved).
    // - LatestHalf    → insert BEFORE the first replaced message
    //                   (i.e., right AFTER the last preserved).
    //
    // This keeps the synthetic summary chronologically next to the
    // window it summarizes, instead of sitting at the end and
    // appearing to come after the user's most recent input.
    let snapshot: Vec<Message> = store.iter().cloned().collect();
    let total = snapshot.len();
    let insert_at: usize = match result.direction {
        PartialCompactDirection::Full => total,
        PartialCompactDirection::EarliestHalf => snapshot
            .iter()
            .position(|m| !result.replaced_uuids.contains(&m.uuid()))
            .unwrap_or(total),
        PartialCompactDirection::LatestHalf => snapshot
            .iter()
            .position(|m| result.replaced_uuids.contains(&m.uuid()))
            .unwrap_or(total),
    };

    let mut new_store = MessageStore::new();
    for (i, msg) in snapshot.into_iter().enumerate() {
        if i == insert_at {
            new_store.push(result.boundary_message.clone())?;
            new_store.push(result.summary_message.clone())?;
        }
        let to_push = if result.replaced_uuids.contains(&msg.uuid()) {
            let header = msg.header().clone();
            Message::Tombstone {
                header,
                reason: "compacted".into(),
            }
        } else {
            msg
        };
        new_store.push(to_push)?;
    }
    if insert_at == total {
        new_store.push(result.boundary_message.clone())?;
        new_store.push(result.summary_message.clone())?;
    }
    *store = new_store;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex as StdMutex;

    use async_trait::async_trait;
    use futures::stream;

    use super::*;
    use crate::message::ContentBlock;
    use crate::provider::{ProviderCapabilities, StreamRequest};
    use crate::stream::EventStream;

    fn user(text: &str) -> Message {
        Message::User {
            header: Header::new(),
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }
    fn assistant(text: &str) -> Message {
        Message::Assistant {
            header: Header::new(),
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }
    fn assistant_tool_use(id: &str) -> Message {
        Message::Assistant {
            header: Header::new(),
            content: vec![ContentBlock::ToolUse {
                id: id.into(),
                name: "Bash".into(),
                input: serde_json::json!({"command": "ls"}),
            }],
        }
    }
    fn user_tool_result(id: &str) -> Message {
        Message::User {
            header: Header::new(),
            content: vec![ContentBlock::ToolResult {
                tool_use_id: id.into(),
                content: ToolResultContent::Text("ok".into()),
                is_error: false,
            }],
        }
    }

    /// The STORE-mutating replaced set must never sever an
    /// `assistant tool_use → user tool_result` pair: tombstoning one half
    /// leaves every later provider render with an orphaned tool message and
    /// the provider 400s ALL subsequent requests — the session wedges
    /// permanently right after compaction.
    #[test]
    fn pair_safe_split_never_severs_a_tool_pair() {
        // Pair spans indexes 1-2; force targets on both sides of it.
        let messages = vec![
            user("start"),
            assistant_tool_use("t1"),
            user_tool_result("t1"),
            assistant("answer"),
            user("next"),
        ];
        // Target 2 lands INSIDE the pair (after tool_use, before result).
        let earliest = pair_safe_split(&messages, 2, PartialCompactDirection::EarliestHalf);
        assert_eq!(earliest, 3, "EarliestHalf snaps forward past the pair");
        let latest = pair_safe_split(&messages, 2, PartialCompactDirection::LatestHalf);
        assert_eq!(latest, 1, "LatestHalf snaps back to the pair start");
        // Boundaries outside the pair pass through untouched.
        assert_eq!(
            pair_safe_split(&messages, 3, PartialCompactDirection::EarliestHalf),
            3
        );
        // A pair that ENDS the transcript: EarliestHalf cannot go forward,
        // so it backs off instead of splitting.
        let tail_pair = vec![
            user("start"),
            assistant("mid"),
            assistant_tool_use("t2"),
            user_tool_result("t2"),
        ];
        assert_eq!(
            pair_safe_split(&tail_pair, 3, PartialCompactDirection::EarliestHalf),
            2,
            "no forward room — back off to the pair start"
        );
    }

    #[test]
    fn sanitize_inserts_result_for_dangling_tool_use() {
        // Slice ends on an assistant tool_use whose result fell outside the
        // slice (or hasn't arrived yet) — a synthetic error result must follow
        // it immediately, or strict gateways 400 the compact request.
        let out = sanitize_slice_for_request(vec![user("go"), assistant_tool_use("t1")]);
        assert_eq!(out.len(), 3);
        let Message::User { content, .. } = &out[2] else {
            panic!(
                "expected a synthetic tool_result user message, got {:?}",
                out[2]
            );
        };
        let ContentBlock::ToolResult {
            tool_use_id,
            is_error,
            ..
        } = &content[0]
        else {
            panic!("expected a tool_result block");
        };
        assert_eq!(tool_use_id, "t1");
        assert!(is_error);
    }

    #[test]
    fn sanitize_downgrades_orphan_tool_result_to_text() {
        // LatestHalf slice starting with a tool_result whose tool_use went to
        // the other half: the block becomes plain text (content preserved).
        let out = sanitize_slice_for_request(vec![user_tool_result("gone"), assistant("done")]);
        let Message::User { content, .. } = &out[0] else {
            panic!("expected user message");
        };
        let ContentBlock::Text { text } = &content[0] else {
            panic!("expected downgraded text block, got {:?}", content[0]);
        };
        assert!(text.contains("ok"), "content must survive: {text}");
    }

    #[test]
    fn sanitize_keeps_paired_slice_unchanged() {
        let msgs = vec![
            user("go"),
            assistant_tool_use("t1"),
            user_tool_result("t1"),
            assistant("done"),
        ];
        let out = sanitize_slice_for_request(msgs.clone());
        assert_eq!(out.len(), msgs.len());
        assert!(matches!(&out[1], Message::Assistant { .. }));
        let Message::User { content, .. } = &out[2] else {
            panic!()
        };
        assert!(matches!(content[0], ContentBlock::ToolResult { .. }));
    }

    #[tokio::test]
    async fn compact_request_has_no_dangling_tool_use() {
        // End-to-end: a transcript ending mid tool call must reach the
        // provider with the dangling tool_use answered.
        let provider = ScriptedProvider::from_text(happy_response());
        let msgs = vec![user("go"), assistant_tool_use("t1")];
        let result = compact_conversation(
            &msgs,
            &provider,
            "m",
            None,
            PartialCompactDirection::Full,
            AbortController::default(),
        )
        .await;
        assert!(result.is_ok(), "{result:?}");
        let requests = provider.requests.lock().unwrap();
        let sent = &requests[0].messages;
        let use_pos = sent
            .iter()
            .position(|m| {
                matches!(m, Message::Assistant { content, .. }
                if content.iter().any(|b| matches!(b, ContentBlock::ToolUse { .. })))
            })
            .expect("tool_use present");
        let Message::User { content, .. } = &sent[use_pos + 1] else {
            panic!("tool_use must be followed by a user message");
        };
        assert!(
            content.iter().any(
                |b| matches!(b, ContentBlock::ToolResult { tool_use_id, .. } if tool_use_id == "t1")
            ),
            "synthetic result must directly follow the tool_use"
        );
    }

    #[derive(Debug)]
    struct ScriptedProvider {
        /// One event script per expected `stream` call, popped front-first.
        /// A drained queue yields an empty stream (→ `EmptyResponse`).
        scripts: StdMutex<std::collections::VecDeque<Vec<Event>>>,
        /// Every request handed to `stream`, for request-shape assertions.
        requests: StdMutex<Vec<StreamRequest>>,
    }

    impl ScriptedProvider {
        fn scripted(scripts: Vec<Vec<Event>>) -> Self {
            Self {
                scripts: StdMutex::new(scripts.into()),
                requests: StdMutex::new(Vec::new()),
            }
        }
        fn from_text(text: &str) -> Self {
            Self::from_texts(&[text])
        }
        /// One text response per expected call.
        fn from_texts(texts: &[&str]) -> Self {
            Self::scripted(
                texts
                    .iter()
                    .map(|t| {
                        vec![
                            Event::TextDelta { delta: (*t).into() },
                            Event::Result {
                                data: Default::default(),
                            },
                        ]
                    })
                    .collect(),
            )
        }
        fn calls(&self) -> usize {
            self.requests.lock().unwrap().len()
        }
    }

    #[async_trait]
    impl Provider for ScriptedProvider {
        fn id(&self) -> &str {
            "scripted"
        }
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::default()
        }
        async fn stream(
            &self,
            req: StreamRequest,
            _abort: AbortController,
        ) -> Result<Box<dyn EventStream>, AgentError> {
            self.requests.lock().unwrap().push(req);
            let events: Vec<Event> = self
                .scripts
                .lock()
                .map(|mut q| q.pop_front().unwrap_or_default())
                .unwrap();
            Ok(Box::new(stream::iter(events.into_iter().map(Ok))))
        }
    }

    fn happy_response() -> &'static str {
        "<analysis>
- User asked X
- Assistant did Y
</analysis>
<summary>
The session was about X. Y happened.
</summary>"
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn compact_conversation_happy_path() {
        let provider = ScriptedProvider::from_text(happy_response());
        let messages = vec![user("hi"), assistant("hello"), user("ok bye")];
        let result = compact_conversation(
            &messages,
            &provider,
            "any-model",
            None,
            PartialCompactDirection::Full,
            AbortController::new(),
        )
        .await
        .unwrap();
        assert!(result.analysis.contains("User asked X"));
        assert!(result.summary.starts_with("The session was about X"));
        assert_eq!(result.replaced_uuids.len(), 3);
        assert!(result.pre_compact_tokens > 0);
        assert!(result.post_compact_tokens > 0);
        // Boundary text is the canonical CONTEXT SUMMARY BELOW.
        match &result.boundary_message {
            Message::System { text, .. } => assert_eq!(text, COMPACT_BOUNDARY_TEXT),
            _ => panic!("expected System boundary"),
        }
    }

    /// The DSML tool-call markup DeepSeek emitted as literal text when asked
    /// to summarize (regression sample from a real corrupted session).
    fn dsml_garbage() -> &'static str {
        "<｜｜DSML｜｜tool_calls>\n<｜｜DSML｜｜invoke name=\"Bash\">\
         \n</｜｜DSML｜｜invoke>\n</｜｜DSML｜｜tool_calls>"
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn summarize_request_ends_with_nudge_user_turn() {
        let provider = ScriptedProvider::from_text(happy_response());
        let messages = vec![user("hi"), assistant("hello")];
        compact_conversation(
            &messages,
            &provider,
            "any-model",
            None,
            PartialCompactDirection::Full,
            AbortController::new(),
        )
        .await
        .unwrap();
        let requests = provider.requests.lock().unwrap();
        let last = requests[0].messages.last().expect("request has messages");
        match last {
            Message::User { content, .. } => {
                assert!(matches!(
                    &content[0],
                    ContentBlock::Text { text } if text == SUMMARIZE_NOW_NUDGE
                ));
            }
            other => panic!("expected closing nudge User turn, got {other:?}"),
        }
        // The nudge is request-only: it must not count as a replaced message.
        assert_eq!(requests[0].messages.len(), messages.len() + 1);
    }

    fn tombstone() -> Message {
        Message::Tombstone {
            header: Header::new(),
            reason: "compacted".into(),
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn earliest_half_split_follows_token_mass() {
        // A transcript whose head is tiny (tombstones from a previous round
        // plus small chatter) and whose token mass sits in one large tool
        // dump. Halving by MESSAGE COUNT would replace only the tiny head,
        // reclaim ~nothing, and an occupancy-driven caller would re-fire
        // compaction forever — the split must follow token mass instead.
        let provider = ScriptedProvider::from_text(happy_response());
        let mut messages: Vec<Message> = (0..4).map(|_| tombstone()).collect();
        messages.push(user("small question"));
        messages.push(assistant(&"x".repeat(40_000))); // the mass, index 5
        messages.push(user("tail"));
        messages.push(assistant("tail answer"));
        let heavy_uuid = messages[5].uuid();
        let result = compact_conversation(
            &messages,
            &provider,
            "any-model",
            None,
            PartialCompactDirection::EarliestHalf,
            AbortController::new(),
        )
        .await
        .unwrap();
        assert!(
            result.replaced_uuids.contains(&heavy_uuid),
            "token-weighted split must reach the message holding the mass"
        );
        // The recent tail survives verbatim.
        assert!(!result.replaced_uuids.contains(&messages[6].uuid()));
        assert!(!result.replaced_uuids.contains(&messages[7].uuid()));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn earliest_half_never_replaces_the_final_message() {
        // All the mass in the LAST message: the token midpoint lands inside
        // it, but the final message must survive (the runtime relies on the
        // just-pushed user prompt being preserved), so the split clamps to
        // everything-but-the-last.
        let provider = ScriptedProvider::from_text(happy_response());
        let messages = vec![user("a"), user("b"), assistant(&"y".repeat(40_000))];
        let result = compact_conversation(
            &messages,
            &provider,
            "any-model",
            None,
            PartialCompactDirection::EarliestHalf,
            AbortController::new(),
        )
        .await
        .unwrap();
        assert_eq!(
            result.replaced_uuids,
            vec![messages[0].uuid(), messages[1].uuid()],
            "split clamps to len-1: replace the two tiny heads, keep the heavy tail"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn partial_pre_compact_tokens_count_only_the_replaced_slice() {
        // For partial directions the "before" figure must describe what the
        // compaction actually replaced, not the whole transcript — otherwise
        // the UI reports "~2000 → ~50 tokens" while the gauge barely moves.
        let provider = ScriptedProvider::from_text(happy_response());
        let messages = vec![
            assistant(&"z".repeat(4_000)),
            user("k"),
            assistant(&"z".repeat(4_000)),
        ];
        let result = compact_conversation(
            &messages,
            &provider,
            "any-model",
            None,
            PartialCompactDirection::EarliestHalf,
            AbortController::new(),
        )
        .await
        .unwrap();
        let slice_tokens: u32 = result
            .replaced_uuids
            .iter()
            .map(|u| {
                messages
                    .iter()
                    .find(|m| m.uuid() == *u)
                    .map(estimate_tokens)
                    .unwrap()
            })
            .sum();
        assert_eq!(
            result.pre_compact_tokens, slice_tokens,
            "pre_compact_tokens must cover the replaced slice only"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn compact_retries_once_past_tool_markup_response() {
        // First response is raw DSML garbage, second is well-formed: the
        // retry rescues the compaction instead of splicing garbage.
        let provider = ScriptedProvider::from_texts(&[dsml_garbage(), happy_response()]);
        let messages = vec![user("hi"), assistant("hello")];
        let result = compact_conversation(
            &messages,
            &provider,
            "any-model",
            None,
            PartialCompactDirection::Full,
            AbortController::new(),
        )
        .await
        .unwrap();
        assert!(result.summary.starts_with("The session was about X"));
        assert_eq!(provider.calls(), 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn compact_gives_up_after_two_tool_markup_responses() {
        // Both attempts return garbage: compaction must FAIL (transcript
        // untouched) rather than store tool markup as the summary.
        let provider = ScriptedProvider::from_texts(&[dsml_garbage(), dsml_garbage()]);
        let messages = vec![user("hi"), assistant("hello")];
        let err = compact_conversation(
            &messages,
            &provider,
            "any-model",
            None,
            PartialCompactDirection::Full,
            AbortController::new(),
        )
        .await
        .unwrap_err();
        assert!(matches!(
            err,
            CompactError::Parse(ParseSummaryError::RawToolMarkup)
        ));
        assert_eq!(provider.calls(), 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn compact_too_few_messages() {
        let provider = ScriptedProvider::from_text(happy_response());
        let messages = vec![user("only one")];
        match compact_conversation(
            &messages,
            &provider,
            "m",
            None,
            PartialCompactDirection::Full,
            AbortController::new(),
        )
        .await
        {
            Err(CompactError::NotEnoughMessages) => {}
            other => panic!("expected NotEnoughMessages, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn compact_empty_response_errors() {
        let provider = ScriptedProvider::from_text("");
        let messages = vec![user("a"), user("b")];
        match compact_conversation(
            &messages,
            &provider,
            "m",
            None,
            PartialCompactDirection::Full,
            AbortController::new(),
        )
        .await
        {
            Err(CompactError::EmptyResponse) => {}
            other => panic!("expected EmptyResponse, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn compact_tagless_response_is_summarized_leniently() {
        // A model that ignores the <analysis>/<summary> format must NOT abort
        // compaction — the plain text becomes the summary so context is still
        // reclaimed (regression guard for the deepseek `/compact` failure).
        let provider = ScriptedProvider::from_text("Just plain text, no tags here.");
        let messages = vec![user("a"), user("b")];
        let result = compact_conversation(
            &messages,
            &provider,
            "m",
            None,
            PartialCompactDirection::Full,
            AbortController::new(),
        )
        .await
        .expect("tagless response should compact leniently, not error");
        assert_eq!(result.summary, "Just plain text, no tags here.");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn compact_blank_response_still_errors() {
        // A truly empty response is a real failure (nothing to summarize).
        let provider = ScriptedProvider::from_text("   ");
        let messages = vec![user("a"), user("b")];
        assert!(matches!(
            compact_conversation(
                &messages,
                &provider,
                "m",
                None,
                PartialCompactDirection::Full,
                AbortController::new(),
            )
            .await,
            Err(CompactError::EmptyResponse)
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn compact_partial_earliest_half_keeps_recent_uuids_intact() {
        let provider = ScriptedProvider::from_text(happy_response());
        let m1 = user("first");
        let m2 = user("second");
        let m3 = user("third");
        let m4 = user("fourth");
        let messages = vec![m1.clone(), m2.clone(), m3.clone(), m4.clone()];
        let result = compact_conversation(
            &messages,
            &provider,
            "m",
            None,
            PartialCompactDirection::EarliestHalf,
            AbortController::new(),
        )
        .await
        .unwrap();
        // Earliest half = first 2; replaced_uuids should only include those.
        assert_eq!(result.replaced_uuids.len(), 2);
        assert!(result.replaced_uuids.contains(&m1.uuid()));
        assert!(result.replaced_uuids.contains(&m2.uuid()));
        assert!(!result.replaced_uuids.contains(&m3.uuid()));
        assert!(!result.replaced_uuids.contains(&m4.uuid()));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn apply_compaction_tombstones_replaced_messages() {
        let provider = ScriptedProvider::from_text(happy_response());
        let m1 = user("first");
        let m2 = assistant("second");
        let mut store = MessageStore::new();
        store.push(m1.clone()).unwrap();
        store.push(m2.clone()).unwrap();
        let messages: Vec<Message> = store.iter().cloned().collect();

        let result = compact_conversation(
            &messages,
            &provider,
            "m",
            None,
            PartialCompactDirection::Full,
            AbortController::new(),
        )
        .await
        .unwrap();

        apply_compaction_to_store(&mut store, &result).unwrap();

        // Originals tombstoned, boundary + summary appended.
        assert_eq!(store.len(), 4);
        let collected: Vec<&Message> = store.iter().collect();
        assert!(matches!(collected[0], Message::Tombstone { reason, .. } if reason == "compacted"));
        assert!(matches!(collected[1], Message::Tombstone { reason, .. } if reason == "compacted"));
        // UUIDs preserved on tombstones.
        assert_eq!(collected[0].uuid(), m1.uuid());
        assert_eq!(collected[1].uuid(), m2.uuid());
        // Boundary + summary at end. Boundary is System; summary is now
        // User (so providers that skip System still see it).
        match collected[2] {
            Message::System { text, .. } => assert_eq!(text, COMPACT_BOUNDARY_TEXT),
            _ => panic!("expected System boundary at index 2"),
        }
        match collected[3] {
            Message::User { content, .. } => {
                let text = content
                    .iter()
                    .find_map(|b| match b {
                        ContentBlock::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .expect("summary user msg has text block");
                assert!(text.starts_with("[Context summary]"));
                assert!(text.contains("The session"));
            }
            _ => panic!("expected User summary at index 3"),
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn apply_compaction_earliest_half_inserts_at_boundary() {
        // 4 equal-mass messages, EarliestHalf compacts the first 2 (the
        // token midpoint = the count midpoint here); boundary + summary
        // insert AFTER the tombstones, BEFORE the preserved tail.
        // Final layout: [Tomb, Tomb, System, User, m3, m4].
        let provider = ScriptedProvider::from_text(happy_response());
        let m1 = user(&"a".repeat(40));
        let m2 = assistant(&"b".repeat(40));
        let m3 = user(&"c".repeat(40));
        let m4 = assistant(&"d".repeat(40));
        let mut store = MessageStore::new();
        for m in [&m1, &m2, &m3, &m4] {
            store.push(m.clone()).unwrap();
        }
        let messages: Vec<Message> = store.iter().cloned().collect();
        let result = compact_conversation(
            &messages,
            &provider,
            "m",
            None,
            PartialCompactDirection::EarliestHalf,
            AbortController::new(),
        )
        .await
        .unwrap();
        apply_compaction_to_store(&mut store, &result).unwrap();

        let collected: Vec<&Message> = store.iter().collect();
        assert_eq!(collected.len(), 6);
        assert!(matches!(collected[0], Message::Tombstone { .. }));
        assert!(matches!(collected[1], Message::Tombstone { .. }));
        match collected[2] {
            Message::System { text, .. } => assert_eq!(text, COMPACT_BOUNDARY_TEXT),
            _ => panic!("expected boundary at idx 2"),
        }
        assert!(matches!(collected[3], Message::User { .. }));
        // Preserved m3, m4 remain at the tail with original UUIDs.
        assert_eq!(collected[4].uuid(), m3.uuid());
        assert_eq!(collected[5].uuid(), m4.uuid());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn apply_compaction_latest_half_inserts_before_replaced() {
        // 4 equal-mass messages, LatestHalf compacts the last 2 (the
        // token midpoint = the count midpoint here); boundary + summary
        // insert BEFORE the replaced tail, AFTER the preserved head.
        // Final layout: [m1, m2, System, User, Tomb, Tomb].
        let provider = ScriptedProvider::from_text(happy_response());
        let m1 = user(&"a".repeat(40));
        let m2 = assistant(&"b".repeat(40));
        let m3 = user(&"c".repeat(40));
        let m4 = assistant(&"d".repeat(40));
        let mut store = MessageStore::new();
        for m in [&m1, &m2, &m3, &m4] {
            store.push(m.clone()).unwrap();
        }
        let messages: Vec<Message> = store.iter().cloned().collect();
        let result = compact_conversation(
            &messages,
            &provider,
            "m",
            None,
            PartialCompactDirection::LatestHalf,
            AbortController::new(),
        )
        .await
        .unwrap();
        apply_compaction_to_store(&mut store, &result).unwrap();

        let collected: Vec<&Message> = store.iter().collect();
        assert_eq!(collected.len(), 6);
        assert_eq!(collected[0].uuid(), m1.uuid());
        assert_eq!(collected[1].uuid(), m2.uuid());
        match collected[2] {
            Message::System { text, .. } => assert_eq!(text, COMPACT_BOUNDARY_TEXT),
            _ => panic!("expected boundary at idx 2"),
        }
        assert!(matches!(collected[3], Message::User { .. }));
        assert!(matches!(collected[4], Message::Tombstone { .. }));
        assert!(matches!(collected[5], Message::Tombstone { .. }));
    }
}
