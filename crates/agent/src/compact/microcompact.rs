//! Micro-compaction: surgical token reduction without an LLM call
//! (claude-code parity, Tier 1).
//!
//! Strategy: replace OLD `ContentBlock::ToolResult` payloads with a
//! marker string. Preserves the message DAG, the assistant's
//! `tool_use` references, and the structural shape of the
//! conversation; surrenders only the bulky tool output bytes that
//! the model has likely already integrated into its assistant
//! response.
//!
//! Mirror of `services/compact/microCompact.ts`. Where Claude Code
//! has a more elaborate time-based config (see `timeBasedMCConfig.ts`),
//! we ship a simpler turn-age + size threshold combo. The gates are
//! tunable via [`MicrocompactConfig`].

use uuid::Uuid;

use super::estimate_tokens;
use crate::message::{ContentBlock, Message, ToolResultContent};

/// Replacement payload that goes into a cleared tool result. Mirrors
/// `TIME_BASED_MC_CLEARED_MESSAGE` from Claude Code.
pub const CLEARED_PLACEHOLDER: &str = "[Old tool result content cleared]";

/// Knobs controlling which tool results are eligible for clearing.
#[derive(Debug, Clone)]
pub struct MicrocompactConfig {
    /// Tool results that appeared **at least this many turns ago** are
    /// candidates. A "turn" is a HUMAN-anchored User message — one whose
    /// content carries any non-`ToolResult` block. Pure tool-result
    /// batches (the query loop feeds every batch back as a User message)
    /// do NOT advance the count: counting them made a result "3 turns
    /// old" after just three more tool calls, so a long agentic turn had
    /// everything but its last few results cleared, and the model
    /// re-ran the same commands to recover what it had just read.
    /// Default 3.
    pub min_age_turns: u32,
    /// Always preserve the most recent N tool results regardless of
    /// age. With human-anchored ages the current turn's results are
    /// already untouchable, so this only shields the tail of PREVIOUS
    /// turns. Default 5.
    pub preserve_last_n: u32,
    /// Only clear tool results whose estimated token cost exceeds
    /// this. Cheap results aren't worth the structural mutation.
    /// Default 100.
    pub min_tokens_per_result: u32,
    /// Tool NAMES whose results are never cleared, regardless of age or
    /// size. Interactive question tools (e.g. `AskUserQuestion`) return
    /// the user's ACTUAL input as their payload — clearing it makes the
    /// model believe it was never answered and re-ask the same question
    /// forever. The names are matched against the `tool_use` that
    /// produced each result (looked up through the assistant message),
    /// so the exemption survives renames of the tool id. Default empty.
    pub protected_tool_names: Vec<String>,
    /// Maximum leading characters retained when a text payload is
    /// cleared. Instead of replacing the WHOLE payload with the marker
    /// (which made weak models re-run the same commands to recover what
    /// they had just read), the head and the tail of the payload
    /// survive and only the middle is replaced. Default 1000.
    pub prune_head_chars: u32,
    /// Maximum trailing characters retained when a text payload is
    /// cleared (errors conventionally live at the END of a command's
    /// output, so the tail is what the model most needs). Default 500.
    pub prune_tail_chars: u32,
}

impl Default for MicrocompactConfig {
    fn default() -> Self {
        Self {
            min_age_turns: 3,
            preserve_last_n: 5,
            min_tokens_per_result: 100,
            protected_tool_names: Vec::new(),
            prune_head_chars: 1000,
            prune_tail_chars: 500,
        }
    }
}

/// Whether a message anchors a turn for age purposes: a User message
/// carrying anything beyond tool results (a prompt, an image). An
/// interrupted turn's folded message (tool_result + user text) counts —
/// the human spoke. Runtime-injected guard nudges do NOT count (see
/// [`is_injected_reminder`]).
fn is_human_turn(msg: &Message) -> bool {
    match msg {
        Message::User { content, .. } => content.iter().any(|b| match b {
            ContentBlock::ToolResult { .. } => false,
            ContentBlock::Text { text } => !is_injected_reminder(text),
            _ => true,
        }),
        _ => false,
    }
}

/// Loop-guard / command-repeat nudges are User messages by transport
/// necessity (provider renderers drop System entries), but nobody typed
/// them. A message that is nothing but one `<system-reminder>` block must
/// not advance tool-result ages — a few guard nudges inside one agentic
/// turn would otherwise re-open the mid-turn clearing hole the
/// human-anchored count exists to close. A REAL prompt that merely starts
/// with an injected reminder (hosts prepend them to the user's text) has
/// trailing content and still counts.
fn is_injected_reminder(text: &str) -> bool {
    let t = text.trim();
    t.starts_with("<system-reminder>") && t.ends_with("</system-reminder>")
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicrocompactResult {
    /// Number of tool result blocks whose content was replaced.
    pub cleared_count: usize,
    /// Estimated tokens freed (`pre - post`).
    pub tokens_freed: u32,
    /// Tool-use IDs whose results were cleared (useful for telemetry
    /// and post-cleanup file restoration).
    pub cleared_tool_use_ids: Vec<String>,
    /// UUIDs of messages whose blocks were modified.
    pub modified_message_uuids: Vec<Uuid>,
}

/// Apply microcompaction in-place over `messages`.
pub fn microcompact(messages: &mut [Message], config: &MicrocompactConfig) -> MicrocompactResult {
    // First, identify how many human-anchored turns separate each
    // ToolResult block from "now" (see [`is_human_turn`] — pure
    // tool-result batches don't advance the count, so results from the
    // CURRENT turn always have age 0 and are never cleared mid-turn).
    let total_turns: u32 = messages.iter().filter(|m| is_human_turn(m)).count() as u32;

    // Map every tool_use id to its tool NAME so protected tools can be
    // exempted by name (ids are per-call random; names are stable).
    let mut use_name: std::collections::HashMap<&str, &str> = std::collections::HashMap::new();
    for m in messages.iter() {
        if let Message::Assistant { content, .. } = m {
            for b in content {
                if let ContentBlock::ToolUse { id, name, .. } = b {
                    use_name.insert(id.as_str(), name.as_str());
                }
            }
        }
    }

    // Pass 1: gather (msg_index, block_index, age_turns, token_cost)
    // for every ToolResult block.
    let mut candidates: Vec<(usize, usize, u32, u32, String)> = Vec::new();
    let mut user_seen: u32 = 0;
    for (mi, msg) in messages.iter().enumerate() {
        if is_human_turn(msg) {
            user_seen = user_seen.saturating_add(1);
        }
        if let Message::User { content, .. } = msg {
            for (bi, block) in content.iter().enumerate() {
                if let ContentBlock::ToolResult {
                    tool_use_id,
                    content: result_content,
                    ..
                } = block
                {
                    // Interactive question tools return the user's actual
                    // input — clearing that payload makes the model
                    // re-ask forever. Skip protected tools outright.
                    let protected = use_name
                        .get(tool_use_id.as_str())
                        .map(|name| config.protected_tool_names.iter().any(|p| p == name))
                        .unwrap_or(false);
                    if protected {
                        continue;
                    }
                    // `user_seen` has been incremented for the current
                    // message if it anchors a turn, so age = total -
                    // user_seen counts full turns strictly after this
                    // result.
                    let age = total_turns.saturating_sub(user_seen);
                    let cost = estimate_block_tokens(result_content);
                    candidates.push((mi, bi, age, cost, tool_use_id.clone()));
                }
            }
        }
    }

    // Pass 2: filter — drop everything within `preserve_last_n` (most
    // recent), drop ages below `min_age_turns`, drop cheap blocks.
    let total_results = candidates.len();
    let preserve_floor = total_results.saturating_sub(config.preserve_last_n as usize);
    let mut to_clear: Vec<(usize, usize, String, u32)> = Vec::new();
    for (idx, (mi, bi, age, cost, tu_id)) in candidates.into_iter().enumerate() {
        if idx >= preserve_floor {
            // In the preserve window — skip.
            continue;
        }
        if age < config.min_age_turns {
            continue;
        }
        if cost < config.min_tokens_per_result {
            continue;
        }
        to_clear.push((mi, bi, tu_id, cost));
    }

    // Pass 3: mutate. Instead of replacing the whole payload with the
    // marker, keep the head and the tail (errors conventionally live at
    // the END of a command's output) and replace only the middle — a
    // fully-cleared result made weak models re-run the same commands to
    // recover what they had just read.
    let mut cleared = 0usize;
    let mut tokens_freed: u32 = 0;
    let mut cleared_ids: Vec<String> = Vec::new();
    let mut modified_uuids: Vec<Uuid> = Vec::new();
    for (mi, bi, tu_id, cost) in to_clear {
        if let Some(Message::User {
            content, header, ..
        }) = messages.get_mut(mi)
        {
            if let Some(ContentBlock::ToolResult { content: c, .. }) = content.get_mut(bi) {
                let pruned = prune_payload(c, config.prune_head_chars, config.prune_tail_chars);
                let pruned_cost: u32 = match &pruned {
                    ToolResultContent::Text(t) => estimate_tokens_in_text(t),
                    ToolResultContent::Blocks(bs) => bs
                        .iter()
                        .map(|b| {
                            let m = Message::User {
                                header: crate::message::Header::new(),
                                content: vec![b.clone()],
                            };
                            estimate_tokens(&m)
                        })
                        .fold(0u32, u32::saturating_add),
                };
                let freed = cost.saturating_sub(pruned_cost);
                *c = pruned;
                cleared = cleared.saturating_add(1);
                tokens_freed = tokens_freed.saturating_add(freed);
                cleared_ids.push(tu_id);
                if !modified_uuids.contains(&header.uuid) {
                    modified_uuids.push(header.uuid);
                }
            }
        }
    }

    MicrocompactResult {
        cleared_count: cleared,
        tokens_freed,
        cleared_tool_use_ids: cleared_ids,
        modified_message_uuids: modified_uuids,
    }
}

/// Replace the MIDDLE of a tool-result payload with the cleared marker,
/// retaining `head` leading and `tail` trailing characters. Text blocks
/// are sliced by Unicode scalar value (a retained boundary never splits
/// a surrogate pair); rich-block payloads are pruned per text block,
/// keeping non-text blocks intact.
fn prune_payload(content: &ToolResultContent, head: u32, tail: u32) -> ToolResultContent {
    match content {
        ToolResultContent::Text(t) => ToolResultContent::Text(prune_text(t, head, tail)),
        ToolResultContent::Blocks(bs) => ToolResultContent::Blocks(
            bs.iter()
                .map(|b| match b {
                    ContentBlock::Text { text } => ContentBlock::Text {
                        text: prune_text(text, head, tail),
                    },
                    other => other.clone(),
                })
                .collect(),
        ),
    }
}

/// Keep `head` leading and `tail` trailing characters of `text`,
/// replacing the middle with the cleared marker. Payloads that fit
/// within `head + tail` are returned untouched (nothing to prune).
fn prune_text(text: &str, head: u32, tail: u32) -> String {
    let chars: Vec<char> = text.chars().collect();
    let total = chars.len();
    let head = head as usize;
    let tail = tail as usize;
    if total <= head.saturating_add(tail) {
        return text.to_string();
    }
    let kept_head: String = chars[..head].iter().collect();
    let kept_tail: String = chars[total - tail..].iter().collect();
    format!(
        "{kept_head}\n\n{CLEARED_PLACEHOLDER} ({pruned} chars pruned)\n\n{kept_tail}",
        pruned = total - head - tail,
    )
}

/// Run [`microcompact`] over a [`MessageStore`]. The store is append-only,
/// so the messages are snapshotted, mutated, and the store rebuilt — but
/// ONLY when something was actually cleared. UUIDs are preserved by the
/// rebuild, so the DAG and `parent_uuid` references stay valid.
pub fn apply_microcompact_to_store(
    store: &mut crate::message::MessageStore,
    config: &MicrocompactConfig,
) -> Result<MicrocompactResult, crate::error::AgentError> {
    let mut snapshot: Vec<Message> = store.iter().cloned().collect();
    let result = microcompact(&mut snapshot, config);
    if result.cleared_count > 0 {
        let mut rebuilt = crate::message::MessageStore::new();
        for msg in snapshot {
            rebuilt.push(msg)?;
        }
        *store = rebuilt;
    }
    Ok(result)
}

/// Per-block token estimate matching the rule in [`super::estimate_tokens`].
fn estimate_block_tokens(content: &ToolResultContent) -> u32 {
    match content {
        ToolResultContent::Text(t) => estimate_tokens_in_text(t),
        ToolResultContent::Blocks(bs) => bs
            .iter()
            .map(|b| {
                // Wrap in a synthetic User message to reuse estimator.
                let m = Message::User {
                    header: crate::message::Header::new(),
                    content: vec![b.clone()],
                };
                estimate_tokens(&m)
            })
            .fold(0u32, u32::saturating_add),
    }
}

fn estimate_tokens_in_text(text: &str) -> u32 {
    super::estimate_text_tokens(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Header;

    fn user_with_tool_result(tu_id: &str, payload_text: &str) -> Message {
        Message::User {
            header: Header::new(),
            content: vec![ContentBlock::ToolResult {
                tool_use_id: tu_id.into(),
                content: ToolResultContent::Text(payload_text.into()),
                is_error: false,
            }],
        }
    }

    fn user_text(text: &str) -> Message {
        Message::User {
            header: Header::new(),
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    #[test]
    fn cleared_placeholder_string_is_canonical() {
        assert_eq!(CLEARED_PLACEHOLDER, "[Old tool result content cleared]");
    }

    #[test]
    fn microcompact_no_op_under_min_age() {
        // Only 1 turn of history; min_age = 3 — nothing eligible.
        let mut messages = vec![user_with_tool_result("tu_1", &"x".repeat(800))];
        let cfg = MicrocompactConfig::default();
        let result = microcompact(&mut messages, &cfg);
        assert_eq!(result.cleared_count, 0);
    }

    #[test]
    fn microcompact_preserves_last_n() {
        // 6 turns, all containing tool_result, preserve_last_n = 5.
        // Only the oldest (1 result) should be eligible.
        let mut messages: Vec<Message> = (0..6)
            .map(|i| user_with_tool_result(&format!("tu_{i}"), &"x".repeat(800)))
            .collect();
        let cfg = MicrocompactConfig {
            min_age_turns: 0,
            preserve_last_n: 5,
            min_tokens_per_result: 0,
            protected_tool_names: Vec::new(),
            prune_head_chars: 1000,
            prune_tail_chars: 500,
        };
        let result = microcompact(&mut messages, &cfg);
        assert_eq!(result.cleared_count, 1);
        assert_eq!(result.cleared_tool_use_ids, vec!["tu_0"]);
    }

    #[test]
    fn microcompact_skips_cheap_results() {
        let mut messages = vec![
            user_with_tool_result("tu_old", "a"), // tiny payload
            user_text("filler"),
            user_text("filler"),
            user_text("filler"),
            user_text("now"),
        ];
        let cfg = MicrocompactConfig {
            min_age_turns: 1,
            preserve_last_n: 0,
            min_tokens_per_result: 100, // require >= 100
            protected_tool_names: Vec::new(),
            prune_head_chars: 1000,
            prune_tail_chars: 500,
        };
        let result = microcompact(&mut messages, &cfg);
        assert_eq!(result.cleared_count, 0);
    }

    #[test]
    fn microcompact_clears_old_large_results() {
        // 5 user turns; only the first has a tool result with 500
        // tokens of payload; min_age = 2; preserve_last_n = 0 (no
        // most-recent floor — age + size are the only gates).
        let big = "x".repeat(2_000); // ~500 tokens
        let mut messages = vec![
            user_with_tool_result("tu_old", &big),
            user_text("turn 2"),
            user_text("turn 3"),
            user_text("turn 4"),
            user_text("turn 5"),
        ];
        let cfg = MicrocompactConfig {
            min_age_turns: 2,
            preserve_last_n: 0,
            min_tokens_per_result: 100,
            protected_tool_names: Vec::new(),
            prune_head_chars: 1000,
            prune_tail_chars: 500,
        };
        let result = microcompact(&mut messages, &cfg);
        assert_eq!(result.cleared_count, 1);
        assert_eq!(result.cleared_tool_use_ids, vec!["tu_old"]);
        assert!(result.tokens_freed > 100);
        // Verify the content was pruned, not erased: head + tail survive
        // around the marker (default head 1000 / tail 500, so of the
        // 2000 chars the head 1000 and tail 500 remain).
        if let Message::User { content, .. } = &messages[0] {
            if let ContentBlock::ToolResult {
                content: ToolResultContent::Text(t),
                ..
            } = &content[0]
            {
                assert!(t.contains(CLEARED_PLACEHOLDER), "{t}");
                assert!(t.starts_with(&"x".repeat(1000)), "head must survive");
                assert!(t.ends_with(&"x".repeat(500)), "tail must survive");
            } else {
                panic!("expected Text content");
            }
        }
    }

    #[test]
    fn prune_text_keeps_head_and_tail_and_marks_the_middle() {
        let pruned = prune_text(&"a".repeat(2_000), 100, 50);
        assert!(pruned.starts_with(&"a".repeat(100)), "head");
        assert!(pruned.ends_with(&"a".repeat(50)), "tail");
        assert!(pruned.contains(CLEARED_PLACEHOLDER));
        assert!(pruned.contains("1850 chars pruned"));
        // Payloads within head+tail pass through untouched.
        assert_eq!(prune_text("short", 100, 50), "short");
        // Unicode-scalar slicing: a CJK string keeps whole characters.
        let cjk = "中".repeat(300);
        let out = prune_text(&cjk, 10, 5);
        assert!(out.starts_with(&"中".repeat(10)));
        assert!(out.ends_with(&"中".repeat(5)));
        assert!(out.contains(CLEARED_PLACEHOLDER));
    }

    #[test]
    fn a_single_turns_tool_results_are_never_cleared_mid_turn() {
        // Regression: one human prompt followed by a long run of pure
        // tool-result batches (how the query loop feeds results back).
        // Counting those batches as turns made everything but the last
        // few results "old" — the model then re-ran the same commands
        // to recover what had just been cleared. With human-anchored
        // ages every result in the current turn is age 0: untouchable.
        let mut messages = vec![user_text("fix the tests")];
        for i in 0..30 {
            messages.push(user_with_tool_result(
                &format!("tu_{i}"),
                &"x".repeat(2_000),
            ));
        }
        let result = microcompact(&mut messages, &MicrocompactConfig::default());
        assert_eq!(result.cleared_count, 0, "mid-turn results must survive");
    }

    #[test]
    fn injected_guard_nudges_do_not_advance_tool_result_ages() {
        // A turn that trips the loop guard a few times must not have its
        // own results aged into clearing range by the nudges.
        let nudge = || Message::User {
            header: Header::new(),
            content: vec![ContentBlock::Text {
                text: "<system-reminder>Loop guard: change approach.</system-reminder>".into(),
            }],
        };
        let mut messages = vec![user_text("fix the tests")];
        for i in 0..6 {
            messages.push(user_with_tool_result(
                &format!("tu_{i}"),
                &"x".repeat(2_000),
            ));
            messages.push(nudge());
        }
        let result = microcompact(&mut messages, &MicrocompactConfig::default());
        assert_eq!(result.cleared_count, 0, "nudges are not human turns");

        // A real prompt that merely STARTS with a prepended reminder still
        // anchors a turn.
        let prefixed = Message::User {
            header: Header::new(),
            content: vec![ContentBlock::Text {
                text: "<system-reminder>note</system-reminder>\nand now do this".into(),
            }],
        };
        assert!(is_human_turn(&prefixed));
        assert!(!is_human_turn(&nudge()));
    }

    #[test]
    fn an_interrupt_fold_counts_as_a_human_turn() {
        // A user interrupt folds its text into the same User message as
        // the pending tool_result — the human spoke, so it anchors a
        // turn.
        let folded = Message::User {
            header: Header::new(),
            content: vec![
                ContentBlock::ToolResult {
                    tool_use_id: "tu_pending".into(),
                    content: ToolResultContent::Text("aborted".into()),
                    is_error: true,
                },
                ContentBlock::Text {
                    text: "stop, do this instead".into(),
                },
            ],
        };
        assert!(is_human_turn(&folded));
        assert!(is_human_turn(&user_text("plain prompt")));
        assert!(!is_human_turn(&user_with_tool_result("tu", "payload")));
    }

    #[test]
    fn microcompact_handles_blocks_payload() {
        // ToolResultContent::Blocks variant should also estimate tokens.
        let inner_blocks = ToolResultContent::Blocks(vec![ContentBlock::Text {
            text: "a".repeat(800),
        }]);
        let mut messages = vec![
            Message::User {
                header: Header::new(),
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "tu_1".into(),
                    content: inner_blocks,
                    is_error: false,
                }],
            },
            user_text("turn 2"),
            user_text("turn 3"),
            user_text("now"),
        ];
        let cfg = MicrocompactConfig {
            min_age_turns: 1,
            preserve_last_n: 0,
            min_tokens_per_result: 50,
            protected_tool_names: Vec::new(),
            prune_head_chars: 1000,
            prune_tail_chars: 500,
        };
        let result = microcompact(&mut messages, &cfg);
        assert_eq!(result.cleared_count, 1);
    }

    #[test]
    fn apply_to_store_rebuilds_and_preserves_uuids() {
        use crate::message::MessageStore;
        let mut store = MessageStore::new();
        store
            .push(user_with_tool_result("tu_old", &"x".repeat(2_000)))
            .unwrap();
        for i in 0..4 {
            store.push(user_text(&format!("turn {i}"))).unwrap();
        }
        let uuids_before: Vec<_> = store.iter().map(|m| m.uuid()).collect();

        let cfg = MicrocompactConfig {
            min_age_turns: 2,
            preserve_last_n: 0,
            min_tokens_per_result: 100,
            protected_tool_names: Vec::new(),
            prune_head_chars: 1000,
            prune_tail_chars: 500,
        };
        let result = apply_microcompact_to_store(&mut store, &cfg).unwrap();
        assert_eq!(result.cleared_count, 1);
        assert_eq!(result.cleared_tool_use_ids, vec!["tu_old"]);

        // UUIDs and order survive the rebuild.
        let uuids_after: Vec<_> = store.iter().map(|m| m.uuid()).collect();
        assert_eq!(uuids_before, uuids_after);

        // The payload is pruned in the store: head + marker + tail.
        let first = store.iter().next().unwrap();
        match first {
            Message::User { content, .. } => match &content[0] {
                ContentBlock::ToolResult { content, .. } => match content {
                    ToolResultContent::Text(t) => {
                        assert!(t.contains(CLEARED_PLACEHOLDER), "{t}");
                        assert!(t.starts_with(&"x".repeat(1000)), "head must survive");
                        assert!(t.ends_with(&"x".repeat(500)), "tail must survive");
                    }
                    other => panic!("expected text payload, got {other:?}"),
                },
                other => panic!("expected tool result, got {other:?}"),
            },
            other => panic!("expected user message, got {other:?}"),
        }
    }

    #[test]
    fn protected_tool_results_are_never_cleared() {
        // AskUserQuestion returns the user's ACTUAL input — clearing it
        // makes the model believe it was never answered and re-ask
        // forever. A protected name must survive every gate (age, size,
        // preserve floor), while an unprotected same-shaped result is
        // still cleared.
        let big = "x".repeat(2_000); // ~500 tokens
        let mut messages = vec![
            Message::Assistant {
                header: Header::new(),
                content: vec![ContentBlock::ToolUse {
                    id: "tu_ask".into(),
                    name: "AskUserQuestion".into(),
                    input: serde_json::json!({"questions": []}),
                }],
            },
            user_with_tool_result("tu_ask", &big),
            user_text("turn 2"),
            user_text("turn 3"),
            user_text("turn 4"),
            user_text("turn 5"),
        ];
        let protected = MicrocompactConfig {
            min_age_turns: 1,
            preserve_last_n: 0,
            min_tokens_per_result: 100,
            protected_tool_names: vec!["AskUserQuestion".into()],
            prune_head_chars: 1000,
            prune_tail_chars: 500,
        };
        let result = microcompact(&mut messages, &protected);
        assert_eq!(
            result.cleared_count, 0,
            "protected result must survive microcompaction"
        );
        // The payload is still intact.
        if let Message::User { content, .. } = &messages[1] {
            if let ContentBlock::ToolResult {
                content: ToolResultContent::Text(t),
                ..
            } = &content[0]
            {
                assert_eq!(t, &big, "payload must be untouched");
            } else {
                panic!("expected Text content");
            }
        }
        // Same shape WITHOUT the protection clears as before.
        let mut messages = vec![
            Message::Assistant {
                header: Header::new(),
                content: vec![ContentBlock::ToolUse {
                    id: "tu_bash".into(),
                    name: "Bash".into(),
                    input: serde_json::json!({"command": "ls"}),
                }],
            },
            user_with_tool_result("tu_bash", &big),
            user_text("turn 2"),
            user_text("turn 3"),
            user_text("turn 4"),
            user_text("turn 5"),
        ];
        let unprotected = MicrocompactConfig {
            min_age_turns: 1,
            preserve_last_n: 0,
            min_tokens_per_result: 100,
            protected_tool_names: vec!["AskUserQuestion".into()],
            prune_head_chars: 1000,
            prune_tail_chars: 500,
        };
        let result = microcompact(&mut messages, &unprotected);
        assert_eq!(result.cleared_count, 1, "unprotected result still clears");
        assert_eq!(result.cleared_tool_use_ids, vec!["tu_bash"]);
    }

    #[test]
    fn apply_to_store_no_op_leaves_store_untouched() {
        use crate::message::MessageStore;
        let mut store = MessageStore::new();
        store
            .push(user_text("only text, nothing to clear"))
            .unwrap();
        let result =
            apply_microcompact_to_store(&mut store, &MicrocompactConfig::default()).unwrap();
        assert_eq!(result.cleared_count, 0);
        assert_eq!(store.len(), 1);
    }
}
