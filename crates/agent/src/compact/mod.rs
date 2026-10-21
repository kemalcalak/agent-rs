//! Token estimation + compaction boundary markers (Phase 4 / Task 4.3).
//!
//! Estimation rule (matches Zig agent/src/compact.zig and Claude Code's
//! approximation):
//! - 4 ASCII characters ≈ 1 token.
//! - 1 CJK character ≈ 2 tokens (Han/Hiragana/Katakana/Hangul ranges;
//!   deliberately conservative — real BPE runs 1.5–3 tokens per glyph,
//!   and over-counting only compacts early while under-counting lets
//!   the provider 400 the request first).
//! - Tool-use JSON: serialize and count bytes / 4.
//! - Other content blocks: text + structural overhead (~8 tokens for
//!   image source headers, ~4 tokens for tool-result wrappers).
//!
//! ## Known accuracy trade-off
//!
//! The CJK 2:1 rule **over-counts** real BPE token cost — Anthropic's
//! actual tokenizer typically produces 1.5–3 tokens per CJK glyph, so
//! 2:1 errs high. The plan deliberately accepts this trade-off (the
//! rule is fast + deterministic; over-counting compacts a little early,
//! while the old 1:1 under-count let CJK-heavy sessions hit the
//! provider's true context limit before our sliding-window / compact
//! logic thought they should — the provider 400s and compaction is
//! forced, mid-turn, at the worst moment). Phase 5+ may ship a
//! tiktoken-based exact estimator behind a feature gate.
//!
//! Boundary marker: a synthetic [`Message::System`] inserted via
//! [`insert_boundary_marker`] separates "old context (summary above)"
//! from "active context (below)" — used by reactive compaction in
//! later phases.
//!
//! ## Submodules (Tier 1 claude-code parity)
//!
//! - [`prompt`] — `<analysis>`/`<summary>` template + parser.
//! - [`summarize`] — [`summarize::compact_conversation`] calls a
//!   provider, parses the model's response, and produces a
//!   [`summarize::CompactionResult`] ready to splice into a
//!   [`MessageStore`].

pub mod auto;
pub mod grouping;
pub mod microcompact;
pub mod post_cleanup;
pub mod prompt;
pub mod session_memory;
pub mod summarize;
pub mod with_hooks;

pub use auto::{
    auto_compact_threshold, effective_context_window, manual_compact_threshold,
    microcompact_threshold, warning_threshold, AutoCompactDecision, AutoCompactReason,
    AutoCompactState, AUTOCOMPACT_BUFFER_TOKENS, ERROR_THRESHOLD_BUFFER_TOKENS,
    MANUAL_COMPACT_BUFFER_TOKENS, MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES,
    WARNING_THRESHOLD_BUFFER_TOKENS,
};
pub use grouping::{group_messages, safe_split_index, GroupKind, MessageGroup};
pub use microcompact::{
    apply_microcompact_to_store, microcompact, MicrocompactConfig, MicrocompactResult,
    CLEARED_PLACEHOLDER,
};
pub use post_cleanup::{
    build_post_compact_message, FileAttachment, PostCompactConfig, PostCompactResult,
};
pub use prompt::{
    parse_summary_response, summarization_prompt, ParseSummaryError, ParsedSummary,
    PartialCompactDirection,
};
pub use session_memory::{
    extract_memories_from_analysis, promote_to_store, InMemoryStore, JsonlMemoryStore,
    SessionMemoryEntry, SessionMemoryError, SessionMemoryKind, SessionMemoryStore,
};
pub use summarize::{
    apply_compaction_to_store, compact_conversation, CompactError, CompactionResult,
    COMPACT_BOUNDARY_TEXT, MAX_OUTPUT_TOKENS_FOR_SUMMARY,
};
pub use with_hooks::{compact_with_hooks, CompactTrigger, CompactWithHooksRequest};

use crate::message::{ContentBlock, Header, ImageSource, Message, MessageStore, ToolResultContent};

const BOUNDARY_MARKER_TEXT: &str = "CONTEXT SUMMARY BELOW";

/// Repair tool-pairing damage in a message slice, IN PLACE.
///
/// A store damaged by an OLDER compaction — whose split tombstoned one
/// half of an `assistant tool_use → user tool_result` pair — carries the
/// orphan half forever. Strict gateways reject that shape with a 400
/// ("The response must be a response to a preceding message with
/// 'tool_calls'"), so EVERY subsequent turn fails and the session looks
/// wedged "right after compaction". Two shapes are repaired:
///
/// - a `tool_result` whose `tool_use` is not live (tombstoned / absent)
///   before it is downgraded to a plain-text block (content preserved);
/// - a `tool_use` with no live result after it is downgraded to a text
///   block. With `keep_trailing_dangling`, a dangling `tool_use` in the
///   FINAL message is left alone — a legal "awaiting result" shape on
///   Anthropic (request snapshots use this). Store repair uses `false`:
///   persisted transcripts keep strict pairing.
///
/// Returns how many messages were rewritten. Tombstoned messages are
/// never touched (providers skip them anyway).
pub fn repair_pairing(messages: &mut [Message], keep_trailing_dangling: bool) -> usize {
    use std::collections::HashMap;
    // Last message index carrying each tool_use id (an id should be unique,
    // but a repeated one counts only if a result follows its LAST use).
    let mut use_at: HashMap<String, usize> = HashMap::new();
    for (i, m) in messages.iter().enumerate() {
        if let Message::Assistant { content, .. } = m {
            for b in content {
                if let ContentBlock::ToolUse { id, .. } = b {
                    use_at.insert(id.clone(), i);
                }
            }
        }
    }
    // Whether each tool_use has a live result AFTER its use.
    let mut result_after: HashMap<String, bool> = HashMap::with_capacity(use_at.len());
    for (id, &at) in &use_at {
        let answered = messages.iter().enumerate().any(|(j, m)| {
            j > at
                && matches!(m, Message::User { content, .. } if content.iter().any(|b| {
                    matches!(b, ContentBlock::ToolResult { tool_use_id, .. } if tool_use_id == id)
                }))
        });
        result_after.insert(id.clone(), answered);
    }
    let last = messages.len().saturating_sub(1);
    let mut rewritten = 0usize;
    for (i, m) in messages.iter_mut().enumerate() {
        match m {
            Message::User { content, .. } => {
                for b in content.iter_mut() {
                    let orphan = match b {
                        ContentBlock::ToolResult {
                            tool_use_id,
                            content: rc,
                            ..
                        } if !use_at
                            .get(tool_use_id.as_str())
                            .map(|&at| at < i)
                            .unwrap_or(false) =>
                        {
                            Some(format!("[tool result]\n{}", tool_result_to_text(rc)))
                        }
                        _ => None,
                    };
                    if let Some(text) = orphan {
                        *b = ContentBlock::Text { text };
                        rewritten = rewritten.saturating_add(1);
                    }
                }
            }
            Message::Assistant { content, .. } => {
                for b in content.iter_mut() {
                    let dangling = match b {
                        ContentBlock::ToolUse { id, name, .. }
                            if (i != last || !keep_trailing_dangling)
                                && !result_after.get(id.as_str()).copied().unwrap_or(false) =>
                        {
                            Some(format!("[tool call: {name}]\n{id}"))
                        }
                        _ => None,
                    };
                    if let Some(text) = dangling {
                        *b = ContentBlock::Text { text };
                        rewritten = rewritten.saturating_add(1);
                    }
                }
            }
            _ => {}
        }
    }
    rewritten
}

/// Repair tool-pairing damage in a [`MessageStore`] IN PLACE (used when a
/// session is loaded from disk — fix the damage once instead of carrying
/// it into every request). Message UUIDs survive; only the broken blocks
/// are rewritten. Returns how many messages were rewritten.
pub fn repair_store_pairing(store: &mut MessageStore) -> usize {
    let mut snapshot: Vec<Message> = store.iter().cloned().collect();
    let rewritten = repair_pairing(&mut snapshot, false);
    if rewritten > 0 {
        let mut rebuilt = MessageStore::new();
        for msg in snapshot {
            let _ = rebuilt.push(msg);
        }
        *store = rebuilt;
    }
    rewritten
}

/// Flatten a tool-result payload to plain text (content feeds the model
/// even after the block is downgraded).
pub fn tool_result_to_text(content: &ToolResultContent) -> String {
    match content {
        ToolResultContent::Text(t) => t.clone(),
        ToolResultContent::Blocks(blocks) => blocks
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

/// Estimate the number of tokens used by `msg`.
///
/// This is a **byte-level approximation**, not a true tokenizer. Don't
/// rely on it for hard prompt-cache deduplication; it's intended for
/// budget triage (sliding window + compact decisions).
pub fn estimate_tokens(msg: &Message) -> u32 {
    match msg {
        Message::User { content, .. } | Message::Assistant { content, .. } => content
            .iter()
            .map(estimate_block_tokens)
            .fold(0, u32::saturating_add),
        Message::System { text, .. }
        | Message::Progress { note: text, .. }
        | Message::Tombstone { reason: text, .. } => estimate_text_tokens(text).saturating_add(2),
    }
}

fn estimate_block_tokens(block: &ContentBlock) -> u32 {
    match block {
        ContentBlock::Text { text } => estimate_text_tokens(text),
        ContentBlock::Image { source } => match source {
            ImageSource::Base64 { data, .. } => {
                // Base64 image: per Anthropic, images cost roughly
                // (width * height) / 750 tokens. We don't have
                // dimensions; fall back to byte-length / 4 + a flat
                // overhead of 8 for the source header.
                ((data.len() as u32) / 4).saturating_add(8)
            }
            ImageSource::Url { url } => estimate_text_tokens(url).saturating_add(8),
            // File-id references are tiny strings on the wire; the
            // server fetches the actual bytes so we don't bill them
            // against this turn's context window.
            ImageSource::File { file_id } => estimate_text_tokens(file_id).saturating_add(8),
        },
        // Document blocks travel as either inline base64 or file
        // references. Same byte-rate heuristic as images, plus a
        // slightly larger overhead because document handling on
        // Anthropic's side adds OCR / parse metadata.
        ContentBlock::Document { source } => match source {
            crate::message::DocumentSource::Base64 { data, .. } => {
                ((data.len() as u32) / 4).saturating_add(16)
            }
            crate::message::DocumentSource::Url { url } => {
                estimate_text_tokens(url).saturating_add(16)
            }
            crate::message::DocumentSource::File { file_id } => {
                estimate_text_tokens(file_id).saturating_add(16)
            }
        },
        ContentBlock::ToolUse { id, name, input } => {
            let json_len = serde_json::to_string(input).map(|s| s.len()).unwrap_or(0);
            estimate_text_tokens(id)
                .saturating_add(estimate_text_tokens(name))
                .saturating_add((json_len as u32) / 4)
                .saturating_add(4)
        }
        ContentBlock::ToolResult {
            tool_use_id,
            content,
            ..
        } => {
            let inner = match content {
                ToolResultContent::Text(t) => estimate_text_tokens(t),
                ToolResultContent::Blocks(bs) => bs
                    .iter()
                    .map(estimate_block_tokens)
                    .fold(0, u32::saturating_add),
            };
            estimate_text_tokens(tool_use_id)
                .saturating_add(inner)
                .saturating_add(4)
        }
        ContentBlock::Thinking { thinking, .. } => estimate_text_tokens(thinking).saturating_add(2),
    }
}

/// Estimate tokens for a single text string.
///
/// - CJK chars are counted 2:1 (each character ≈ 2 tokens).
/// - ASCII / non-CJK runs are counted at 4 chars per token.
///
/// The CJK 2:1 rule is deliberately CONSERVATIVE (over-count vs. the
/// real BPE tokenizer, which typically produces 1.5–3 tokens per CJK
/// glyph): under-counting makes auto-compaction fire late, so a
/// CJK-heavy session hits the provider's true context limit first and
/// the provider 400s the request — the exact "compaction never ran"
/// wedge. A slight over-count only compacts a little early.
pub fn estimate_text_tokens(text: &str) -> u32 {
    let mut cjk = 0u32;
    let mut other = 0u32;
    for ch in text.chars() {
        if is_cjk(ch) {
            cjk = cjk.saturating_add(1);
        } else {
            other = other.saturating_add(1);
        }
    }
    cjk.saturating_mul(2).saturating_add(other / 4)
}

fn is_cjk(ch: char) -> bool {
    matches!(
        ch as u32,
        0x3040..=0x30FF      // Hiragana / Katakana
        | 0x3400..=0x4DBF    // CJK Unified Ideographs Extension A
        | 0x4E00..=0x9FFF    // CJK Unified Ideographs
        | 0xAC00..=0xD7AF    // Hangul Syllables
        | 0xF900..=0xFAFF    // CJK Compatibility Ideographs
        | 0x20000..=0x2A6DF  // CJK Extension B
        | 0x2A700..=0x2B73F  // CJK Extension C
    )
}

/// Insert a `Message::System` marker at `position` separating older
/// context from newer. The marker carries the literal text
/// "CONTEXT SUMMARY BELOW" so any provider rendering it as a system
/// message will see the boundary explicitly.
///
/// `position` is the index BEFORE which the marker is inserted (a
/// position of 0 prepends; a position equal to the message count
/// appends). Returns `Err(InvalidMessage)` on out-of-bounds.
pub fn insert_boundary_marker(
    store: &mut MessageStore,
    position: usize,
) -> Result<uuid::Uuid, crate::error::AgentError> {
    let marker = Message::System {
        header: Header::new(),
        text: BOUNDARY_MARKER_TEXT.into(),
    };
    let marker_uuid = marker.uuid();
    if position > store.len() {
        return Err(crate::error::AgentError::InvalidMessage(format!(
            "boundary marker position {position} exceeds store length {}",
            store.len()
        )));
    }
    if position == store.len() {
        store.push(marker)?;
        return Ok(marker_uuid);
    }
    // Insertion in the middle requires a different MessageStore
    // surface — append-only doesn't support arbitrary insertion. We
    // implement this by collecting all messages, building a fresh
    // store, and pushing the marker at the right spot.
    let mut snapshot: Vec<Message> = store.iter().cloned().collect();
    snapshot.insert(position, marker);
    *store = MessageStore::new();
    for m in snapshot {
        store.push(m)?;
    }
    Ok(marker_uuid)
}

/// Returns true if `msg` is the boundary marker inserted by
/// [`insert_boundary_marker`].
pub fn is_boundary_marker(msg: &Message) -> bool {
    matches!(msg, Message::System { text, .. } if text == BOUNDARY_MARKER_TEXT)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{ContentBlock, Header, Message};

    fn user_text(text: &str) -> Message {
        Message::User {
            header: Header::new(),
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    #[test]
    fn estimate_ascii_4_chars_per_token() {
        assert_eq!(estimate_text_tokens("abcd"), 1);
        assert_eq!(estimate_text_tokens("abcdefgh"), 2);
        assert_eq!(estimate_text_tokens(""), 0);
    }

    #[test]
    fn estimate_cjk_two_per_char() {
        assert_eq!(estimate_text_tokens("你好世界"), 8);
        assert_eq!(estimate_text_tokens("こんにちは"), 10);
        assert_eq!(estimate_text_tokens("안녕하세요"), 10);
    }

    #[test]
    fn estimate_mixed_cjk_and_ascii() {
        // 4 ASCII + 4 CJK = 1 + 8 = 9 tokens.
        assert_eq!(estimate_text_tokens("abcd你好世界"), 9);
    }

    #[test]
    fn estimate_user_message_text_only() {
        let msg = user_text("hello world");
        // 11 chars / 4 = 2 tokens.
        assert_eq!(estimate_tokens(&msg), 2);
    }

    #[test]
    fn estimate_tool_use_includes_json_bytes() {
        let msg = Message::Assistant {
            header: Header::new(),
            content: vec![ContentBlock::ToolUse {
                id: "tu_1".into(),
                name: "calc".into(),
                input: serde_json::json!({"a": 1, "b": 2}),
            }],
        };
        // id 4 chars / 4 = 1, name 4 chars / 4 = 1, json `{"a":1,"b":2}` = 13 / 4 = 3, +4 overhead.
        // Total = 1 + 1 + 3 + 4 = 9.
        let est = estimate_tokens(&msg);
        assert!((5..=15).contains(&est), "got {est}");
    }

    #[test]
    fn estimate_within_5_percent_for_repeated_short_text() {
        // 100 messages, each "abcd" (1 token by our rule) = 100 tokens nominal.
        let msgs: Vec<Message> = (0..100).map(|_| user_text("abcd")).collect();
        let total: u32 = msgs.iter().map(estimate_tokens).sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn estimate_corpus_within_5_percent_of_manual_count() {
        // Mixed-language sample where we manually computed the
        // expected token count under our 4-ASCII / 2-CJK rule. This
        // pins the heuristic against drift; if a future change moves
        // the estimate by more than 5%, this test catches it.
        //
        // Per-message expected tokens (our rule, NOT real BPE):
        //   "Hello, world!"          (13 chars, ASCII)         → 13/4 = 3
        //   "你好世界"               (4 CJK)                    → 8
        //   "ありがとう"             (5 Hiragana)               → 10
        //   "안녕하세요"             (5 Hangul)                 → 10
        //   "abcdabcdabcdabcd"       (16 ASCII)                 → 4
        //   "Mixed: 你好 abcd!"      (4 CJK + 11 ASCII = 8 + 2) → 10
        //   "こんにちは abcd"        (5 Hiragana + 5 ASCII)     → 10 + 1 = 11
        //   "" (empty)                                          → 0
        //   "x" * 400 (400 ASCII)                              → 100
        //   "中" * 50  (50 CJK)                                 → 100
        let samples: &[(&str, u32)] = &[
            ("Hello, world!", 3),
            ("你好世界", 8),
            ("ありがとう", 10),
            ("안녕하세요", 10),
            ("abcdabcdabcdabcd", 4),
            ("Mixed: 你好 abcd!", 10),
            ("こんにちは abcd", 11),
            ("", 0),
            (&"x".repeat(400), 100),
            (&"中".repeat(50), 100),
        ];
        let expected_total: u32 = samples.iter().map(|(_, e)| *e).sum();
        let actual_total: u32 = samples
            .iter()
            .map(|(s, _)| estimate_tokens(&user_text(s)))
            .sum();
        let diff = actual_total.abs_diff(expected_total);
        let tolerance = expected_total * 5 / 100;
        assert!(
            diff <= tolerance,
            "estimate drift: expected {expected_total}, got {actual_total} (diff {diff}, tolerance {tolerance})"
        );
    }

    #[test]
    fn boundary_marker_appends() {
        let mut s = MessageStore::new();
        s.push(user_text("u1")).unwrap();
        s.push(user_text("u2")).unwrap();
        let marker_id = insert_boundary_marker(&mut s, 2).unwrap();
        assert_eq!(s.len(), 3);
        let last = s.iter().last().unwrap();
        assert!(is_boundary_marker(last));
        assert_eq!(last.uuid(), marker_id);
    }

    #[test]
    fn boundary_marker_inserts_in_middle() {
        let mut s = MessageStore::new();
        s.push(user_text("u1")).unwrap();
        s.push(user_text("u2")).unwrap();
        s.push(user_text("u3")).unwrap();
        insert_boundary_marker(&mut s, 1).unwrap();
        assert_eq!(s.len(), 4);
        let snapshot: Vec<&Message> = s.iter().collect();
        assert!(matches!(snapshot[0], Message::User { .. }));
        assert!(is_boundary_marker(snapshot[1]));
        assert!(matches!(snapshot[2], Message::User { .. }));
        assert!(matches!(snapshot[3], Message::User { .. }));
    }

    #[test]
    fn boundary_marker_out_of_bounds_errors() {
        let mut s = MessageStore::new();
        s.push(user_text("u1")).unwrap();
        let result = insert_boundary_marker(&mut s, 99);
        assert!(matches!(
            result,
            Err(crate::error::AgentError::InvalidMessage(_))
        ));
    }
}
