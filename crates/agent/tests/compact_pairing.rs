//! Regression: compaction must never leave an orphaned tool_result /
//! dangling tool_use in the store, or every later provider request 400s
//! ("The response must be a response to a preceding message with
//! 'tool_calls'") and the session wedges. Covers the store-mutating
//! split (pair_safe_split), the multi-result grouping (group_messages),
//! and the request-side defensive repair (sanitize_request_messages).

use std::sync::Mutex;

use agent::abort::AbortController;
use agent::compact::{apply_compaction_to_store, compact_conversation, PartialCompactDirection};
use agent::message::{ContentBlock, Header, Message, MessageStore, ToolResultContent};
use agent::provider::{Provider, ProviderCapabilities, StreamRequest};
use agent::stream::Event;
use async_trait::async_trait;
use futures::stream;

fn happy_response() -> &'static str {
    "<analysis>- did X</analysis><summary>Did X.</summary>"
}

#[derive(Debug)]
struct ScriptedProvider(Mutex<Vec<Event>>);

impl ScriptedProvider {
    fn from_text(t: &str) -> Self {
        Self(Mutex::new(vec![
            Event::TextDelta { delta: t.into() },
            Event::Result {
                data: Default::default(),
            },
        ]))
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
        _req: StreamRequest,
        _abort: AbortController,
    ) -> Result<Box<dyn EventStream>, AgentError> {
        let evs: Vec<Event> = self.0.lock().map(|mut g| std::mem::take(&mut *g)).unwrap();
        Ok(Box::new(stream::iter(evs.into_iter().map(Ok))))
    }
}

use agent::error::AgentError;
use agent::stream::EventStream;

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

/// A long realistic transcript: user prompts interleaved with tool pairs.
fn long_transcript(turns: usize) -> Vec<Message> {
    let mut msgs = vec![user("start the task")];
    for i in 0..turns {
        msgs.push(assistant_tool_use(&format!("t{i}")));
        msgs.push(user_tool_result(&format!("t{i}")));
        msgs.push(assistant(&format!("step {i} done")));
        msgs.push(user(&format!("continue {i}")));
    }
    msgs
}

/// Assert the store holds no orphaned tool_result and no dangling tool_use.
/// Tombstones do not count as holders.
fn assert_paired(msgs: &[Message], ctx: &str) {
    // Map from tool_use_id -> still-live tool_use message index (in order).
    let mut live_uses: Vec<(String, usize)> = Vec::new();
    for (i, m) in msgs.iter().enumerate() {
        if let Message::Assistant { content, .. } = m {
            for b in content {
                if let ContentBlock::ToolUse { id, .. } = b {
                    live_uses.push((id.clone(), i));
                }
            }
        }
    }
    for (i, m) in msgs.iter().enumerate() {
        if let Message::User { content, .. } = m {
            for b in content {
                if let ContentBlock::ToolResult { tool_use_id, .. } = b {
                    let holder = live_uses
                        .iter()
                        .find(|(id, idx)| id == tool_use_id && *idx < i);
                    assert!(
                        holder.is_some(),
                        "{ctx}: orphaned tool_result {tool_use_id:?} at message {i} \
                         (no live tool_use before it) — this shape 400s strict providers"
                    );
                }
            }
        }
    }
    // Every live tool_use must have a live result AFTER it.
    for (id, idx) in &live_uses {
        let answered = msgs.iter().enumerate().any(|(i, m)| {
            i > *idx
                && matches!(m, Message::User { content, .. } if content.iter().any(|b| {
                    matches!(b, ContentBlock::ToolResult { tool_use_id, .. } if tool_use_id == id)
                }))
        });
        assert!(
            answered,
            "{ctx}: dangling tool_use {id:?} at message {idx} — this shape 400s strict providers"
        );
    }
}

#[tokio::test]
async fn long_transcript_earliest_half_stays_paired() {
    // The user's reported scenario: a long agentic session hits the
    // auto-compact threshold; the STORE-mutating split must not sever any
    // assistant tool_use → user tool_result pair.
    let msgs = long_transcript(30);
    let provider = ScriptedProvider::from_text(happy_response());
    let result = compact_conversation(
        &msgs,
        &provider,
        "m",
        None,
        PartialCompactDirection::EarliestHalf,
        AbortController::default(),
    )
    .await
    .expect("compact should succeed");
    assert!(!result.replaced_uuids.is_empty());

    let mut store = MessageStore::new();
    for m in msgs {
        store.push(m).unwrap();
    }
    apply_compaction_to_store(&mut store, &result).unwrap();
    let after: Vec<Message> = store.iter().cloned().collect();
    assert_paired(&after, "long transcript EarliestHalf");
}

#[tokio::test]
async fn two_message_pair_earliest_half_stays_paired() {
    // Boundary: a manual /compact on a transcript that is exactly one
    // tool pair. pair_safe_split's clamp(1, len-1) lands INSIDE the pair.
    let msgs = vec![assistant_tool_use("t1"), user_tool_result("t1")];
    let provider = ScriptedProvider::from_text(happy_response());
    let result = compact_conversation(
        &msgs,
        &provider,
        "m",
        None,
        PartialCompactDirection::EarliestHalf,
        AbortController::default(),
    )
    .await
    .expect("compact should succeed");
    let mut store = MessageStore::new();
    for m in msgs {
        store.push(m).unwrap();
    }
    apply_compaction_to_store(&mut store, &result).unwrap();
    let after: Vec<Message> = store.iter().cloned().collect();
    assert_paired(&after, "two-message pair EarliestHalf");
}

#[tokio::test]
async fn multi_tool_use_split_results_stay_paired() {
    // One assistant message with two tool_uses whose results arrive in
    // TWO separate user messages. group_messages only binds the FIRST
    // result message to the pair; the second result message is
    // UserOnly, so a split right after the first result severs the
    // second pair half.
    let msgs = vec![
        user("go"),
        Message::Assistant {
            header: Header::new(),
            content: vec![
                ContentBlock::ToolUse {
                    id: "t1".into(),
                    name: "Bash".into(),
                    input: serde_json::json!({"command": "a"}),
                },
                ContentBlock::ToolUse {
                    id: "t2".into(),
                    name: "Bash".into(),
                    input: serde_json::json!({"command": "b"}),
                },
            ],
        },
        user_tool_result("t1"),
        user_tool_result("t2"),
        user("next"),
    ];
    let provider = ScriptedProvider::from_text(happy_response());
    let result = compact_conversation(
        &msgs,
        &provider,
        "m",
        None,
        PartialCompactDirection::EarliestHalf,
        AbortController::default(),
    )
    .await
    .expect("compact should succeed");
    let mut store = MessageStore::new();
    for m in msgs {
        store.push(m).unwrap();
    }
    apply_compaction_to_store(&mut store, &result).unwrap();
    let after: Vec<Message> = store.iter().cloned().collect();
    assert_paired(&after, "multi tool_use split results EarliestHalf");
}

#[tokio::test]
async fn pre_orphaned_store_is_repaired_by_next_compact() {
    // A store ALREADY containing an orphan (from an older zode whose
    // compaction severed a pair) must not stay broken: the next
    // compaction (manual or auto) has to repair or at least not
    // re-emit the orphan shape.
    let msgs = [
        user("go"),
        assistant_tool_use("t1"),
        user_tool_result("t1"),
        assistant_tool_use("t2"),
        user_tool_result("t2"),
        user("next"),
    ];
    // Simulate the old bug: tombstone t2's tool_use, keep its result.
    let mut store = MessageStore::new();
    for (i, m) in msgs.iter().enumerate() {
        if i == 3 {
            let header = m.header().clone();
            store
                .push(Message::Tombstone {
                    header,
                    reason: "compacted".into(),
                })
                .unwrap();
        } else {
            store.push(m.clone()).unwrap();
        }
    }
    let broken: Vec<Message> = store.iter().cloned().collect();
    // The broken store already fails the pairing invariant.
    let has_orphan = broken.iter().enumerate().any(|(i, m)| {
        matches!(m, Message::User { content, .. } if content.iter().any(|b| {
            matches!(b, ContentBlock::ToolResult { tool_use_id, .. } if !broken[..i].iter().any(|m2| {
                matches!(m2, Message::Assistant { content, .. } if content.iter().any(|b2| {
                    matches!(b2, ContentBlock::ToolUse { id, .. } if id == tool_use_id)
                }))
            }))
        }))
    });
    assert!(has_orphan, "precondition: store is broken");

    // Now run a Full compaction over it: everything is replaced, so the
    // orphan must disappear.
    let provider = ScriptedProvider::from_text(happy_response());
    let result = compact_conversation(
        &broken,
        &provider,
        "m",
        None,
        PartialCompactDirection::Full,
        AbortController::default(),
    )
    .await
    .expect("compact should succeed");
    apply_compaction_to_store(&mut store, &result).unwrap();
    let after: Vec<Message> = store.iter().cloned().collect();
    assert_paired(&after, "pre-orphaned store Full compact");
}

#[test]
fn repair_store_pairing_fixes_severed_pairs_in_place() {
    // A loaded transcript damaged by an OLDER compaction: the tool_use
    // half is tombstoned, the result half survives (orphan), and a later
    // dangling tool_use has no result. repair_store_pairing must fix both
    // in place, preserving UUIDs.
    let mut store = MessageStore::new();
    store.push(user("go")).unwrap();
    let use_header = Header::new();
    // t1's tool_use was tombstoned by an OLDER compaction — its result
    // survives as an orphan.
    store
        .push(Message::Tombstone {
            header: use_header.clone(),
            reason: "compacted".into(),
        })
        .unwrap();
    store.push(user_tool_result("t1")).unwrap();
    store.push(assistant_tool_use("t2")).unwrap(); // dangling — no result
    store.push(user("next")).unwrap();

    let rewritten = agent::compact::repair_store_pairing(&mut store);
    assert_eq!(rewritten, 2, "orphan result + dangling tool_use repaired");

    let after: Vec<Message> = store.iter().cloned().collect();
    // The orphan result became a text block.
    match &after[2] {
        Message::User { content, .. } => {
            assert!(matches!(content[0], ContentBlock::Text { .. }));
        }
        other => panic!("expected user message, got {other:?}"),
    }
    // The dangling tool_use became a text block.
    match &after[3] {
        Message::Assistant { content, .. } => {
            assert!(matches!(content[0], ContentBlock::Text { .. }));
        }
        other => panic!("expected assistant message, got {other:?}"),
    }
    // UUIDs survived the rebuild.
    assert_eq!(after[1].uuid(), use_header.uuid);
    assert_paired(&after, "repaired store");
}

#[test]
fn repair_store_pairing_keeps_healthy_store_untouched() {
    let mut store = MessageStore::new();
    store.push(user("go")).unwrap();
    store.push(assistant_tool_use("t1")).unwrap();
    store.push(user_tool_result("t1")).unwrap();
    store.push(assistant("done")).unwrap();
    let before: Vec<Message> = store.iter().cloned().collect();
    let rewritten = agent::compact::repair_store_pairing(&mut store);
    assert_eq!(rewritten, 0);
    let after: Vec<Message> = store.iter().cloned().collect();
    assert_eq!(before.len(), after.len());
    assert!(before.iter().zip(&after).all(|(a, b)| a.uuid() == b.uuid()));
}
