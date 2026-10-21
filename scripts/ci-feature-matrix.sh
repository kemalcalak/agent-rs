#!/bin/bash
set -e
for f in "" anthropic openai ollama mcp session-jsonl swarm tiktoken full; do
  if [ -z "$f" ]; then
    echo "=== agent: no-default-features (none) ==="
    cargo check -q -p agent --no-default-features 2>&1 | tail -3
  else
    echo "=== agent: no-default-features --features $f ==="
    cargo check -q -p agent --no-default-features --features "$f" 2>&1 | tail -3
  fi
done
for f in "" fs search shell bash-async web web-search task todo notebook all; do
  if [ -z "$f" ]; then
    echo "=== agent-tools-code: no-default-features (none) ==="
    cargo check -q -p agent-tools-code --no-default-features 2>&1 | tail -3
  else
    echo "=== agent-tools-code: no-default-features --features $f ==="
    cargo check -q -p agent-tools-code --no-default-features --features "$f" 2>&1 | tail -3
  fi
done
echo "ALL FEATURE COMBOS DONE"
