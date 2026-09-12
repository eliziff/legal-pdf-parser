#!/usr/bin/env bash
set -euo pipefail

repo="$1"
upstream="$2"
guide="$3"
test -f "$guide"
if test -n "$(git -C "$repo" status --porcelain)"; then
  echo 'PDF Inspector merge requires a clean candidate worktree.' >&2
  exit 1
fi

if ! git -C "$repo" merge-base --is-ancestor "$upstream" HEAD; then
  if ! git -C "$repo" merge --no-commit --no-ff "$upstream"; then
    git -C "$repo" rev-parse --verify -q MERGE_HEAD >/dev/null || exit 1
  fi
fi

# Agent guidance belongs to this fork, including when upstream edits it cleanly.
cp "$guide" "$repo/AGENTS.md"
git -C "$repo" add -- AGENTS.md
if test -n "$(git -C "$repo" diff --name-only --diff-filter=U)"; then
  git -C "$repo" merge --abort
  echo 'PDF Inspector source conflicts require resolution before gating.' >&2
  exit 1
fi
if git -C "$repo" rev-parse --verify -q MERGE_HEAD >/dev/null; then
  git -C "$repo" commit --no-edit
elif ! git -C "$repo" diff --cached --quiet; then
  git -C "$repo" commit -m 'Maintain fork agent guidance'
fi
