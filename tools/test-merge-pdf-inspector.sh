#!/usr/bin/env bash
set -euo pipefail

helper="$(cd "$(dirname "$0")" && pwd -P)/merge-pdf-inspector.sh"
temp_parent="$(cd "${TMPDIR:-/tmp}" && pwd -P)"
scratch="$(mktemp -d "$temp_parent/pdf-inspector-merge.XXXXXX")"
[[ "$scratch" == "$temp_parent/pdf-inspector-merge."* ]] || exit 1
trap 'rm -rf -- "$scratch"' EXIT
repo="$scratch/inspector"
guide="$scratch/fork-guide.md"
report="$scratch/report.md"
git init -q -b upstream "$repo"
git -C "$repo" config user.name 'Merge test'
git -C "$repo" config user.email 'merge-test@example.invalid'
git -C "$repo" config core.autocrlf false
printf 'Base guidance\n' > "$repo/AGENTS.md"
printf 'Base engine\n' > "$repo/engine.txt"
printf '[package]\nversion = "1.0.0"\n' > "$repo/Cargo.toml"
git -C "$repo" add -- AGENTS.md engine.txt Cargo.toml
git -C "$repo" commit -qm base
base="$(git -C "$repo" rev-parse HEAD)"

# The fork shares upstream's tree but not its history, as after a history rewrite.
git -C "$repo" switch -q --orphan fork
git -C "$repo" checkout -q "$base" -- .
printf 'commit %s\nversion 1.0.0\n' "$base" > "$repo/.upstream-pdf-inspector"
printf 'Prior fork guidance\n' > "$repo/AGENTS.md"
git -C "$repo" add -- .upstream-pdf-inspector AGENTS.md
git -C "$repo" commit -qm 'Fork guidance'
git -C "$repo" switch -q upstream
printf 'Upstream private infrastructure\n' > "$repo/AGENTS.md"
printf 'Upstream capability\n' > "$repo/upstream.txt"
printf '[package]\nversion = "1.1.0"\n' > "$repo/Cargo.toml"
git -C "$repo" add -- AGENTS.md upstream.txt Cargo.toml
git -C "$repo" commit -qm 'Upstream capability and guidance'
upstream="$(git -C "$repo" rev-parse HEAD)"
git -C "$repo" switch -q fork

printf 'Canonical fork guidance\n' > "$guide"
previous="$(git -C "$repo" rev-parse HEAD)"
bash "$helper" "$repo" upstream "$guide"
cmp "$guide" "$repo/AGENTS.md"
test -f "$repo/upstream.txt"
test "$(git -C "$repo" rev-parse 'HEAD^@')" = "$previous"
if git -C "$repo" merge-base --is-ancestor upstream HEAD; then exit 1; fi
test "$(cat "$repo/.upstream-pdf-inspector")" = "$(printf 'commit %s\nversion 1.1.0' "$upstream")"
git -C "$repo" log -1 --format=%B | grep -qx "Upstream-PDF-Inspector: $upstream (1.1.0)"
test -z "$(git -C "$repo" status --porcelain)"

# Policy updates also apply when no new upstream code exists, without no-op commits.
printf 'Updated fork guidance\n' > "$guide"
bash "$helper" "$repo" upstream "$guide"
cmp "$guide" "$repo/AGENTS.md"
accepted="$(git -C "$repo" rev-parse HEAD)"
bash "$helper" "$repo" upstream "$guide"
test "$accepted" = "$(git -C "$repo" rev-parse HEAD)"

# Guidance ownership must never hide a source conflict or overwrite local work.
git -C "$repo" switch -q upstream
printf 'Upstream engine change\n' > "$repo/engine.txt"
git -C "$repo" commit -qam 'Upstream engine'
git -C "$repo" switch -q fork
printf 'Fork engine change\n' > "$repo/engine.txt"
git -C "$repo" commit -qam 'Fork engine'
accepted="$(git -C "$repo" rev-parse HEAD)"
if bash "$helper" "$repo" upstream "$guide" "$report"; then exit 1; fi
grep -qx engine.txt "$report"
test "$accepted" = "$(git -C "$repo" rev-parse HEAD)"
test -z "$(git -C "$repo" status --porcelain)"
cmp "$guide" "$repo/AGENTS.md"
printf 'Uncommitted work\n' > "$repo/engine.txt"
if bash "$helper" "$repo" upstream "$guide"; then exit 1; fi
test "$(cat "$repo/engine.txt")" = 'Uncommitted work'
git -C "$repo" checkout -q -- engine.txt

# Personal home paths in upstream files never enter the fork.
git -C "$repo" switch -q upstream
git -C "$repo" reset -q --hard HEAD~1
printf 'corpus = "%s/evals"\n' "/Users/""private-person" > "$repo/config.txt"
git -C "$repo" add -- config.txt
git -C "$repo" commit -qm 'Upstream personal path'
git -C "$repo" switch -q fork
if bash "$helper" "$repo" upstream "$guide" "$report"; then exit 1; fi
grep -q 'config.txt$' "$report"
test "$accepted" = "$(git -C "$repo" rev-parse HEAD)"

# A branch that regained upstream ancestry is refused rather than extended.
git -C "$repo" merge -q --allow-unrelated-histories -s ours --no-edit upstream
if bash "$helper" "$repo" upstream "$guide" "$report"; then exit 1; fi
echo 'PASS: squash sync keeps fork guidance and upstream code without upstream history; conflicts, personal paths and local work are protected.'
