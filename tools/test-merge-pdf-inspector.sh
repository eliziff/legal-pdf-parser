#!/usr/bin/env bash
set -euo pipefail

helper="$(cd "$(dirname "$0")" && pwd -P)/merge-pdf-inspector.sh"
temp_parent="$(cd "${TMPDIR:-/tmp}" && pwd -P)"
scratch="$(mktemp -d "$temp_parent/pdf-inspector-merge.XXXXXX")"
[[ "$scratch" == "$temp_parent/pdf-inspector-merge."* ]] || exit 1
trap 'rm -rf -- "$scratch"' EXIT
repo="$scratch/inspector"
guide="$scratch/fork-guide.md"
git init -q -b main "$repo"
git -C "$repo" config user.name 'Merge test'
git -C "$repo" config user.email 'merge-test@example.invalid'
git -C "$repo" config core.autocrlf false
printf 'Base guidance\n' > "$repo/AGENTS.md"
printf 'Base engine\n' > "$repo/engine.txt"
git -C "$repo" add -- AGENTS.md engine.txt
git -C "$repo" commit -qm base
git -C "$repo" branch upstream
git -C "$repo" switch -qc fork
printf 'Prior fork guidance\n' > "$repo/AGENTS.md"
git -C "$repo" commit -qam 'Fork guidance'
git -C "$repo" switch -q upstream
printf 'Upstream private infrastructure\n' > "$repo/AGENTS.md"
printf 'Upstream capability\n' > "$repo/upstream.txt"
git -C "$repo" add -- AGENTS.md upstream.txt
git -C "$repo" commit -qm 'Upstream capability and guidance'
git -C "$repo" switch -q fork

printf 'Canonical fork guidance\n' > "$guide"
bash "$helper" "$repo" upstream "$guide"
cmp "$guide" "$repo/AGENTS.md"
test -f "$repo/upstream.txt"
git -C "$repo" merge-base --is-ancestor upstream HEAD
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
if bash "$helper" "$repo" upstream "$guide"; then exit 1; fi
test "$accepted" = "$(git -C "$repo" rev-parse HEAD)"
test -z "$(git -C "$repo" status --porcelain)"
cmp "$guide" "$repo/AGENTS.md"
printf 'Uncommitted work\n' > "$repo/engine.txt"
if bash "$helper" "$repo" upstream "$guide"; then exit 1; fi
test "$(cat "$repo/engine.txt")" = 'Uncommitted work'
echo 'PASS: fork guidance survives merges; upstream code is retained; source conflicts and local work are protected.'
