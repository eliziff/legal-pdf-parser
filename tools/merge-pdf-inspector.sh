#!/usr/bin/env bash
# Squash-sync upstream PDF Inspector onto the fork branch checked out in <repo>.
# The fork never carries upstream ancestry. The upstream commit it last
# incorporated is recorded in .upstream-pdf-inspector and is the merge base, so
# each sync is one single-parent commit and upstream history is never published.
set -euo pipefail

repo="$1"
upstream="$(git -C "$repo" rev-parse --verify "$2^{commit}")"
guide="$3"
report="${4:-/dev/stderr}"
record=.upstream-pdf-inspector
test -f "$guide"
fail() {
  printf '%s\n' "$@" > "$report"
  echo "PDF Inspector sync stopped: $1" >&2
  exit 1
}
if test -n "$(git -C "$repo" status --porcelain)"; then
  echo 'PDF Inspector sync requires a clean candidate worktree.' >&2
  exit 1
fi
if git -C "$repo" merge-base --is-ancestor "$upstream" HEAD; then
  fail 'The fork branch already carries upstream ancestry; it may only take squashed upstream trees.'
fi
base="$(git -C "$repo" show "HEAD:$record" 2>/dev/null | sed -n 's/^commit //p')"
test -n "$base" || fail "HEAD records no upstream base in $record."
git -C "$repo" cat-file -e "$base^{commit}" || fail "Recorded upstream base $base is not available."
version="$(git -C "$repo" show "$upstream:Cargo.toml" | sed -n 's/^version = "\(.*\)"$/\1/p' | head -n 1)"

tree="$(git -C "$repo" rev-parse 'HEAD^{tree}')"
if test "$base" != "$upstream"; then
  status=0
  merge="$(git -C "$repo" merge-tree --write-tree --name-only --merge-base="$base" HEAD "$upstream")" || status=$?
  test "$status" -le 1 || exit "$status"
  tree="$(sed -n 1p <<<"$merge")"
  # Agent guidance belongs to this fork, so only a guidance conflict is resolvable here.
  conflicts="$(sed -n '2,/^$/p' <<<"$merge" | sed '/^$/d' | sort -u | grep -vx AGENTS.md || true)"
  if test -n "$conflicts"; then
    fail "Upstream $upstream ($version) conflicts with the fork from recorded base $base." '' \
      'Conflicting paths:' "$conflicts" '' 'Merge messages:' "$(sed '1,/^$/d' <<<"$merge")" '' \
      'Resolve on a pdf-inspector checkout with upstream fetched, then commit one single-parent commit:' \
      "  git diff --binary $base $upstream | git apply --3way" \
      "  printf 'commit $upstream\\nversion $version\\n' > $record"
  fi
fi

index="$(git -C "$repo" rev-parse --absolute-git-dir)/pdf-inspector-sync.index"
trap 'rm -f "$index"' EXIT
guide_blob="$(git -C "$repo" hash-object -w --path=AGENTS.md --stdin < "$guide")"
record_blob="$(printf 'commit %s\nversion %s\n' "$upstream" "$version" | git -C "$repo" hash-object -w --stdin)"
GIT_INDEX_FILE="$index" git -C "$repo" read-tree "$tree"
GIT_INDEX_FILE="$index" git -C "$repo" update-index --add \
  --cacheinfo "100644,$guide_blob,AGENTS.md" --cacheinfo "100644,$record_blob,$record"
tree="$(GIT_INDEX_FILE="$index" git -C "$repo" write-tree)"

# Personal home paths never enter the branch, from upstream files or anywhere else.
home='(?i)(?:[a-z]:[\\/]+Users[\\/]+|/mnt/[a-z]/Users/|(?<![a-z0-9_.-])/(?:Users|home)/)(?!(?:runner|runneradmin|sandbox|build|Public|Shared|Default|user|someone)(?:[\\/]|$))[a-z0-9_.@-]+[\\/]'
if personal="$(git -C "$repo" grep -I -l -P -e "$home" "$tree")"; then
  fail "Upstream $upstream ($version) would add personal home paths." '' 'Files:' "$personal" '' \
    'Replace them with relative paths or runtime configuration before syncing.'
fi

test "$tree" != "$(git -C "$repo" rev-parse 'HEAD^{tree}')" || exit 0
if test "$base" = "$upstream"; then subject='Maintain fork agent guidance'; else subject="Sync PDF Inspector $version from upstream"; fi
commit="$(printf '%s\n\nUpstream-PDF-Inspector: %s (%s)\n' "$subject" "$upstream" "$version" |
  git -C "$repo" commit-tree "$tree" -p HEAD)"
git -C "$repo" merge --ff-only -q "$commit"
