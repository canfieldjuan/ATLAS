#!/usr/bin/env bash
# Owned-PR watcher (docs/OVERNIGHT_ARC_WORKFLOW.md section 5).
# Watches ONE PR the current builder session owns and exits the moment there
# is something to act on:
#   MERGED/CLOSED - PR reached terminal state (stop watching)
#   HEAD-MOVED    - branch advanced past the SHA this watch was armed on
#   ACTIONABLE    - red required context / unresolved review threads /
#                   CHANGES_REQUESTED review decision
#   MERGE-READY   - EVERY required context present and success +
#                   Codex review attestation exists on this exact head SHA +
#                   0 unresolved threads (no unfetched pages) +
#                   review decision not CHANGES_REQUESTED + mergeable
# Required contexts are read from origin/main's ci/gates.yml and the app pin is
# read from origin/main's scripts/check_required_status_checks.py (trusted ref
# -- the watched branch cannot weaken its own gate); MERGE-READY requires their
# PRESENCE with success and no unsettled (queued/rerunning) required run, so a
# context that has not started or is rerunning fails closed.
# The watcher never merges and holds no merge authority (AGENTS.md 3c.1.1);
# it reports state and exits fast so the builder session acts.
#
# Usage: PR=<number> SHA=<full 40-char head sha> bash scripts/watch_owned_pr.sh
# Optional: REPO=<owner/name> (default: derived from origin), CYCLES=<n> (default 32, ~16h).
set -uo pipefail
PR="${PR:?set PR=<number>}"
SHA="${SHA:?set SHA=<full 40-char head sha this watch is armed on>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="${REPO:-$(git -C "$ROOT" remote get-url origin 2>/dev/null | sed -E 's#^(git@github\.com:|https://github\.com/)##; s#\.git$##')}"
[ -n "$REPO" ] || { echo "cannot derive REPO from origin; set REPO=<owner/name>" >&2; exit 2; }
OWNER="${REPO%%/*}"; NAME="${REPO##*/}"
CYCLES="${CYCLES:-32}"
TOK="$(grep -m1 '^GITHUB_ACCESS_TOKEN=' "$ROOT/.env" 2>/dev/null | cut -d= -f2-)"
[ -n "$TOK" ] || TOK="${GH_TOKEN:-}"
[ -n "$TOK" ] || { echo "no GITHUB_ACCESS_TOKEN in $ROOT/.env and no GH_TOKEN set" >&2; exit 2; }
# Canonical required contexts (branch protection), read from the TRUSTED ref
# (origin/main), never from the watched branch's working tree -- a PR that
# edits ci/gates.yml or check_required_status_checks.py must not be able to
# weaken its own gate. When a trusted registry is available, parse it through
# the trusted checker implementation and fail closed on parser errors. Falls
# back to the documented legacy four only when no registry exists yet.
GATES_SRC="$(git -C "$ROOT" show origin/main:ci/gates.yml 2>/dev/null)"
[ -n "$GATES_SRC" ] || GATES_SRC="$(cat "$ROOT/ci/gates.yml" 2>/dev/null)"
CHECKER_SRC="$(git -C "$ROOT" show origin/main:scripts/check_required_status_checks.py 2>/dev/null)"
[ -n "$CHECKER_SRC" ] || CHECKER_SRC="$(cat "$ROOT/scripts/check_required_status_checks.py" 2>/dev/null)"
REQ_CONTEXTS=()
if [ -n "$GATES_SRC" ]; then
  CHECKER_TMP="$(mktemp --suffix=.py)"
  GATES_TMP="$(mktemp)"
  REQ_CONTEXTS_TMP="$(mktemp)"
  REQ_CONTEXTS_ERR="$(mktemp)"
  printf '%s\n' "$CHECKER_SRC" > "$CHECKER_TMP"
  printf '%s\n' "$GATES_SRC" > "$GATES_TMP"
  if ! python3 - "$CHECKER_TMP" "$GATES_TMP" > "$REQ_CONTEXTS_TMP" 2>"$REQ_CONTEXTS_ERR" <<'PY'
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

checker_path = Path(sys.argv[1])
registry_path = Path(sys.argv[2])
spec = importlib.util.spec_from_file_location(
    "trusted_check_required_status_checks",
    checker_path,
)
if spec is None or spec.loader is None:
    raise SystemExit("could not load trusted required-status checker")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
if not hasattr(checker, "default_required_contexts"):
    raise SystemExit("trusted required-status checker has no registry parser")
for context in checker.default_required_contexts(registry_path):
    print(context)
PY
  then
    echo "watch_owned_pr.sh: failed to parse trusted ci/gates.yml:" >&2
    sed 's/^/  /' "$REQ_CONTEXTS_ERR" >&2
    rm -f "$CHECKER_TMP" "$GATES_TMP" "$REQ_CONTEXTS_TMP" "$REQ_CONTEXTS_ERR"
    exit 2
  fi
  mapfile -t REQ_CONTEXTS < "$REQ_CONTEXTS_TMP"
  rm -f "$CHECKER_TMP" "$GATES_TMP" "$REQ_CONTEXTS_TMP" "$REQ_CONTEXTS_ERR"
fi
if [ "${#REQ_CONTEXTS[@]}" -eq 0 ]; then
  REQ_CONTEXTS=("live-reconciliation" "diff-budget" "Gitleaks PR secret scan" "Gitleaks baseline growth guard")
fi
REQ_JSON=$(printf '%s\n' "${REQ_CONTEXTS[@]}" | jq -R . | jq -cs .)
REQ_TOTAL=${#REQ_CONTEXTS[@]}
CODEX_LOGINS_JSON='["chatgpt-codex-connector","chatgpt-codex-connector[bot]"]'
# Required contexts are pinned to the GitHub Actions app (same pin as
# check_required_status_checks.py) so a same-named check published by any
# other app can neither green nor red the required gate.
REQ_APP_ID=$(printf '%s\n' "$CHECKER_SRC" \
  | grep -oE '^GITHUB_ACTIONS_APP_ID = [0-9]+' | grep -oE '[0-9]+')
[ -n "$REQ_APP_ID" ] || REQ_APP_ID=15368
echo "owned-pr watcher armed: $REPO#$PR @ ${SHA:0:9} $(date '+%F %H:%M') (required contexts: ${REQ_CONTEXTS[*]})"
for i in $(seq 0 "$CYCLES"); do
  [ "$i" -gt 0 ] && sleep 1740
  CUR=$(GH_TOKEN="$TOK" gh api "repos/$REPO/pulls/$PR" --jq '.head.sha' 2>/dev/null) || { echo "cycle $i: API error, retrying"; continue; }
  if [ "$CUR" != "$SHA" ]; then echo "HEAD-MOVED: ${SHA:0:9} -> ${CUR:0:9} (new push; reconcile + re-arm on new head)"; exit 0; fi
  THREAD_QUERY='query($owner:String!,$name:String!,$pr:Int!){ repository(owner:$owner,name:$name){ pullRequest(number:$pr){ state merged mergeable mergeStateStatus reviewDecision reviewThreads(first:100){ pageInfo{ hasNextPage } nodes{ isResolved isOutdated comments(first:1){ nodes{ author{ login } } } } } } } }'
  ST=$(GH_TOKEN="$TOK" gh api graphql -f query="$THREAD_QUERY" -f owner="$OWNER" -f name="$NAME" -F pr="$PR" 2>/dev/null)
  if ! echo "$ST" | jq -e '
      (((.errors // []) | length) == 0)
      and ((.data.repository.pullRequest | type) == "object")
      and ((.data.repository.pullRequest.reviewThreads | type) == "object")
      and ((.data.repository.pullRequest.reviewThreads.nodes | type) == "array")
      and ((.data.repository.pullRequest.reviewThreads.pageInfo | type) == "object")
      and ((.data.repository.pullRequest.reviewThreads.pageInfo.hasNextPage | type) == "boolean")
    ' >/dev/null 2>&1; then
    echo "cycle $i: GraphQL reviewThreads snapshot incomplete/malformed, retrying"
    continue
  fi
  STATE=$(echo "$ST" | jq -r '.data.repository.pullRequest | .state + (if .merged then "/merged" else "" end)')
  MERGEABLE=$(echo "$ST" | jq -r '.data.repository.pullRequest.mergeable')
  MSTATE=$(echo "$ST" | jq -r '.data.repository.pullRequest.mergeStateStatus // "UNKNOWN"')
  DECISION=$(echo "$ST" | jq -r '.data.repository.pullRequest.reviewDecision // "NONE"')
  UNRES=$(echo "$ST" | jq --argjson codex "$CODEX_LOGINS_JSON" '[.data.repository.pullRequest.reviewThreads.nodes[]? | select((.isResolved==false) and ((((.comments.nodes[0].author.login // "") | ascii_downcase) as $login | $codex | index($login)) != null))] | length')
  MORE=$(echo "$ST" | jq -r '.data.repository.pullRequest.reviewThreads.pageInfo.hasNextPage')
  # Fail closed when more thread pages exist than we fetched.
  [ "$MORE" = "true" ] && UNRES="${UNRES}+unfetched-pages"
  REVIEW_QUERY='query($owner:String!,$name:String!,$pr:Int!,$cursor:String){ repository(owner:$owner,name:$name){ pullRequest(number:$pr){ reviews(first:100, after:$cursor){ pageInfo{ hasNextPage endCursor } nodes{ author{ login } commit{ oid } state } } } } }'
  COMMENT_QUERY='query($owner:String!,$name:String!,$pr:Int!,$cursor:String){ repository(owner:$owner,name:$name){ pullRequest(number:$pr){ comments(first:100, after:$cursor){ pageInfo{ hasNextPage endCursor } nodes{ author{ login } body bodyText } } } } }'
  REVIEW_NODES='[]'
  REVIEW_CURSOR=''
  REVIEW_PAGES=0
  REVIEWS_COMPLETE=true
  while :; do
    REVIEW_ARGS=(gh api graphql -f query="$REVIEW_QUERY" -f owner="$OWNER" -f name="$NAME" -F pr="$PR")
    [ -n "$REVIEW_CURSOR" ] && REVIEW_ARGS+=(-f cursor="$REVIEW_CURSOR")
    REVIEW_PAGE=$(GH_TOKEN="$TOK" "${REVIEW_ARGS[@]}" 2>/dev/null) || { REVIEWS_COMPLETE=false; break; }
    if ! echo "$REVIEW_PAGE" | jq -e '
        (((.errors // []) | length) == 0)
        and ((.data.repository.pullRequest | type) == "object")
        and ((.data.repository.pullRequest.reviews | type) == "object")
        and ((.data.repository.pullRequest.reviews.nodes | type) == "array")
        and ((.data.repository.pullRequest.reviews.pageInfo | type) == "object")
        and ((.data.repository.pullRequest.reviews.pageInfo.hasNextPage | type) == "boolean")
        and (
          (.data.repository.pullRequest.reviews.pageInfo.hasNextPage == false)
          or (
            ((.data.repository.pullRequest.reviews.pageInfo.endCursor // "") | type) == "string"
            and (((.data.repository.pullRequest.reviews.pageInfo.endCursor // "") | length) > 0)
          )
        )
      ' >/dev/null 2>&1; then
      REVIEWS_COMPLETE=false
      break
    fi
    PAGE_NODES=$(echo "$REVIEW_PAGE" | jq -c '.data.repository.pullRequest.reviews.nodes // []') || { REVIEWS_COMPLETE=false; break; }
    REVIEW_NODES=$(jq -n -c --argjson existing "$REVIEW_NODES" --argjson new "$PAGE_NODES" '$existing + $new') || { REVIEWS_COMPLETE=false; break; }
    REVIEW_PAGES=$((REVIEW_PAGES + 1))
    HAS_NEXT=$(echo "$REVIEW_PAGE" | jq -r '.data.repository.pullRequest.reviews.pageInfo.hasNextPage')
    [ "$HAS_NEXT" = "true" ] || break
    REVIEW_CURSOR=$(echo "$REVIEW_PAGE" | jq -r '.data.repository.pullRequest.reviews.pageInfo.endCursor // empty')
    [ -n "$REVIEW_CURSOR" ] || { REVIEWS_COMPLETE=false; break; }
  done
  COMMENT_NODES='[]'
  COMMENT_CURSOR=''
  while [ "$REVIEWS_COMPLETE" = "true" ]; do
    COMMENT_ARGS=(gh api graphql -f query="$COMMENT_QUERY" -f owner="$OWNER" -f name="$NAME" -F pr="$PR")
    [ -n "$COMMENT_CURSOR" ] && COMMENT_ARGS+=(-f cursor="$COMMENT_CURSOR")
    COMMENT_PAGE=$(GH_TOKEN="$TOK" "${COMMENT_ARGS[@]}" 2>/dev/null) || { REVIEWS_COMPLETE=false; break; }
    if ! echo "$COMMENT_PAGE" | jq -e '
        (((.errors // []) | length) == 0)
        and ((.data.repository.pullRequest | type) == "object")
        and ((.data.repository.pullRequest.comments | type) == "object")
        and ((.data.repository.pullRequest.comments.nodes | type) == "array")
        and ((.data.repository.pullRequest.comments.pageInfo | type) == "object")
        and ((.data.repository.pullRequest.comments.pageInfo.hasNextPage | type) == "boolean")
        and (
          (.data.repository.pullRequest.comments.pageInfo.hasNextPage == false)
          or (
            ((.data.repository.pullRequest.comments.pageInfo.endCursor // "") | type) == "string"
            and (((.data.repository.pullRequest.comments.pageInfo.endCursor // "") | length) > 0)
          )
        )
      ' >/dev/null 2>&1; then
      REVIEWS_COMPLETE=false
      break
    fi
    PAGE_NODES=$(echo "$COMMENT_PAGE" | jq -c '.data.repository.pullRequest.comments.nodes // []') || { REVIEWS_COMPLETE=false; break; }
    COMMENT_NODES=$(jq -n -c --argjson existing "$COMMENT_NODES" --argjson new "$PAGE_NODES" '$existing + $new') || { REVIEWS_COMPLETE=false; break; }
    REVIEW_PAGES=$((REVIEW_PAGES + 1))
    HAS_NEXT=$(echo "$COMMENT_PAGE" | jq -r '.data.repository.pullRequest.comments.pageInfo.hasNextPage')
    [ "$HAS_NEXT" = "true" ] || break
    COMMENT_CURSOR=$(echo "$COMMENT_PAGE" | jq -r '.data.repository.pullRequest.comments.pageInfo.endCursor // empty')
    [ -n "$COMMENT_CURSOR" ] || { REVIEWS_COMPLETE=false; break; }
  done
  CODEX_FORMAL_REVIEWS=$(echo "$REVIEW_NODES" | jq --arg sha "$SHA" --argjson codex "$CODEX_LOGINS_JSON" '[.[]? | select(((((.author.login // "") | ascii_downcase) as $login | $codex | index($login)) != null) and ((.commit.oid // "") == $sha) and ((.state // "") | IN("COMMENTED","APPROVED")))] | length')
  CODEX_CLEAN_COMMENTS=$(echo "$COMMENT_NODES" | jq --arg sha "$SHA" --argjson codex "$CODEX_LOGINS_JSON" '[.[]? | ((.body // .bodyText // "") as $body | ((.author.login // "") | ascii_downcase) as $login | select(($codex | index($login)) != null) | select(($body | ascii_downcase | contains("didn'\''t find any major issues"))) | ((try ($body | capture("\\*\\*Reviewed commit:\\*\\*\\s*`(?<reviewed>[0-9a-fA-F]{10,40})`").reviewed) catch "") | ascii_downcase) as $reviewed | select(($reviewed | length) > 0 and ($sha | startswith($reviewed))))] | length')
  CODEX_HEAD_REVIEWS=$((CODEX_FORMAL_REVIEWS + CODEX_CLEAN_COMMENTS))
  # --paginate + re-wrap: required contexts beyond the first 100 runs stay visible
  CR=$(GH_TOKEN="$TOK" gh api --paginate "repos/$REPO/commits/$SHA/check-runs?per_page=100" 2>/dev/null | jq -s '{check_runs:[.[].check_runs[]]}')
  PEND=$(echo "$CR" | jq '[.check_runs[]|select(.status!="completed")]|length')
  # App-pin BEFORE picking the latest run per name, so a same-named run from
  # another app can neither green the gate nor mask the genuine run.
  REQLATEST=$(echo "$CR" | jq --argjson app "$REQ_APP_ID" '[.check_runs[]|select(.app.id==$app)]|group_by(.name)|map(sort_by(.started_at)|last)')
  REQRED=$(echo "$REQLATEST" | jq --argjson req "$REQ_JSON" '[.[]|select(.name as $n|$req|index($n))|select(.status=="completed" and (.conclusion|IN("failure","cancelled","timed_out","action_required","stale","startup_failure")))]|length')
  # Green mirrors branch-protection semantics: neutral/skipped required checks
  # count as passing on the server, so the advisory gate matches the enforcer.
  REQGREEN=$(echo "$REQLATEST" | jq --argjson req "$REQ_JSON" '[.[]|select(.name as $n|$req|index($n))|select(.status=="completed" and (.conclusion|IN("success","neutral","skipped")))]|length')
  # A rerun of a required check creates a fresh queued run beside the old
  # completed one; ANY not-completed run of a required name (across all runs,
  # not just the latest-pick) blocks readiness until it settles.
  REQUNSETTLED=$(echo "$CR" | jq --argjson app "$REQ_APP_ID" --argjson req "$REQ_JSON" '[.check_runs[]|select(.app.id==$app)|select(.name as $n|$req|index($n))|select(.status!="completed")]|length')
  echo "cycle $i $(date +%H:%M): state=$STATE req-green=$REQGREEN/$REQ_TOTAL req-red=$REQRED req-unsettled=$REQUNSETTLED pending=$PEND codex-head-attestations=$CODEX_HEAD_REVIEWS attestation-pages=$REVIEW_PAGES attestations-complete=$REVIEWS_COMPLETE threads=$UNRES decision=$DECISION mergeable=$MERGEABLE merge-state=$MSTATE"
  case "$STATE" in MERGED/merged|CLOSED) echo "TERMINAL: PR $STATE"; exit 0;; esac
  # Definite negatives are actionable on ANY cycle, including the first.
  if [ "$REQRED" -gt 0 ] || [ "$UNRES" != "0" ] || [ "$DECISION" = "CHANGES_REQUESTED" ] || [ "$REVIEWS_COMPLETE" != "true" ]; then
    echo "ACTIONABLE: req-red=$REQRED codex-head-attestations=$CODEX_HEAD_REVIEWS threads=$UNRES decision=$DECISION -> reconcile/fix, push, re-arm"; exit 0
  fi
  # Readiness is presence-based: every required context must be reporting
  # success (a not-yet-started context keeps this false, so no early race).
  if [ "$REQGREEN" -eq "$REQ_TOTAL" ] && [ "$REQUNSETTLED" -eq 0 ] && [ "$CODEX_HEAD_REVIEWS" -gt 0 ] && [ "$MERGEABLE" = "MERGEABLE" ] \
     && { [ "$MSTATE" = "CLEAN" ] || [ "$MSTATE" = "UNSTABLE" ]; }; then
    CUR=$(GH_TOKEN="$TOK" gh api "repos/$REPO/pulls/$PR" --jq '.head.sha' 2>/dev/null) || { echo "cycle $i: API error before readiness, retrying"; continue; }
    if [ "$CUR" != "$SHA" ]; then echo "HEAD-MOVED: ${SHA:0:9} -> ${CUR:0:9} (new push; reconcile + re-arm on new head)"; exit 0; fi
    FINAL_ST=$(GH_TOKEN="$TOK" gh api graphql -f query="$THREAD_QUERY" -f owner="$OWNER" -f name="$NAME" -F pr="$PR" 2>/dev/null)
    if ! echo "$FINAL_ST" | jq -e '
        (((.errors // []) | length) == 0)
        and ((.data.repository.pullRequest | type) == "object")
        and ((.data.repository.pullRequest.reviewThreads | type) == "object")
        and ((.data.repository.pullRequest.reviewThreads.nodes | type) == "array")
        and ((.data.repository.pullRequest.reviewThreads.pageInfo | type) == "object")
        and ((.data.repository.pullRequest.reviewThreads.pageInfo.hasNextPage | type) == "boolean")
      ' >/dev/null 2>&1; then
      echo "cycle $i: final GraphQL reviewThreads snapshot incomplete/malformed, retrying"
      continue
    fi
    FINAL_DECISION=$(echo "$FINAL_ST" | jq -r '.data.repository.pullRequest.reviewDecision // "NONE"')
    FINAL_MERGEABLE=$(echo "$FINAL_ST" | jq -r '.data.repository.pullRequest.mergeable')
    FINAL_MSTATE=$(echo "$FINAL_ST" | jq -r '.data.repository.pullRequest.mergeStateStatus // "UNKNOWN"')
    FINAL_UNRES=$(echo "$FINAL_ST" | jq --argjson codex "$CODEX_LOGINS_JSON" '[.data.repository.pullRequest.reviewThreads.nodes[]? | select((.isResolved==false) and ((((.comments.nodes[0].author.login // "") | ascii_downcase) as $login | $codex | index($login)) != null))] | length')
    FINAL_MORE=$(echo "$FINAL_ST" | jq -r '.data.repository.pullRequest.reviewThreads.pageInfo.hasNextPage')
    [ "$FINAL_MORE" = "true" ] && FINAL_UNRES="${FINAL_UNRES}+unfetched-pages"
    FINAL_REVIEW_NODES='[]'
    FINAL_REVIEW_CURSOR=''
    FINAL_REVIEW_PAGES=0
    FINAL_REVIEWS_COMPLETE=true
    while :; do
      FINAL_REVIEW_ARGS=(gh api graphql -f query="$REVIEW_QUERY" -f owner="$OWNER" -f name="$NAME" -F pr="$PR")
      [ -n "$FINAL_REVIEW_CURSOR" ] && FINAL_REVIEW_ARGS+=(-f cursor="$FINAL_REVIEW_CURSOR")
      FINAL_REVIEW_PAGE=$(GH_TOKEN="$TOK" "${FINAL_REVIEW_ARGS[@]}" 2>/dev/null) || { FINAL_REVIEWS_COMPLETE=false; break; }
      if ! echo "$FINAL_REVIEW_PAGE" | jq -e '
          (((.errors // []) | length) == 0)
          and ((.data.repository.pullRequest | type) == "object")
          and ((.data.repository.pullRequest.reviews | type) == "object")
          and ((.data.repository.pullRequest.reviews.nodes | type) == "array")
          and ((.data.repository.pullRequest.reviews.pageInfo | type) == "object")
          and ((.data.repository.pullRequest.reviews.pageInfo.hasNextPage | type) == "boolean")
          and (
            (.data.repository.pullRequest.reviews.pageInfo.hasNextPage == false)
            or (
              ((.data.repository.pullRequest.reviews.pageInfo.endCursor // "") | type) == "string"
              and (((.data.repository.pullRequest.reviews.pageInfo.endCursor // "") | length) > 0)
            )
          )
        ' >/dev/null 2>&1; then
        FINAL_REVIEWS_COMPLETE=false
        break
      fi
      FINAL_PAGE_NODES=$(echo "$FINAL_REVIEW_PAGE" | jq -c '.data.repository.pullRequest.reviews.nodes // []') || { FINAL_REVIEWS_COMPLETE=false; break; }
      FINAL_REVIEW_NODES=$(jq -n -c --argjson existing "$FINAL_REVIEW_NODES" --argjson new "$FINAL_PAGE_NODES" '$existing + $new') || { FINAL_REVIEWS_COMPLETE=false; break; }
      FINAL_REVIEW_PAGES=$((FINAL_REVIEW_PAGES + 1))
      FINAL_HAS_NEXT=$(echo "$FINAL_REVIEW_PAGE" | jq -r '.data.repository.pullRequest.reviews.pageInfo.hasNextPage')
      [ "$FINAL_HAS_NEXT" = "true" ] || break
      FINAL_REVIEW_CURSOR=$(echo "$FINAL_REVIEW_PAGE" | jq -r '.data.repository.pullRequest.reviews.pageInfo.endCursor // empty')
      [ -n "$FINAL_REVIEW_CURSOR" ] || { FINAL_REVIEWS_COMPLETE=false; break; }
    done
    FINAL_COMMENT_NODES='[]'
    FINAL_COMMENT_CURSOR=''
    while [ "$FINAL_REVIEWS_COMPLETE" = "true" ]; do
      FINAL_COMMENT_ARGS=(gh api graphql -f query="$COMMENT_QUERY" -f owner="$OWNER" -f name="$NAME" -F pr="$PR")
      [ -n "$FINAL_COMMENT_CURSOR" ] && FINAL_COMMENT_ARGS+=(-f cursor="$FINAL_COMMENT_CURSOR")
      FINAL_COMMENT_PAGE=$(GH_TOKEN="$TOK" "${FINAL_COMMENT_ARGS[@]}" 2>/dev/null) || { FINAL_REVIEWS_COMPLETE=false; break; }
      if ! echo "$FINAL_COMMENT_PAGE" | jq -e '
          (((.errors // []) | length) == 0)
          and ((.data.repository.pullRequest | type) == "object")
          and ((.data.repository.pullRequest.comments | type) == "object")
          and ((.data.repository.pullRequest.comments.nodes | type) == "array")
          and ((.data.repository.pullRequest.comments.pageInfo | type) == "object")
          and ((.data.repository.pullRequest.comments.pageInfo.hasNextPage | type) == "boolean")
          and (
            (.data.repository.pullRequest.comments.pageInfo.hasNextPage == false)
            or (
              ((.data.repository.pullRequest.comments.pageInfo.endCursor // "") | type) == "string"
              and (((.data.repository.pullRequest.comments.pageInfo.endCursor // "") | length) > 0)
            )
          )
        ' >/dev/null 2>&1; then
        FINAL_REVIEWS_COMPLETE=false
        break
      fi
      FINAL_COMMENT_PAGE_NODES=$(echo "$FINAL_COMMENT_PAGE" | jq -c '.data.repository.pullRequest.comments.nodes // []') || { FINAL_REVIEWS_COMPLETE=false; break; }
      FINAL_COMMENT_NODES=$(jq -n -c --argjson existing "$FINAL_COMMENT_NODES" --argjson new "$FINAL_COMMENT_PAGE_NODES" '$existing + $new') || { FINAL_REVIEWS_COMPLETE=false; break; }
      FINAL_REVIEW_PAGES=$((FINAL_REVIEW_PAGES + 1))
      FINAL_COMMENT_HAS_NEXT=$(echo "$FINAL_COMMENT_PAGE" | jq -r '.data.repository.pullRequest.comments.pageInfo.hasNextPage')
      [ "$FINAL_COMMENT_HAS_NEXT" = "true" ] || break
      FINAL_COMMENT_CURSOR=$(echo "$FINAL_COMMENT_PAGE" | jq -r '.data.repository.pullRequest.comments.pageInfo.endCursor // empty')
      [ -n "$FINAL_COMMENT_CURSOR" ] || { FINAL_REVIEWS_COMPLETE=false; break; }
    done
    FINAL_CODEX_FORMAL_REVIEWS=$(echo "$FINAL_REVIEW_NODES" | jq --arg sha "$SHA" --argjson codex "$CODEX_LOGINS_JSON" '[.[]? | select(((((.author.login // "") | ascii_downcase) as $login | $codex | index($login)) != null) and ((.commit.oid // "") == $sha) and ((.state // "") | IN("COMMENTED","APPROVED")))] | length')
    FINAL_CODEX_CLEAN_COMMENTS=$(echo "$FINAL_COMMENT_NODES" | jq --arg sha "$SHA" --argjson codex "$CODEX_LOGINS_JSON" '[.[]? | ((.body // .bodyText // "") as $body | ((.author.login // "") | ascii_downcase) as $login | select(($codex | index($login)) != null) | select(($body | ascii_downcase | contains("didn'\''t find any major issues"))) | ((try ($body | capture("\\*\\*Reviewed commit:\\*\\*\\s*`(?<reviewed>[0-9a-fA-F]{10,40})`").reviewed) catch "") | ascii_downcase) as $reviewed | select(($reviewed | length) > 0 and ($sha | startswith($reviewed))))] | length')
    FINAL_CODEX_HEAD_REVIEWS=$((FINAL_CODEX_FORMAL_REVIEWS + FINAL_CODEX_CLEAN_COMMENTS))
    FINAL_CR=$(GH_TOKEN="$TOK" gh api --paginate "repos/$REPO/commits/$SHA/check-runs?per_page=100" 2>/dev/null | jq -s '{check_runs:[.[].check_runs[]]}')
    FINAL_REQLATEST=$(echo "$FINAL_CR" | jq --argjson app "$REQ_APP_ID" '[.check_runs[]|select(.app.id==$app)]|group_by(.name)|map(sort_by(.started_at)|last)')
    FINAL_REQRED=$(echo "$FINAL_REQLATEST" | jq --argjson req "$REQ_JSON" '[.[]|select(.name as $n|$req|index($n))|select(.status=="completed" and (.conclusion|IN("failure","cancelled","timed_out","action_required","stale","startup_failure")))]|length')
    FINAL_REQGREEN=$(echo "$FINAL_REQLATEST" | jq --argjson req "$REQ_JSON" '[.[]|select(.name as $n|$req|index($n))|select(.status=="completed" and (.conclusion|IN("success","neutral","skipped")))]|length')
    FINAL_REQUNSETTLED=$(echo "$FINAL_CR" | jq --argjson app "$REQ_APP_ID" --argjson req "$REQ_JSON" '[.check_runs[]|select(.app.id==$app)|select(.name as $n|$req|index($n))|select(.status!="completed")]|length')
    if [ "$FINAL_UNRES" != "0" ] || [ "$FINAL_DECISION" = "CHANGES_REQUESTED" ] || [ "$FINAL_MERGEABLE" != "MERGEABLE" ] \
       || { [ "$FINAL_MSTATE" != "CLEAN" ] && [ "$FINAL_MSTATE" != "UNSTABLE" ]; } \
       || [ "$FINAL_REVIEWS_COMPLETE" != "true" ] || [ "$FINAL_CODEX_HEAD_REVIEWS" -eq 0 ] \
       || [ "$FINAL_REQRED" -gt 0 ] || [ "$FINAL_REQGREEN" -ne "$REQ_TOTAL" ] || [ "$FINAL_REQUNSETTLED" -ne 0 ]; then
      echo "ACTIONABLE: final-read req-green=$FINAL_REQGREEN/$REQ_TOTAL req-red=$FINAL_REQRED req-unsettled=$FINAL_REQUNSETTLED codex-head-attestations=$FINAL_CODEX_HEAD_REVIEWS attestation-pages=$FINAL_REVIEW_PAGES attestations-complete=$FINAL_REVIEWS_COMPLETE threads=$FINAL_UNRES decision=$FINAL_DECISION mergeable=$FINAL_MERGEABLE merge-state=$FINAL_MSTATE -> reconcile/fix, push, re-arm"; exit 0
    fi
    echo "MERGE-READY: all $REQ_TOTAL required contexts green + current-head Codex review attestation + threads clear + merge-state $MSTATE."
    echo "-> pre-merge checklist first (clean tree, local==remote, re-verify threads=0), then merge + alert."
    exit 0
  fi
done
echo "watch window elapsed; re-arm to continue"
