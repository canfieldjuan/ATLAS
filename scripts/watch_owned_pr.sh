#!/usr/bin/env bash
# Owned-PR watcher (docs/OVERNIGHT_ARC_WORKFLOW.md section 5).
# Watches ONE PR the current builder session owns and exits the moment there
# is something to act on:
#   MERGED/CLOSED - PR reached terminal state (stop watching)
#   HEAD-MOVED    - branch advanced past the SHA this watch was armed on
#   ACTIONABLE    - red required context / unresolved review threads /
#                   CHANGES_REQUESTED review decision
#   MERGE-READY   - EVERY required context present and success +
#                   Codex review exists on this exact head SHA +
#                   0 unresolved threads (no unfetched pages) +
#                   review decision not CHANGES_REQUESTED + mergeable
# Required contexts + app pin are read from origin/main's
# scripts/check_required_status_checks.py (trusted ref -- the watched branch
# cannot weaken its own gate); MERGE-READY requires their PRESENCE with
# success and no unsettled (queued/rerunning) required run, so a context
# that has not started or is rerunning fails closed.
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
# edits check_required_status_checks.py must not be able to weaken its own
# gate. Falls back to the checkout copy, then the documented four.
CHECKER_SRC="$(git -C "$ROOT" show origin/main:scripts/check_required_status_checks.py 2>/dev/null)"
[ -n "$CHECKER_SRC" ] || CHECKER_SRC="$(cat "$ROOT/scripts/check_required_status_checks.py" 2>/dev/null)"
mapfile -t REQ_CONTEXTS < <(printf '%s\n' "$CHECKER_SRC" \
  | sed -n '/^DEFAULT_REQUIRED_CONTEXTS = (/,/^)/p' | grep -oE '"[^"]+"' | tr -d '"')
if [ "${#REQ_CONTEXTS[@]}" -eq 0 ]; then
  REQ_CONTEXTS=("live-reconciliation" "diff-budget" "Gitleaks PR secret scan" "Gitleaks baseline growth guard")
fi
REQ_JSON=$(printf '%s\n' "${REQ_CONTEXTS[@]}" | jq -R . | jq -cs .)
REQ_TOTAL=${#REQ_CONTEXTS[@]}
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
  ST=$(GH_TOKEN="$TOK" gh api graphql -f query="{ repository(owner:\"$OWNER\",name:\"$NAME\"){ pullRequest(number:$PR){ state merged mergeable mergeStateStatus reviewDecision reviews(first:100){ nodes{ author{ login } commit{ oid } state } } reviewThreads(first:100){ pageInfo{ hasNextPage } nodes{ isResolved isOutdated comments(first:1){ nodes{ author{ login } } } } } } } }" 2>/dev/null)
  STATE=$(echo "$ST" | jq -r '.data.repository.pullRequest | .state + (if .merged then "/merged" else "" end)')
  MERGEABLE=$(echo "$ST" | jq -r '.data.repository.pullRequest.mergeable')
  MSTATE=$(echo "$ST" | jq -r '.data.repository.pullRequest.mergeStateStatus // "UNKNOWN"')
  DECISION=$(echo "$ST" | jq -r '.data.repository.pullRequest.reviewDecision // "NONE"')
  CODEX_HEAD_REVIEWS=$(echo "$ST" | jq --arg sha "$SHA" '[.data.repository.pullRequest.reviews.nodes[]? | select(((.author.login // "") | ascii_downcase | contains("codex")) and ((.commit.oid // "") == $sha) and ((.state // "") | IN("COMMENTED","APPROVED","CHANGES_REQUESTED")))] | length')
  UNRES=$(echo "$ST" | jq '[.data.repository.pullRequest.reviewThreads.nodes[]? | select((.isResolved==false) and (.isOutdated!=true) and (((.comments.nodes[0].author.login // "") | ascii_downcase | contains("codex"))))] | length')
  MORE=$(echo "$ST" | jq -r '.data.repository.pullRequest.reviewThreads.pageInfo.hasNextPage')
  # Fail closed when more thread pages exist than we fetched.
  [ "$MORE" = "true" ] && UNRES="${UNRES}+unfetched-pages"
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
  echo "cycle $i $(date +%H:%M): state=$STATE req-green=$REQGREEN/$REQ_TOTAL req-red=$REQRED req-unsettled=$REQUNSETTLED pending=$PEND codex-head-reviews=$CODEX_HEAD_REVIEWS threads=$UNRES decision=$DECISION mergeable=$MERGEABLE merge-state=$MSTATE"
  case "$STATE" in MERGED/merged|CLOSED) echo "TERMINAL: PR $STATE"; exit 0;; esac
  # Definite negatives are actionable on ANY cycle, including the first.
  if [ "$REQRED" -gt 0 ] || [ "$UNRES" != "0" ] || [ "$DECISION" = "CHANGES_REQUESTED" ] || [ "$CODEX_HEAD_REVIEWS" -eq 0 ]; then
    echo "ACTIONABLE: req-red=$REQRED codex-head-reviews=$CODEX_HEAD_REVIEWS threads=$UNRES decision=$DECISION -> reconcile/fix, push, re-arm"; exit 0
  fi
  # Readiness is presence-based: every required context must be reporting
  # success (a not-yet-started context keeps this false, so no early race).
  if [ "$REQGREEN" -eq "$REQ_TOTAL" ] && [ "$REQUNSETTLED" -eq 0 ] && [ "$CODEX_HEAD_REVIEWS" -gt 0 ] && [ "$MERGEABLE" = "MERGEABLE" ] \
     && { [ "$MSTATE" = "CLEAN" ] || [ "$MSTATE" = "UNSTABLE" ]; }; then
    echo "MERGE-READY: all $REQ_TOTAL required contexts green + current-head Codex review + threads clear + merge-state $MSTATE."
    echo "-> pre-merge checklist first (clean tree, local==remote, re-verify threads=0), then merge + alert."
    exit 0
  fi
done
echo "watch window elapsed; re-arm to continue"
