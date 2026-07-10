#!/usr/bin/env bash
# Owned-PR watcher (docs/OVERNIGHT_ARC_WORKFLOW.md section 5).
# Watches ONE PR the current builder session owns and exits the moment there
# is something to act on:
#   MERGED/CLOSED - PR reached terminal state (stop watching)
#   HEAD-MOVED    - branch advanced past the SHA this watch was armed on
#   ACTIONABLE    - red required check / unresolved review threads / CHANGES_REQUESTED
#   MERGE-READY   - required green + 0 unresolved threads + no CHANGES_REQUESTED + mergeable
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
# Required checks; everything else is advisory (state once, do not gate).
REQ_FILTER='select((.name=="live-reconciliation" or (.name|startswith("Gitleaks"))) and .conclusion!="success" and .conclusion!="skipped" and .status=="completed")'
echo "owned-pr watcher armed: $REPO#$PR @ ${SHA:0:9} $(date '+%F %H:%M')"
for i in $(seq 0 "$CYCLES"); do
  [ "$i" -gt 0 ] && sleep 1740
  CUR=$(GH_TOKEN="$TOK" gh api "repos/$REPO/pulls/$PR" --jq '.head.sha' 2>/dev/null) || { echo "cycle $i: API error, retrying"; continue; }
  if [ "$CUR" != "$SHA" ]; then echo "HEAD-MOVED: ${SHA:0:9} -> ${CUR:0:9} (new push; reconcile + re-arm on new head)"; exit 0; fi
  ST=$(GH_TOKEN="$TOK" gh api graphql -f query="{ repository(owner:\"$OWNER\",name:\"$NAME\"){ pullRequest(number:$PR){ state merged mergeable reviewThreads(first:100){ nodes{ isResolved } } reviews(last:30){ nodes{ state } } } } }" 2>/dev/null)
  STATE=$(echo "$ST" | jq -r '.data.repository.pullRequest | .state + (if .merged then "/merged" else "" end)')
  MERGEABLE=$(echo "$ST" | jq -r '.data.repository.pullRequest.mergeable')
  UNRES=$(echo "$ST" | jq '[.data.repository.pullRequest.reviewThreads.nodes[]|select(.isResolved==false)]|length')
  CHREQ=$(echo "$ST" | jq '[.data.repository.pullRequest.reviews.nodes[]|select(.state=="CHANGES_REQUESTED")]|length')
  CR=$(GH_TOKEN="$TOK" gh api "repos/$REPO/commits/$SHA/check-runs?per_page=100" 2>/dev/null)
  PEND=$(echo "$CR" | jq '[.check_runs[]|select(.status!="completed")]|length')
  REQRED=$(echo "$CR" | jq "[[.check_runs[]]|group_by(.name)|map(sort_by(.started_at)|last)|.[]|$REQ_FILTER]|length")
  # claude-review is a per-SHA commit STATUS (not a check-run); absent or
  # failure is not clean (AGENTS.md 3c.1.8, docs/REVIEWER_MERGE_GATE.md).
  CLREV=$(GH_TOKEN="$TOK" gh api "repos/$REPO/commits/$SHA/status" --jq '([.statuses[]|select(.context=="claude-review")]|last|.state) // "absent"' 2>/dev/null || echo "absent")
  echo "cycle $i $(date +%H:%M): state=$STATE pending=$PEND req-red=$REQRED threads=$UNRES chreq=$CHREQ claude-review=$CLREV mergeable=$MERGEABLE"
  case "$STATE" in MERGED/merged|CLOSED) echo "TERMINAL: PR $STATE"; exit 0;; esac
  if [ "$i" -gt 0 ]; then # give checks one cycle to start before judging
    if [ "$REQRED" -gt 0 ] || [ "$UNRES" -gt 0 ] || [ "$CHREQ" -gt 0 ] || [ "$CLREV" = "failure" ]; then
      echo "ACTIONABLE: req-red=$REQRED threads=$UNRES chreq=$CHREQ claude-review=$CLREV -> reconcile/fix, push, re-arm"; exit 0
    fi
    if [ "$PEND" -eq 0 ] && [ "$MERGEABLE" = "MERGEABLE" ] && [ "$CLREV" = "success" ]; then
      echo "MERGE-READY: required green + threads clear + claude-review success + mergeable."
      echo "-> pre-merge checklist first (clean tree, local==remote, re-verify threads=0), then merge + alert."
      exit 0
    fi
  fi
done
echo "watch window elapsed; re-arm to continue"
