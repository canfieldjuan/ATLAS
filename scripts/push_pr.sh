#!/usr/bin/env bash
# Push a PR branch with the PR body env wired into local review and pre-push.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

usage() {
    cat <<'EOF'
Usage: bash scripts/push_pr.sh BODY_FILE [git-push-args...]

Runs local PR review with BODY_FILE, then pushes with ATLAS_CURRENT_PR_BODY_FILE
exported so the installed pre-push hook validates the same body.

Examples:
  bash scripts/push_pr.sh tmp/pr-body-my-slice.md -u origin HEAD
  bash scripts/push_pr.sh tmp/pr-body-my-slice.md -u origin claude/pr-my-slice
EOF
}

if [ "$#" -lt 1 ] || [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    usage
    exit 2
fi

body_file="$1"
shift

if [ ! -f "$body_file" ]; then
    echo "push_pr.sh: PR body file not found: $body_file" >&2
    echo "Create the body file first, then rerun this wrapper." >&2
    exit 2
fi

if [ "$#" -eq 0 ]; then
    set -- -u origin HEAD
fi

if [ "${ATLAS_PUSH_PR_DRY_RUN:-}" = "1" ]; then
    echo "DRY RUN: ATLAS_CURRENT_PR_BODY_FILE=$body_file bash scripts/local_pr_review.sh --current-pr-body-file $body_file"
    echo "DRY RUN: ATLAS_CURRENT_PR_BODY_FILE=$body_file git push $*"
    exit 0
fi

echo "Running local PR review with PR body: $body_file"
ATLAS_CURRENT_PR_BODY_FILE="$body_file" \
    bash scripts/local_pr_review.sh --current-pr-body-file "$body_file"

echo "Pushing with PR body env available to pre-push hook: $body_file"
ATLAS_CURRENT_PR_BODY_FILE="$body_file" git push "$@"
