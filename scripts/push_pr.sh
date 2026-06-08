#!/usr/bin/env bash
# Push a PR branch with the PR body env wired into exactly one local review.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

usage() {
    cat <<'EOF'
Usage: bash scripts/push_pr.sh BODY_FILE [git-push-args...]

Pushes with ATLAS_CURRENT_PR_BODY_FILE exported so the installed pre-push hook
validates the same body. When the managed hook is installed, the hook is the
single local-review runner. Without that hook, this wrapper runs local review
before pushing.

Examples:
  bash scripts/push_pr.sh tmp/pr-body-my-slice.md -u origin HEAD
  bash scripts/push_pr.sh tmp/pr-body-my-slice.md -u origin claude/pr-my-slice
EOF
}

has_managed_pre_push_hook() {
    local hook_path
    hook_path="$(git rev-parse --git-path hooks/pre-push)"
    [ -x "$hook_path" ] && grep -q "ATLAS_LOCAL_PR_REVIEW_HOOK" "$hook_path"
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

for arg in "$@"; do
    if [ "$arg" = "--no-verify" ]; then
        echo "push_pr.sh: refusing to forward --no-verify; local PR review must run once" >&2
        exit 2
    fi
done

run_wrapper_review=1
if has_managed_pre_push_hook && [ "${ATLAS_SKIP_LOCAL_PR_REVIEW:-}" != "1" ]; then
    run_wrapper_review=0
fi

if [ "${ATLAS_PUSH_PR_DRY_RUN:-}" = "1" ]; then
    if [ "$run_wrapper_review" -eq 1 ]; then
        echo "DRY RUN: ATLAS_CURRENT_PR_BODY_FILE=$body_file bash scripts/local_pr_review.sh --current-pr-body-file $body_file"
    else
        echo "DRY RUN: managed pre-push hook will run local PR review once with body: $body_file"
    fi
    echo "DRY RUN: ATLAS_CURRENT_PR_BODY_FILE=$body_file git push $*"
    exit 0
fi

if [ "$run_wrapper_review" -eq 1 ]; then
    echo "Running local PR review with PR body: $body_file"
    ATLAS_CURRENT_PR_BODY_FILE="$body_file" \
        bash scripts/local_pr_review.sh --current-pr-body-file "$body_file"
else
    echo "Managed pre-push hook detected; local PR review will run once in the hook."
fi

echo "Pushing with PR body env available to pre-push hook: $body_file"
ATLAS_CURRENT_PR_BODY_FILE="$body_file" git push "$@"
