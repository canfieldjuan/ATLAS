#!/usr/bin/env bash
# Open or update a GitHub PR while feeding the body through stdin.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

usage() {
    cat <<'EOF'
Usage: bash scripts/open_pr.sh BODY_FILE [gh-pr-create-args...]

Creates a PR for the current branch, or updates the existing PR body for that
branch. The PR body is always passed as stdin (`--body-file - < BODY_FILE`) so
the GitHub CLI never has to open BODY_FILE itself.

Examples:
  bash scripts/open_pr.sh tmp/pr-body-my-slice.md --title "My slice" --base main
  bash scripts/open_pr.sh tmp/pr-body-my-slice.md

Use scripts/push_pr.sh before this wrapper to push the branch with the local
review body env wired into the pre-push hook.
EOF
}

if [ "$#" -lt 1 ] || [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    usage
    exit 2
fi

body_file="$1"
shift

if [ ! -f "$body_file" ]; then
    echo "open_pr.sh: PR body file not found: $body_file" >&2
    echo "Create the body file first, then rerun this wrapper." >&2
    exit 2
fi

for arg in "$@"; do
    case "$arg" in
        --body|--body-file|-b|-F)
            echo "open_pr.sh: pass the PR body as BODY_FILE, not via $arg" >&2
            exit 2
            ;;
    esac
done

branch="$(git branch --show-current)"
if [ -z "$branch" ]; then
    echo "open_pr.sh: current checkout is detached; switch to a branch first" >&2
    exit 2
fi

if gh pr view "$branch" >/dev/null 2>&1; then
    if [ "$#" -gt 0 ]; then
        echo "open_pr.sh: PR already exists for $branch; update body with no create args" >&2
        echo "Use gh pr edit manually for title/base/label changes." >&2
        exit 2
    fi

    if [ "${ATLAS_OPEN_PR_DRY_RUN:-}" = "1" ]; then
        echo "DRY RUN: gh pr edit $branch --body-file - < $body_file"
        exit 0
    fi

    gh pr edit "$branch" --body-file - < "$body_file"
else
    if [ "${ATLAS_OPEN_PR_DRY_RUN:-}" = "1" ]; then
        echo "DRY RUN: gh pr create $* --body-file - < $body_file"
        exit 0
    fi

    gh pr create "$@" --body-file - < "$body_file"
fi
