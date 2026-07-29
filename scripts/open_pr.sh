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

snapshot_body() {
    local source="$1" snapshot
    snapshot="$(mktemp "${TMPDIR:-/tmp}/atlas-pr-body.XXXXXX")"
    cp "$source" "$snapshot"
    printf '%s\n' "$snapshot"
}

cleanup_body_snapshot() {
    if [ -n "${body_snapshot:-}" ]; then
        rm -f "$body_snapshot"
    fi
}

refresh_base_ref() {
    echo "Refreshing origin/main before PR body audit..."
    if ! git fetch --quiet origin main; then
        echo "open_pr.sh: failed to refresh origin/main; fetch/rebase before opening or updating a PR" >&2
        exit 1
    fi
}

body_sha256() {
    python - "$1" <<'PY'
import hashlib
import sys
from pathlib import Path

print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
}

proof_value() {
    local key="$1" proof_file="$2"
    awk -F= -v key="$key" '$1 == key {print substr($0, length(key) + 2); exit}' "$proof_file"
}

existing_pr_number_for_branch() {
    local branch="$1" current_repo_json pr_json
    if ! current_repo_json="$(gh repo view --json owner,name)"; then
        echo "open_pr.sh: failed to query current GitHub repository" >&2
        exit 1
    fi
    if ! pr_json="$(gh pr list --state open --head "$branch" --json number,headRefName,headRepository,headRepositoryOwner,isCrossRepository --limit 20)"; then
        echo "open_pr.sh: failed to query existing PR for head branch $branch" >&2
        exit 1
    fi
    python - "$branch" "$current_repo_json" "$pr_json" <<'PY'
import json
import sys

branch = sys.argv[1]
current_repo = json.loads(sys.argv[2])
current_owner = (current_repo.get("owner") or {}).get("login")
current_name = current_repo.get("name")
matches = [
    item for item in json.loads(sys.argv[3])
    if item.get("headRefName") == branch
    and not item.get("isCrossRepository", False)
    and ((item.get("headRepositoryOwner") or {}).get("login") == current_owner)
    and ((item.get("headRepository") or {}).get("name") == current_name)
]
if len(matches) > 1:
    raise SystemExit(f"multiple open PRs found for head branch {branch}")
if matches:
    print(matches[0]["number"])
PY
}

require_local_review_proof() {
    local proof_path expected_branch expected_head expected_base expected_body actual_branch actual_head actual_base actual_body
    proof_path="$(git rev-parse --git-path atlas-local-pr-review-proof)"
    if [ ! -f "$proof_path" ]; then
        echo "open_pr.sh: missing local review proof for this head/body." >&2
        echo "Run scripts/push_pr.sh with this PR body before opening or updating the PR." >&2
        exit 2
    fi

    expected_branch="$(proof_value branch "$proof_path")"
    expected_head="$(proof_value head_sha "$proof_path")"
    expected_base="$(proof_value base_sha "$proof_path")"
    expected_body="$(proof_value body_sha256 "$proof_path")"
    actual_branch="$(git branch --show-current)"
    actual_head="$(git rev-parse HEAD)"
    actual_base="$(git rev-parse origin/main)"
    actual_body="$(body_sha256 "$body_snapshot")"

    if [ "$expected_branch" != "$actual_branch" ]; then
        echo "open_pr.sh: stale local review proof: expected branch $actual_branch, found ${expected_branch:-<missing>}." >&2
        echo "Run scripts/push_pr.sh again from this branch before opening or updating the PR." >&2
        exit 2
    fi
    if [ "$expected_head" != "$actual_head" ]; then
        echo "open_pr.sh: stale local review proof: expected HEAD $actual_head, found ${expected_head:-<missing>}." >&2
        echo "Run scripts/push_pr.sh again before opening or updating the PR." >&2
        exit 2
    fi
    if [ "$expected_base" != "$actual_base" ]; then
        echo "open_pr.sh: stale local review proof: expected origin/main $actual_base, found ${expected_base:-<missing>}." >&2
        echo "Run scripts/push_pr.sh again before opening or updating the PR." >&2
        exit 2
    fi
    if [ "$expected_body" != "$actual_body" ]; then
        echo "open_pr.sh: stale local review proof: PR body changed after local review." >&2
        echo "Run scripts/push_pr.sh again with this body before opening or updating the PR." >&2
        exit 2
    fi
}

verify_published_head() {
    local actual_branch actual_head actual_remote
    actual_branch="$(git branch --show-current)"
    actual_head="$(git rev-parse HEAD)"
    if ! git fetch --quiet origin "$actual_branch"; then
        echo "open_pr.sh: failed to refresh origin/$actual_branch before PR mutation." >&2
        echo "Run scripts/push_pr.sh again before opening or updating the PR." >&2
        exit 1
    fi
    if ! actual_remote="$(git rev-parse "refs/remotes/origin/$actual_branch" 2>/dev/null)"; then
        echo "open_pr.sh: reviewed branch is not published at origin/$actual_branch." >&2
        echo "Run scripts/push_pr.sh again before opening or updating the PR." >&2
        exit 2
    fi
    if [ "$actual_remote" != "$actual_head" ]; then
        echo "open_pr.sh: stale local review proof: origin/$actual_branch is $actual_remote, current HEAD is $actual_head." >&2
        echo "Run scripts/push_pr.sh again before opening or updating the PR." >&2
        exit 2
    fi
}

reject_environment_target_overrides() {
    if [ -n "${GH_REPO:-}" ]; then
        echo "open_pr.sh: refusing GH_REPO target override: $GH_REPO" >&2
        echo "Open PRs in the current repository only; unset GH_REPO and rerun this wrapper." >&2
        exit 2
    fi
}

validate_create_target_args() {
    local arg value
    while [ "$#" -gt 0 ]; do
        arg="$1"
        shift
        case "$arg" in
            --head|-H|--repo|-R)
                echo "open_pr.sh: refusing target-changing create arg: $arg" >&2
                echo "Open PRs from the current branch in this repo only; use a separate reviewed helper for target overrides." >&2
                exit 2
                ;;
            --head=*|--repo=*|-H*|-R*)
                echo "open_pr.sh: refusing target-changing create arg: $arg" >&2
                echo "Open PRs from the current branch in this repo only; use a separate reviewed helper for target overrides." >&2
                exit 2
                ;;
            --base)
                if [ "$#" -eq 0 ]; then
                    echo "open_pr.sh: --base requires a value" >&2
                    exit 2
                fi
                value="$1"
                shift
                if [ "$value" != "main" ]; then
                    echo "open_pr.sh: refusing non-main base: $value" >&2
                    echo "The local review proof is bound to origin/main; use a separate reviewed helper for target overrides." >&2
                    exit 2
                fi
                ;;
            --base=*)
                value="${arg#--base=}"
                if [ "$value" != "main" ]; then
                    echo "open_pr.sh: refusing non-main base: $value" >&2
                    echo "The local review proof is bound to origin/main; use a separate reviewed helper for target overrides." >&2
                    exit 2
                fi
                ;;
            -B)
                if [ "$#" -eq 0 ]; then
                    echo "open_pr.sh: -B requires a value" >&2
                    exit 2
                fi
                value="$1"
                shift
                if [ "$value" != "main" ]; then
                    echo "open_pr.sh: refusing non-main base: $value" >&2
                    echo "The local review proof is bound to origin/main; use a separate reviewed helper for target overrides." >&2
                    exit 2
                fi
                ;;
            -B*)
                value="${arg#-B}"
                if [ "$value" != "main" ]; then
                    echo "open_pr.sh: refusing non-main base: $value" >&2
                    echo "The local review proof is bound to origin/main; use a separate reviewed helper for target overrides." >&2
                    exit 2
                fi
                ;;
        esac
    done
}

body_snapshot="$(snapshot_body "$body_file")"
trap cleanup_body_snapshot EXIT

refresh_base_ref

body_audit_args=(--base-ref origin/main)
if [ -n "${ATLAS_CURRENT_PR_AUTHOR:-}" ]; then
    body_audit_args+=(--pr-author "$ATLAS_CURRENT_PR_AUTHOR")
fi
python scripts/audit_pr_body.py "${body_audit_args[@]}" "$body_snapshot"

for arg in "$@"; do
    case "$arg" in
        --body|--body-file|-b|-F)
            echo "open_pr.sh: pass the PR body as BODY_FILE, not via $arg" >&2
            exit 2
            ;;
    esac
done
validate_create_target_args "$@"
reject_environment_target_overrides

branch="$(git branch --show-current)"
if [ -z "$branch" ]; then
    echo "open_pr.sh: current checkout is detached; switch to a branch first" >&2
    exit 2
fi

require_local_review_proof
verify_published_head

existing_pr_number="$(existing_pr_number_for_branch "$branch")"
if [ -n "$existing_pr_number" ]; then
    if [ "$#" -gt 0 ]; then
        echo "open_pr.sh: PR already exists for $branch; update body with no create args" >&2
        echo "Use gh pr edit manually for title/base/label changes." >&2
        exit 2
    fi

    if [ "${ATLAS_OPEN_PR_DRY_RUN:-}" = "1" ]; then
        verify_published_head
        echo "DRY RUN: gh pr edit $existing_pr_number --body-file - < $body_file"
        exit 0
    fi

    verify_published_head
    gh pr edit "$existing_pr_number" --body-file - < "$body_snapshot"
else
    if [ "${ATLAS_OPEN_PR_DRY_RUN:-}" = "1" ]; then
        verify_published_head
        echo "DRY RUN: gh pr create --head $branch $* --body-file - < $body_file"
        exit 0
    fi

    verify_published_head
    gh pr create --head "$branch" "$@" --body-file - < "$body_snapshot"
fi
