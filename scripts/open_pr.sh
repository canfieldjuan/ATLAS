#!/usr/bin/env bash
# Open or update a GitHub PR while feeding the body through stdin.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
python_bin="${PYTHON:-python3}"

usage() {
    cat <<'EOF'
Usage: bash scripts/open_pr.sh BODY_FILE [gh-pr-create-args...]

Creates a PR for the current branch, or updates the existing PR body for that
branch. The PR body is always passed as stdin (`--body-file - < BODY_FILE`) so
the GitHub CLI never has to open BODY_FILE itself.

Examples:
  bash scripts/open_pr.sh tmp/pr-body-my-slice.md --title "My slice" --base main
  bash scripts/open_pr.sh tmp/pr-body-my-slice.md

Draft PRs require explicit operator consent:
  ATLAS_OPEN_PR_DRAFT_CONSENT=1 bash scripts/open_pr.sh tmp/pr-body-my-slice.md --draft

Use scripts/push_pr.sh before this wrapper to push the branch with the local
review body env wired into the pre-push hook.

This wrapper enforces a single local writer with an advisory git lock, then
checks the reviewed head/body/PR snapshot immediately before and after the
GitHub mutation. GitHub does not expose an atomic compare-and-swap for PR body
edits, so competing writers must use this wrapper or be treated as a failed
post-mutation verification.
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

refresh_base_ref() {
    echo "Refreshing origin/main before PR body audit..."
    if ! git fetch --quiet origin main; then
        echo "open_pr.sh: failed to refresh origin/main; fetch/rebase before opening or updating a PR" >&2
        exit 1
    fi
}

reject_target_overrides() {
    if [ -n "${GH_REPO:-}" ]; then
        echo "open_pr.sh: refusing GH_REPO target override: $GH_REPO" >&2
        echo "Open PRs in the current repository only; unset GH_REPO and rerun this wrapper." >&2
        exit 2
    fi

    local arg value
    while [ "$#" -gt 0 ]; do
        arg="$1"
        shift
        case "$arg" in
            --head|-H|--repo|-R)
                echo "open_pr.sh: refusing target-changing create arg: $arg" >&2
                exit 2
                ;;
            --head=*|--repo=*|-H*|-R*)
                echo "open_pr.sh: refusing target-changing create arg: $arg" >&2
                exit 2
                ;;
            --draft|--draft=*|-d*)
                if [ "${ATLAS_OPEN_PR_DRAFT_CONSENT:-}" != "1" ]; then
                    echo "open_pr.sh: refusing draft PR without explicit operator consent: $arg" >&2
                    echo "Set ATLAS_OPEN_PR_DRAFT_CONSENT=1 only when the operator asked for a draft." >&2
                    exit 2
                fi
                ;;
            --base|-B)
                if [ "$#" -eq 0 ]; then
                    echo "open_pr.sh: $arg requires a value" >&2
                    exit 2
                fi
                value="$1"
                shift
                if [ "$value" != "main" ]; then
                    echo "open_pr.sh: refusing non-main base: $value" >&2
                    echo "The local review gate is bound to origin/main." >&2
                    exit 2
                fi
                ;;
            --base=*)
                value="${arg#--base=}"
                if [ "$value" != "main" ]; then
                    echo "open_pr.sh: refusing non-main base: $value" >&2
                    echo "The local review gate is bound to origin/main." >&2
                    exit 2
                fi
                ;;
            -B*)
                value="${arg#-B}"
                if [ "$value" != "main" ]; then
                    echo "open_pr.sh: refusing non-main base: $value" >&2
                    echo "The local review gate is bound to origin/main." >&2
                    exit 2
                fi
                ;;
        esac
    done
}

verify_published_head() {
    local branch="$1" expected_sha="${2:-}" head_sha remote_sha
    head_sha="$(git rev-parse HEAD)"
    if ! git fetch --quiet origin "refs/heads/$branch"; then
        echo "open_pr.sh: failed to refresh origin/$branch before PR mutation." >&2
        echo "Run scripts/push_pr.sh before opening or updating the PR." >&2
        exit 1
    fi
    if ! remote_sha="$(git rev-parse FETCH_HEAD 2>/dev/null)"; then
        echo "open_pr.sh: current branch is not published at origin/$branch." >&2
        echo "Run scripts/push_pr.sh before opening or updating the PR." >&2
        exit 2
    fi
    if [ -n "$expected_sha" ] && [ "$head_sha" != "$expected_sha" ]; then
        echo "open_pr.sh: current HEAD changed after review: reviewed $expected_sha, now $head_sha." >&2
        echo "Rerun this wrapper so local review covers the exact mutation snapshot." >&2
        exit 2
    fi
    if [ -n "$expected_sha" ] && [ "$remote_sha" != "$expected_sha" ]; then
        echo "open_pr.sh: origin/$branch changed after review: reviewed $expected_sha, now $remote_sha." >&2
        echo "Rerun this wrapper so local review covers the exact mutation snapshot." >&2
        exit 2
    fi
    if [ -z "$expected_sha" ] && [ "$remote_sha" != "$head_sha" ]; then
        echo "open_pr.sh: origin/$branch is $remote_sha, current HEAD is $head_sha." >&2
        echo "Run scripts/push_pr.sh before opening or updating the PR." >&2
        exit 2
    fi
    printf '%s\n' "$head_sha"
}

origin_repo_name() {
    local url
    url="$(git config --get remote.origin.url)"
    "$python_bin" - "$url" <<'PY'
import re
import sys
from urllib.parse import urlparse

url = sys.argv[1]
path = ""
if url.startswith("git@github.com:"):
    path = url.split(":", 1)[1]
else:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host not in {"github.com", "www.github.com"}:
        raise SystemExit(f"open_pr.sh: origin is not a GitHub remote: {url}")
    path = parsed.path.lstrip("/")
path = re.sub(r"\.git$", "", path)
if not re.fullmatch(r"[^/]+/[^/]+", path):
    raise SystemExit(f"open_pr.sh: could not derive owner/repo from origin: {url}")
print(path)
PY
}

body_hash() {
    git hash-object "$body_file"
}

verify_body_hash() {
    local expected="$1" current
    current="$(body_hash)"
    if [ "$current" != "$expected" ]; then
        echo "open_pr.sh: PR body changed after review." >&2
        echo "Rerun this wrapper so local review covers the exact body being published." >&2
        exit 2
    fi
}

run_final_local_review() {
    echo "Running final local PR review before GitHub mutation..."
    ATLAS_CURRENT_PR_BODY_FILE="$body_file" \
        bash scripts/local_pr_review.sh --current-pr-body-file "$body_file"
}

acquire_mutation_lock() {
    local lock_path
    lock_path="$(git rev-parse --git-path open_pr_wrapper.lock)"
    exec 9>"$lock_path"
    if ! flock -n 9; then
        echo "open_pr.sh: another open_pr.sh mutation is already running in this checkout." >&2
        echo "Wait for that wrapper to finish, then rerun so review covers one writer's snapshot." >&2
        exit 2
    fi
}

existing_pr_number_for_branch() {
    local branch="$1" current_repo="$2" pr_json
    if ! pr_json="$(gh pr list --repo "$current_repo" --state open --head "$branch" --json number,headRefName,headRefOid,baseRefName,headRepository,isCrossRepository --limit 20)"; then
        echo "open_pr.sh: failed to query existing PR for head branch $branch" >&2
        exit 1
    fi
    "$python_bin" - "$branch" "$current_repo" "$pr_json" <<'PY'
import json
import sys

branch = sys.argv[1]
current_repo = sys.argv[2]

def repo_key(value):
    if not isinstance(value, str):
        return None
    parts = value.split("/")
    if len(parts) != 2 or not all(parts):
        return None
    return tuple(part.lower() for part in parts)

current_repo_key = repo_key(current_repo)
items = [item for item in json.loads(sys.argv[3]) if item.get("headRefName") == branch]
matches = [
    item
    for item in items
    if item.get("baseRefName") == "main"
    and repo_key(item.get("headRepository", {}).get("nameWithOwner")) == current_repo_key
    and not item.get("isCrossRepository")
]
if items and len(matches) != len(items):
    raise SystemExit(
        f"open_pr.sh: found open PRs for head branch {branch} outside {current_repo}->main; use gh manually"
    )
if len(matches) > 1:
    raise SystemExit(f"multiple open PRs found for head branch {branch}")
if matches:
    item = matches[0]
    print(f"{item['number']}\t{item.get('headRefOid', '')}")
PY
}

snapshot_head_oid() {
    local snapshot="$1"
    if [ "$snapshot" = "${snapshot#*$'\t'}" ]; then
        printf '\n'
    else
        printf '%s\n' "${snapshot#*$'\t'}"
    fi
}

verify_pr_snapshot_head() {
    local snapshot="$1" expected="$2" actual
    actual="$(snapshot_head_oid "$snapshot")"
    if [ "$actual" != "$expected" ]; then
        echo "open_pr.sh: existing PR head does not match reviewed head: reviewed $expected, PR reports ${actual:-<missing>}." >&2
        echo "Wait for GitHub to reflect the pushed branch or rerun after refreshing the PR." >&2
        exit 2
    fi
}

normalize_create_args() {
    trusted_create_args=()
    local arg value
    while [ "$#" -gt 0 ]; do
        arg="$1"
        shift
        case "$arg" in
            --base|-B)
                value="${1:-}"
                [ "$#" -gt 0 ] && shift
                [ "$value" = "main" ] && continue
                ;;
            --base=main|-Bmain)
                continue
                ;;
        esac
        trusted_create_args+=("$arg")
    done
}

refresh_base_ref

body_audit_args=(--base-ref origin/main)
if [ -n "${ATLAS_CURRENT_PR_AUTHOR:-}" ]; then
    body_audit_args+=(--pr-author "$ATLAS_CURRENT_PR_AUTHOR")
fi
"$python_bin" scripts/audit_pr_body.py "${body_audit_args[@]}" "$body_file"

for arg in "$@"; do
    case "$arg" in
        --body|--body-file|-b|-F)
            echo "open_pr.sh: pass the PR body as BODY_FILE, not via $arg" >&2
            exit 2
            ;;
    esac
done
reject_target_overrides "$@"

branch="$(git branch --show-current)"
if [ -z "$branch" ]; then
    echo "open_pr.sh: current checkout is detached; switch to a branch first" >&2
    exit 2
fi

acquire_mutation_lock

trusted_repo="$(origin_repo_name)"
reviewed_head="$(verify_published_head "$branch")"
reviewed_body_hash="$(body_hash)"
normalize_create_args "$@"

existing_pr_snapshot="$(existing_pr_number_for_branch "$branch" "$trusted_repo")"
existing_pr_number="${existing_pr_snapshot%%$'\t'*}"
if [ -n "$existing_pr_number" ]; then
    verify_pr_snapshot_head "$existing_pr_snapshot" "$reviewed_head"

    if [ "$#" -gt 0 ]; then
        echo "open_pr.sh: PR already exists for $branch; update body with no create args" >&2
        echo "Use gh pr edit manually for title/base/label changes." >&2
        exit 2
    fi

    if [ "${ATLAS_OPEN_PR_DRY_RUN:-}" = "1" ]; then
        echo "DRY RUN: gh pr edit $existing_pr_number --body-file - < $body_file"
        exit 0
    fi

    run_final_local_review
    verify_body_hash "$reviewed_body_hash"
    latest_existing_pr_snapshot="$(existing_pr_number_for_branch "$branch" "$trusted_repo")"
    if [ "$latest_existing_pr_snapshot" != "$existing_pr_snapshot" ]; then
        echo "open_pr.sh: existing PR identity changed after review; rerun this wrapper." >&2
        exit 2
    fi
    verify_pr_snapshot_head "$latest_existing_pr_snapshot" "$reviewed_head"
    verify_published_head "$branch" "$reviewed_head" >/dev/null
    gh pr edit "$existing_pr_number" --repo "$trusted_repo" --body-file - < "$body_file"
    verify_published_head "$branch" "$reviewed_head" >/dev/null
    verify_body_hash "$reviewed_body_hash"
    latest_existing_pr_snapshot="$(existing_pr_number_for_branch "$branch" "$trusted_repo")"
    if [ "$latest_existing_pr_snapshot" != "$existing_pr_snapshot" ]; then
        echo "open_pr.sh: existing PR identity changed during mutation; inspect before continuing." >&2
        exit 2
    fi
    verify_pr_snapshot_head "$latest_existing_pr_snapshot" "$reviewed_head"
else
    if [ "${ATLAS_OPEN_PR_DRY_RUN:-}" = "1" ]; then
        echo "DRY RUN: gh pr create ${trusted_create_args[*]} --repo $trusted_repo --base main --body-file - < $body_file"
        exit 0
    fi

    run_final_local_review
    verify_body_hash "$reviewed_body_hash"
    if [ -n "$(existing_pr_number_for_branch "$branch" "$trusted_repo")" ]; then
        echo "open_pr.sh: PR identity changed after review; rerun this wrapper." >&2
        exit 2
    fi
    verify_published_head "$branch" "$reviewed_head" >/dev/null
    gh pr create "${trusted_create_args[@]}" --repo "$trusted_repo" --base main --body-file - < "$body_file"
    verify_published_head "$branch" "$reviewed_head" >/dev/null
    verify_body_hash "$reviewed_body_hash"
    created_pr_snapshot="$(existing_pr_number_for_branch "$branch" "$trusted_repo")"
    if [ -z "$created_pr_snapshot" ]; then
        echo "open_pr.sh: created PR was not discoverable with the reviewed identity; inspect before continuing." >&2
        exit 2
    fi
    created_pr_head="${created_pr_snapshot#*$'\t'}"
    if [ "$created_pr_head" != "$reviewed_head" ]; then
        echo "open_pr.sh: created PR head changed during mutation; inspect before continuing." >&2
        exit 2
    fi
fi
