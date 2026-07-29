#!/usr/bin/env bash
# Push a PR branch with the PR body env wired into exactly one local review.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
repo_root="$(pwd)"

usage() {
    cat <<'EOF'
Usage: bash scripts/push_pr.sh BODY_FILE [git-push-args...]

Runs local review once in a temporary worktree pinned to the captured HEAD,
then pushes with ATLAS_CURRENT_PR_BODY_FILE exported for hook/body context.

Examples:
  bash scripts/push_pr.sh tmp/pr-body-my-slice.md -u origin HEAD
  bash scripts/push_pr.sh tmp/pr-body-my-slice.md -u origin claude/pr-my-slice
EOF
}

body_sha256() {
    python - "$1" <<'PY'
import hashlib
import sys
from pathlib import Path

print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
}

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
    cleanup_review_worktree
}

cleanup_review_worktree() {
    if [ -n "${review_worktree:-}" ]; then
        git worktree remove --force "$review_worktree" >/dev/null 2>&1 || true
        if [ -n "${review_parent:-}" ]; then
            rmdir "$review_parent" >/dev/null 2>&1 || true
        fi
        review_worktree=""
        review_parent=""
    fi
}

current_branch() {
    local branch
    branch="$(git branch --show-current)"
    if [ -z "$branch" ]; then
        echo "push_pr.sh: current checkout is detached; switch to a branch before pushing" >&2
        exit 2
    fi
    printf '%s\n' "$branch"
}

capture_proof_inputs() {
    proof_branch="$(current_branch)"
    proof_head_sha="$(git rev-parse HEAD)"
    proof_base_sha="$(git rev-parse origin/main)"
    proof_body_hash="$(body_sha256 "$body_snapshot")"
}

reject_unproofable_push_args() {
    local branch="$1"
    local saw_remote=0
    local saw_refspec=0
    local arg
    shift

    while [ "$#" -gt 0 ]; do
        arg="$1"
        shift
        case "$arg" in
            --dr*)
                echo "push_pr.sh: refusing git push dry-run; local review proof requires a real remote update" >&2
                exit 2
                ;;
            --no-veri*)
                echo "push_pr.sh: refusing to forward --no-verify or its Git-abbreviated spellings; local PR review must run once" >&2
                exit 2
                ;;
            --repo|--repo=*)
                echo "push_pr.sh: refusing git push target override: $arg" >&2
                echo "Use the default origin remote so the local review proof can verify origin/$branch." >&2
                exit 2
                ;;
            --receive-pack|--exec|--recurse-submodules|--push-option)
                if [ "$#" -eq 0 ]; then
                    echo "push_pr.sh: $arg requires a value" >&2
                    exit 2
                fi
                shift
                continue
                ;;
            --receive-pack=*|--exec=*|--recurse-submodules=*|--push-option=*)
                continue
                ;;
            --*)
                continue
                ;;
            -o)
                if [ "$#" -eq 0 ]; then
                    echo "push_pr.sh: -o requires a value" >&2
                    exit 2
                fi
                shift
                continue
                ;;
            -o?*)
                continue
                ;;
            -*)
                if [[ "${arg#-}" == *n* ]]; then
                    echo "push_pr.sh: refusing git push dry-run; local review proof requires a real remote update" >&2
                    exit 2
                fi
                continue
                ;;
        esac

        if [ "$saw_remote" -eq 0 ]; then
            if [ "$arg" != "origin" ]; then
                echo "push_pr.sh: refusing git push remote $arg; local review proof must publish origin/$branch" >&2
                echo "Use: bash scripts/push_pr.sh \"$body_file\" -u origin HEAD" >&2
                exit 2
            fi
            saw_remote=1
            continue
        fi

        saw_refspec=1
        case "$arg" in
            HEAD|"$branch"|HEAD:"$branch"|HEAD:refs/heads/"$branch"|"$branch":"$branch"|"$branch":refs/heads/"$branch")
                ;;
            *)
                echo "push_pr.sh: refusing to write proof for refspec that does not publish the current branch HEAD: $arg" >&2
                echo "Use: bash scripts/push_pr.sh \"$body_file\" -u origin HEAD" >&2
                exit 2
                ;;
        esac
    done

    if [ "$saw_refspec" -eq 0 ]; then
        echo "push_pr.sh: refusing ambiguous push args; pass an explicit current-branch refspec such as HEAD" >&2
        exit 2
    fi
}

verify_current_head_published() {
    local branch="$1" head_sha remote_sha
    head_sha="$(git rev-parse HEAD)"
    if ! remote_sha="$(git rev-parse "refs/remotes/origin/$branch" 2>/dev/null)"; then
        echo "push_pr.sh: pushed branch proof missing refs/remotes/origin/$branch" >&2
        echo "Fetch or push the current branch with: bash scripts/push_pr.sh \"$body_file\" -u origin HEAD" >&2
        exit 2
    fi
    if [ "$remote_sha" != "$head_sha" ]; then
        echo "push_pr.sh: refusing local review proof; origin/$branch is $remote_sha, current HEAD is $head_sha" >&2
        echo "Push the current branch HEAD before opening or updating the PR." >&2
        exit 2
    fi
}

verify_proof_inputs_unchanged() {
    local branch head_sha base_sha body_hash
    branch="$(current_branch)"
    head_sha="$(git rev-parse HEAD)"
    base_sha="$(git rev-parse origin/main)"
    body_hash="$(body_sha256 "$body_snapshot")"
    if [ "$branch" != "$proof_branch" ]; then
        echo "push_pr.sh: refusing local review proof; branch changed from $proof_branch to $branch" >&2
        exit 2
    fi
    if [ "$head_sha" != "$proof_head_sha" ]; then
        echo "push_pr.sh: refusing local review proof; HEAD changed from $proof_head_sha to $head_sha" >&2
        exit 2
    fi
    if [ "$base_sha" != "$proof_base_sha" ]; then
        echo "push_pr.sh: refusing local review proof; origin/main changed from $proof_base_sha to $base_sha" >&2
        echo "Rerun scripts/push_pr.sh so local review covers the current base." >&2
        exit 2
    fi
    if [ "$body_hash" != "$proof_body_hash" ]; then
        echo "push_pr.sh: refusing local review proof; reviewed body snapshot changed" >&2
        exit 2
    fi
}

verify_source_worktree_clean() {
    local dirty
    dirty="$(git status --porcelain)"
    if [ -n "$dirty" ]; then
        echo "push_pr.sh: source worktree has uncommitted changes; commit or stash before pushing" >&2
        echo "$dirty" >&2
        exit 1
    fi
}

write_local_review_proof() {
    local proof_path proof_dir
    proof_path="$(git rev-parse --git-path atlas-local-pr-review-proof)"
    proof_dir="$(dirname "$proof_path")"
    mkdir -p "$proof_dir"
    {
        printf 'branch=%s\n' "$proof_branch"
        printf 'head_sha=%s\n' "$proof_head_sha"
        printf 'base_sha=%s\n' "$proof_base_sha"
        printf 'body_sha256=%s\n' "$proof_body_hash"
    } > "$proof_path"
    echo "Wrote local review proof for $proof_head_sha: $proof_path"
}

run_immutable_local_review() {
    local state_file="${ATLAS_SESSION_STATE_FILE:-}"
    if [ -n "$state_file" ] && [ "${state_file#/}" = "$state_file" ]; then
        state_file="$repo_root/$state_file"
    fi
    review_parent="$(mktemp -d "${TMPDIR:-/tmp}/atlas-pr-review-worktree.XXXXXX")"
    review_worktree="$review_parent/worktree"
    git worktree add --quiet --detach "$review_worktree" "$proof_head_sha"
    echo "Running local PR review in immutable worktree for $proof_head_sha"
    (
        cd "$review_worktree"
        if [ -n "$state_file" ]; then
            export ATLAS_SESSION_STATE_FILE="$state_file"
        fi
        export GITHUB_HEAD_REF="$proof_branch"
        ATLAS_CURRENT_PR_BODY_FILE="$body_snapshot" \
            bash scripts/local_pr_review.sh --current-pr-body-file "$body_snapshot"
    )
    cleanup_review_worktree
}

refresh_base_ref() {
    echo "Refreshing origin/main before local PR review..."
    if ! git fetch --quiet origin main; then
        echo "push_pr.sh: failed to refresh origin/main; fetch/rebase before pushing" >&2
        exit 1
    fi
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

body_snapshot="$(snapshot_body "$body_file")"
trap cleanup_body_snapshot EXIT

if [ "$#" -eq 0 ]; then
    set -- -u origin HEAD
fi

branch="$(current_branch)"
reject_unproofable_push_args "$branch" "$@"

if [ "${ATLAS_PUSH_PR_DRY_RUN:-}" = "1" ]; then
    echo "DRY RUN: git fetch --quiet origin main"
    echo "DRY RUN: ATLAS_CURRENT_PR_BODY_FILE=$body_file bash scripts/local_pr_review.sh --current-pr-body-file $body_file in immutable captured-head worktree"
    echo "DRY RUN: ATLAS_SKIP_LOCAL_PR_REVIEW=1 ATLAS_CURRENT_PR_BODY_FILE=$body_file git push $*"
    exit 0
fi

refresh_base_ref
capture_proof_inputs

body_audit_args=(--base-ref origin/main)
if [ -n "${ATLAS_CURRENT_PR_AUTHOR:-}" ]; then
    body_audit_args+=(--pr-author "$ATLAS_CURRENT_PR_AUTHOR")
fi
python scripts/audit_pr_body.py "${body_audit_args[@]}" "$body_snapshot"

verify_source_worktree_clean
run_immutable_local_review

echo "Pushing with PR body env available to pre-push hook: $body_file"
ATLAS_SKIP_LOCAL_PR_REVIEW=1 ATLAS_CURRENT_PR_BODY_FILE="$body_snapshot" git push "$@"
verify_source_worktree_clean
verify_proof_inputs_unchanged
verify_current_head_published "$branch"
write_local_review_proof
