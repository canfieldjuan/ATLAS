#!/usr/bin/env bash
# Update an existing PR body through the Atlas body-only fast path.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
python_bin="${PYTHON:-python3}"
export GH_PROMPT_DISABLED=1

usage() {
    cat <<'EOF'
Usage: bash scripts/update_pr_body.sh BODY_FILE

Updates the existing GitHub PR for the current branch. This is for body-only
edits after a PR is already open: AI reconciliation ledgers, verification
receipts, deferred notes, and wrapper-marker repair. It validates the same body
contract fields that a body-only edit can affect, verifies session ownership and
PR head identity, and publishes through stdin.

Use scripts/open_pr.sh for new PRs or after code changes.
EOF
}

die() {
    echo "update_pr_body.sh: $*" >&2
    exit 2
}

if [ "$#" -lt 1 ] || [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    usage
    exit 2
fi
input_body_file="$1"
shift
[ "$#" -eq 0 ] || die "body-only updates do not accept PR create/edit args"
[ -f "$input_body_file" ] || die "PR body file not found: $input_body_file"
[ -z "${GH_REPO:-}" ] || die "refusing GH_REPO target override: $GH_REPO"

open_pr_wrapper_marker="<!-- atlas-open-pr-wrapper: v1 -->"
body_file="$input_body_file"
temporary_body_file=""
cleanup_body() {
    [ -z "$temporary_body_file" ] || rm -f "$temporary_body_file"
}
trap cleanup_body EXIT

body_uses_docs_only_marker() {
    "$python_bin" - "$1" <<'PY'
import re
import sys
from pathlib import Path

body = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
first = next((line.strip() for line in body.splitlines() if line.strip()), "")
raise SystemExit(0 if re.fullmatch(r"Docs-only:\s*true", first, re.IGNORECASE) else 1)
PY
}

body_has_open_pr_wrapper_marker() {
    grep -Fxq "$open_pr_wrapper_marker" "$1"
}

body_author_is_dependabot() {
    local author_lc="${ATLAS_CURRENT_PR_AUTHOR:-}"
    author_lc="${author_lc,,}"
    [ "$author_lc" = "dependabot[bot]" ] || [ "$author_lc" = "app/dependabot" ] || [ "$author_lc" = "dependabot" ]
}

prepare_publish_body() {
    if body_author_is_dependabot || body_uses_docs_only_marker "$input_body_file" || body_has_open_pr_wrapper_marker "$input_body_file"; then
        body_file="$input_body_file"
        return 0
    fi
    temporary_body_file="$(mktemp "${TMPDIR:-/tmp}/atlas-update-pr-body.XXXXXX")"
    cp "$input_body_file" "$temporary_body_file"
    printf '\n%s\n' "$open_pr_wrapper_marker" >> "$temporary_body_file"
    body_file="$temporary_body_file"
}

body_hash() {
    git hash-object "$body_file"
}

verify_body_hash() {
    local expected="$1" current
    current="$(body_hash)"
    [ "$current" = "$expected" ] || die "PR body changed after review"
}

origin_repo_name() {
    "$python_bin" - <<'PY'
import re
import subprocess

url = subprocess.check_output(["git", "config", "--get", "remote.origin.url"], text=True).strip()
patterns = (
    r"^git@github\.com:(?P<repo>[^/]+/[^/]+?)(?:\.git)?$",
    r"^ssh://git@github\.com/(?P<repo>[^/]+/[^/]+?)(?:\.git)?$",
    r"^https://github\.com/(?P<repo>[^/]+/[^/]+?)(?:\.git)?$",
)
for pattern in patterns:
    match = re.match(pattern, url)
    if match:
        print(match.group("repo"))
        break
else:
    raise SystemExit(f"update_pr_body.sh: could not derive owner/repo from origin: {url}")
PY
}

refresh_base_ref() {
    echo "Refreshing origin/main before PR body audit..."
    git fetch --quiet origin main || die "failed to refresh origin/main"
}

verify_published_head() {
    local branch="$1" expected_sha="${2:-}" head_sha remote_sha
    head_sha="$(git rev-parse HEAD)"
    git fetch --quiet origin "refs/heads/$branch" || die "failed to refresh origin/$branch"
    remote_sha="$(git rev-parse FETCH_HEAD 2>/dev/null)" || die "current branch is not published at origin/$branch"
    if [ -n "$expected_sha" ] && [ "$head_sha" != "$expected_sha" ]; then
        die "current HEAD changed after review: reviewed $expected_sha, now $head_sha"
    fi
    if [ -n "$expected_sha" ] && [ "$remote_sha" != "$expected_sha" ]; then
        die "origin/$branch changed after review: reviewed $expected_sha, now $remote_sha"
    fi
    [ "$remote_sha" = "$head_sha" ] || die "origin/$branch is $remote_sha, current HEAD is $head_sha"
    printf '%s\n' "$head_sha"
}

existing_pr_number_for_branch() {
    local branch="$1" current_repo="$2" pr_json
    pr_json="$(gh pr list --repo "$current_repo" --state open --head "$branch" --json number,headRefName,headRefOid,baseRefName,headRepository,isCrossRepository --limit 20)" || {
        echo "update_pr_body.sh: failed to query existing PR for head branch $branch" >&2
        exit 1
    }
    "$python_bin" - "$branch" "$current_repo" "$pr_json" <<'PY'
import json
import sys

branch = sys.argv[1]
current_repo = sys.argv[2].lower()
items = [item for item in json.loads(sys.argv[3]) if item.get("headRefName") == branch]
matches = []
for item in items:
    repo = ((item.get("headRepository") or {}).get("nameWithOwner") or "").lower()
    if item.get("baseRefName") == "main" and repo == current_repo and not item.get("isCrossRepository"):
        matches.append(item)
if items and len(items) != len(matches):
    raise SystemExit(f"update_pr_body.sh: found open PRs for head branch {branch} outside {current_repo}->main")
if len(matches) != 1:
    raise SystemExit(f"update_pr_body.sh: expected exactly one open PR for branch {branch}, found {len(matches)}")
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
    [ "$actual" = "$expected" ] || die "existing PR head does not match reviewed head: reviewed $expected, PR reports ${actual:-<missing>}"
}

guard_existing_pr_ownership() {
    "$python_bin" scripts/check_session_pr_ownership.py \
        --pr "$1" \
        --branch "$2" \
        --head-sha "$3"
}

run_body_audits() {
    local checker_args=(--base-ref origin/main)
    if [ -n "${ATLAS_CURRENT_PR_AUTHOR:-}" ]; then
        checker_args+=(--pr-author "$ATLAS_CURRENT_PR_AUTHOR")
    fi
    if ! body_author_is_dependabot && ! body_uses_docs_only_marker "$body_file"; then
        checker_args+=(--require-wrapper-marker)
    fi
    "$python_bin" scripts/audit_pr_body.py "${checker_args[@]}" "$body_file"
    "$python_bin" scripts/audit_ai_reconciliation.py --current-pr-body-file "$body_file"
    "$python_bin" scripts/audit_fix_loop_disposition.py --repo-root . --current-pr-body-file "$body_file" --base-ref origin/main
}

run_live_reconciliation() {
    local repo="$1" pr_number="$2"
    if [ "${ATLAS_UPDATE_PR_BODY_SKIP_LIVE:-}" = "1" ]; then
        echo "live reconciliation: skipped by ATLAS_UPDATE_PR_BODY_SKIP_LIVE=1"
        return 0
    fi
    "$python_bin" scripts/check_ai_reconciliation_live.py --repo "$repo" --pr "$pr_number" --body-file "$body_file"
}

acquire_mutation_lock() {
    local lock_path
    lock_path="$(git rev-parse --git-path update_pr_body_wrapper.lock)"
    exec 9>"$lock_path"
    flock -n 9 || die "another update_pr_body.sh mutation is already running in this checkout"
}

prepare_publish_body
branch="$(git branch --show-current)"
[ -n "$branch" ] || die "current checkout is detached; switch to a branch first"
"$python_bin" scripts/check_pr_branch_name.py --branch "$branch" "$body_file"
refresh_base_ref
run_body_audits
trusted_repo="$(origin_repo_name)"
reviewed_head="$(verify_published_head "$branch")"
existing_pr_snapshot="$(existing_pr_number_for_branch "$branch" "$trusted_repo")"
existing_pr_number="${existing_pr_snapshot%%$'\t'*}"
verify_pr_snapshot_head "$existing_pr_snapshot" "$reviewed_head"
guard_existing_pr_ownership "$existing_pr_number" "$branch" "$reviewed_head"
run_live_reconciliation "$trusted_repo" "$existing_pr_number"
reviewed_body_hash="$(body_hash)"
acquire_mutation_lock
latest_snapshot="$(existing_pr_number_for_branch "$branch" "$trusted_repo")"
[ "$latest_snapshot" = "$existing_pr_snapshot" ] || die "existing PR identity changed before mutation; rerun this wrapper"
verify_pr_snapshot_head "$latest_snapshot" "$reviewed_head"
verify_published_head "$branch" "$reviewed_head" >/dev/null
verify_body_hash "$reviewed_body_hash"
if [ "${ATLAS_UPDATE_PR_BODY_DRY_RUN:-}" = "1" ]; then
    echo "DRY RUN: gh pr edit $existing_pr_number --repo $trusted_repo --body-file - < $body_file"
    exit 0
fi
gh pr edit "$existing_pr_number" --repo "$trusted_repo" --body-file - < "$body_file"
verify_published_head "$branch" "$reviewed_head" >/dev/null
verify_body_hash "$reviewed_body_hash"
