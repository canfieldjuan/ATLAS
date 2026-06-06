#!/usr/bin/env bash
# Run the local mechanical review bundle before opening or updating a PR.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

base_ref="origin/main"
base_ref_set=0
allow_dirty=0
current_pr_body_file="${ATLAS_CURRENT_PR_BODY_FILE:-}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --allow-dirty)
            allow_dirty=1
            shift
            ;;
        --current-pr-body-file|--pr-body-file)
            if [ "$#" -lt 2 ]; then
                echo "local_pr_review.sh: $1 requires a path" >&2
                exit 2
            fi
            current_pr_body_file="$2"
            shift 2
            ;;
        --help|-h)
            cat <<'EOF'
Usage: bash scripts/local_pr_review.sh [--allow-dirty] [--current-pr-body-file PATH] [base-ref]

Run the local mechanical review bundle before opening or updating a PR.
By default, the worktree must be clean so committed-diff checks cannot
silently ignore uncommitted edits.

When running before the GitHub PR exists, pass --current-pr-body-file
with the PR description you plan to use. The drift audit validates that
body's Slice phase against the branch plan. Installed pre-push hooks can
use ATLAS_CURRENT_PR_BODY_FILE=PATH for the same check.
EOF
            exit 0
            ;;
        --*)
            echo "local_pr_review.sh: unknown option: $1" >&2
            exit 2
            ;;
        *)
            if [ "$base_ref_set" -eq 1 ]; then
                echo "local_pr_review.sh: multiple base refs supplied" >&2
                exit 2
            fi
            base_ref="$1"
            base_ref_set=1
            shift
            ;;
    esac
done

failures=0

run_check() {
    local label="$1"
    shift
    echo
    echo "==> $label"
    if "$@"; then
        echo "    PASS"
    else
        echo "    FAIL"
        failures=$((failures + 1))
    fi
}

if ! git rev-parse --verify "$base_ref" >/dev/null 2>&1; then
    echo "local_pr_review.sh: base ref not found: $base_ref" >&2
    echo "fetch trunk first, or pass an explicit base ref" >&2
    exit 2
fi

if [ "$allow_dirty" -ne 1 ] && [ -n "$(git status --porcelain)" ]; then
    echo "local_pr_review.sh: worktree has uncommitted changes." >&2
    echo "Commit or stash them before running local review, or pass --allow-dirty for a partial/advisory run." >&2
    echo >&2
    git status --short >&2
    exit 1
fi

base="$(git merge-base HEAD "$base_ref")"

echo "local PR review"
echo "base ref: $base_ref"
echo "merge base: $base"
echo
echo "changed files:"
git diff --name-status "$base"...HEAD || true

run_check "Pre-push audit wrapper" bash scripts/pre_push_audit.sh

if [ -f scripts/audit_extracted_pipeline_ci_enrollment.py ]; then
    run_check "Extracted pipeline CI enrollment" \
        python scripts/audit_extracted_pipeline_ci_enrollment.py --atlas-brain-tests-from "$base_ref"
else
    echo
    echo "==> Extracted pipeline CI enrollment"
    echo "    SKIP (scripts/audit_extracted_pipeline_ci_enrollment.py not found)"
fi

if [ -f scripts/audit_pr_session_drift.py ]; then
    drift_args=(scripts/audit_pr_session_drift.py "$base_ref" --require-current-pr-body)
    if [ -n "$current_pr_body_file" ]; then
        drift_args+=(--current-pr-body-file "$current_pr_body_file")
    fi
    run_check "Cross-session PR drift" python "${drift_args[@]}"
else
    echo
    echo "==> Cross-session PR drift"
    echo "    SKIP (scripts/audit_pr_session_drift.py not found)"
fi

if [ -f scripts/audit_cross_layer_callers.py ]; then
    run_check "Cross-layer caller hints" python scripts/audit_cross_layer_callers.py "$base_ref"
else
    echo
    echo "==> Cross-layer caller hints"
    echo "    SKIP (scripts/audit_cross_layer_callers.py not found)"
fi

committed_plan_docs=$(
    git diff --name-only --diff-filter=AM "$base"...HEAD -- 'plans/PR-*.md' 2>/dev/null |
        sort -u |
        grep -v '^$' || true
)

if [ -n "$committed_plan_docs" ]; then
    while IFS= read -r doc; do
        [ -z "$doc" ] && continue
        if [ -f scripts/audit_plan_code_consistency.py ]; then
            run_check "Plan/code consistency: $doc" \
                python scripts/audit_plan_code_consistency.py "$doc"
        fi
    done <<< "$committed_plan_docs"
else
    echo
    echo "==> Plan/code consistency"
    echo "    SKIP (no committed plans/PR-*.md changed vs $base_ref)"
fi

run_check "git diff --check" git diff --check

# Advisory only: nudge to archive merged plan docs once the plans/ root grows
# past the threshold. Never affects the failure count -- archive_plans.py check
# always exits 0, and the guard keeps set -e happy if it is ever absent.
echo
echo "==> Plans archive backlog (advisory, non-blocking)"
if [ -f scripts/archive_plans.py ]; then
    python scripts/archive_plans.py check || true
    echo "    advisory only -- run 'python scripts/archive_plans.py archive' to archive merged plans"
else
    echo "    SKIP (scripts/archive_plans.py not found)"
fi

echo
if [ "$failures" -eq 0 ]; then
    echo "local PR review passed"
    echo
    echo "Next: hand this branch to the local reviewer session for judgment review."
    exit 0
fi

echo "$failures local review check(s) failed"
exit 1
