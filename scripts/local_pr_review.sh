#!/usr/bin/env bash
# Run the local mechanical review bundle before opening or updating a PR.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

base_ref="${1:-origin/main}"

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

base="$(git merge-base HEAD "$base_ref")"

echo "local PR review"
echo "base ref: $base_ref"
echo "merge base: $base"
echo
echo "changed files:"
git diff --name-status "$base"...HEAD || true

run_check "Pre-push audit wrapper" bash scripts/pre_push_audit.sh

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

echo
if [ "$failures" -eq 0 ]; then
    echo "local PR review passed"
    echo
    echo "Next: hand this branch to the local reviewer session for judgment review."
    exit 0
fi

echo "$failures local review check(s) failed"
exit 1
