#!/usr/bin/env bash
# pre_push_audit.sh - run mechanical pre-push checks.
#
# Runs (in order):
#   1. scripts/audit_claude_md_claims.py
#       MCP tool counts in CLAUDE.md vs @mcp.tool decorators.
#   2. scripts/audit_plan_doc.py <path>  (for each plans/PR-*.md
#       modified vs origin/main on this branch -- if any).
#   3. scripts/check_ascii_python.sh
#       ASCII-only .py policy.
#
# Exits 0 if all checks pass. Exits 1 if any failed.
#
# Usage:
#   bash scripts/pre_push_audit.sh
#
# Intended to be run manually before opening a PR. Wiring into a
# git pre-push hook is deferred to a follow-up slice.

set -u
cd "$(git rev-parse --show-toplevel)"

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

run_check "CLAUDE.md MCP tool counts" \
    python scripts/audit_claude_md_claims.py

# Find plan docs touched on this branch: committed vs origin/main +
# uncommitted in the working tree (untracked or modified).
base="$(git merge-base HEAD origin/main 2>/dev/null || echo HEAD~1)"
committed=$(git diff --name-only --diff-filter=AM "$base"...HEAD -- 'plans/PR-*.md' 2>/dev/null || true)
uncommitted=$(git status --porcelain -- 'plans/PR-*.md' 2>/dev/null | awk '{print $NF}' || true)
plan_docs=$(printf '%s\n%s\n' "$committed" "$uncommitted" | sort -u | grep -v '^$' || true)

if [ -n "$plan_docs" ]; then
    while IFS= read -r doc; do
        [ -z "$doc" ] && continue
        run_check "Plan doc: $doc" \
            python scripts/audit_plan_doc.py "$doc"
    done <<< "$plan_docs"
else
    echo
    echo "==> Plan docs"
    echo "    SKIP (no plans/PR-*.md added or modified vs $base or in working tree)"
fi

if [ -x scripts/check_ascii_python.sh ]; then
    run_check "ASCII Python policy" bash scripts/check_ascii_python.sh
else
    echo
    echo "==> ASCII Python policy"
    echo "    SKIP (scripts/check_ascii_python.sh not executable)"
fi

echo
if [ "$failures" -eq 0 ]; then
    echo "all checks passed"
    exit 0
else
    echo "$failures check(s) failed"
    exit 1
fi
