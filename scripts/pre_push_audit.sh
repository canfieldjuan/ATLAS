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

set -euo pipefail
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

# Find plan docs touched on this branch: committed vs the resolved
# trunk base + uncommitted in the working tree (untracked or modified).
#
# Trunk base resolution priority:
#   1. refs/remotes/origin/HEAD (the canonical default branch pointer,
#      typically origin/main but survives a rename).
#   2. origin/main literally, if it exists.
#
# Note: we intentionally do NOT use @{upstream} here. For a feature
# branch that pushes to its own name (e.g. claude/pr-foo ->
# origin/claude/pr-foo), @{upstream} is the branch's own remote
# tracker, not the trunk -- which makes the diff empty and the
# plan-doc audit SKIP silently.
#
# If neither tier resolves, fail with a clear message rather than
# silently falling back to HEAD~1 (which would skip plan-doc audits
# for any branch with more than one commit ahead).
resolve_base() {
    local ref
    if ref=$(git symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null); then
        echo "$ref"; return 0
    fi
    if git rev-parse --verify origin/main >/dev/null 2>&1; then
        echo "origin/main"; return 0
    fi
    return 1
}

if ! base_ref=$(resolve_base); then
    echo "pre_push_audit.sh: could not resolve a trunk base ref." >&2
    echo "  tried: refs/remotes/origin/HEAD, origin/main" >&2
    echo "  fix: 'git remote set-head origin -a' OR ensure origin/main exists" >&2
    exit 2
fi

base="$(git merge-base HEAD "$base_ref")"
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
    echo "    SKIP (no plans/PR-*.md added or modified vs $base_ref or in working tree)"
fi

if [ -f scripts/check_ascii_python.sh ]; then
    run_check "ASCII Python policy" bash scripts/check_ascii_python.sh
else
    echo
    echo "==> ASCII Python policy"
    echo "    SKIP (scripts/check_ascii_python.sh not found)"
fi

echo
if [ "$failures" -eq 0 ]; then
    echo "all checks passed"
    exit 0
else
    echo "$failures check(s) failed"
    exit 1
fi
