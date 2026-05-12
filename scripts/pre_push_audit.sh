#!/usr/bin/env bash
# Run mechanical audit checks before opening or updating a PR.

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

resolve_base_ref() {
    local ref
    if ref=$(git symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null); then
        echo "$ref"
        return 0
    fi
    if git rev-parse --verify origin/main >/dev/null 2>&1; then
        echo "origin/main"
        return 0
    fi
    return 1
}

if ! base_ref=$(resolve_base_ref); then
    echo "pre_push_audit.sh: could not resolve trunk base ref." >&2
    echo "tried: refs/remotes/origin/HEAD, origin/main" >&2
    exit 2
fi

base="$(git merge-base HEAD "$base_ref")"

run_check "CLAUDE.md MCP tool counts" python scripts/audit_claude_md_claims.py
run_check "MCP port assignments" python scripts/audit_mcp_port_assignments.py

committed=$(
    git diff --name-only --diff-filter=AM "$base"...HEAD -- 'plans/PR-*.md' 2>/dev/null || true
)
uncommitted=$(
    git status --porcelain -- 'plans/PR-*.md' 2>/dev/null |
        awk 'substr($0, 1, 2) !~ /D/ {print substr($0, 4)}' || true
)
committed_plan_docs=$(printf '%s\n' "$committed" | sort -u | grep -v '^$' || true)
uncommitted_plan_docs=$(printf '%s\n' "$uncommitted" | sort -u | grep -v '^$' || true)
plan_docs=$(printf '%s\n%s\n' "$committed_plan_docs" "$uncommitted_plan_docs" | sort -u | grep -v '^$' || true)

if [ -n "$plan_docs" ]; then
    while IFS= read -r doc; do
        [ -z "$doc" ] && continue
        run_check "Plan shape: $doc" python scripts/audit_plan_doc.py "$doc"
    done <<< "$plan_docs"

    while IFS= read -r doc; do
        [ -z "$doc" ] && continue
        run_check "Plan files touched: $doc" python scripts/audit_plan_doc_files_touched.py "$doc" "$base_ref"
        run_check "Plan diff size: $doc" python scripts/audit_plan_doc_diff_size.py "$doc" "$base_ref"
    done <<< "$committed_plan_docs"
else
    echo
    echo "==> Plan docs"
    echo "    SKIP (no plans/PR-*.md added or modified vs $base_ref or working tree)"
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
fi

echo "$failures check(s) failed"
exit 1
