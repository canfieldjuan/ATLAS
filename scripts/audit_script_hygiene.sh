#!/usr/bin/env bash
# audit_script_hygiene.sh - lint our own audit scripts for recurring
# anti-patterns that Copilot reviewers have caught across PRs #483 /
# #484 / #485.
#
# Checks (any failure -> exit 1):
#   1. Every scripts/*.sh declares `set -euo pipefail` near the top.
#   2. Every Path(...).read_text(...) in scripts/audit_*.py includes
#      `encoding="utf-8"`.
#   3. No scripts/audit_*.py uses `startswith("/")` as an absolute-
#      path check (POSIX-only; misses Windows drives + UNC paths).
#      Use PurePosixPath/PureWindowsPath .is_absolute() instead.
#   4. No scripts/*.sh uses `[ -x SOMETHING.sh ]` followed by
#      `bash SOMETHING.sh` -- the exec bit is irrelevant when
#      invoking via `bash`, and on systems where the bit isn't set
#      the guard would spuriously skip the call. Use `[ -f ]`.
#
# Usage:
#   bash scripts/audit_script_hygiene.sh

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# The encoding + path-validation checks (2, 3) only apply to the
# pre-push-gate audit scripts -- the ones this AGENTS.md workflow
# is contracting. Preexisting B2B-pipeline audit scripts use
# different conventions and are out of scope. Maintain this list
# as new pre-push-gate auditors are added.
PRE_PUSH_GATE_AUDITS=(
    scripts/audit_claude_md_claims.py
    scripts/audit_plan_doc.py
    scripts/audit_extracted_manifests.py
    scripts/audit_mcp_tool_names_match_docs.py
    scripts/audit_review_source_count.py
    scripts/audit_plan_doc_files_touched.py
    scripts/audit_plan_doc_diff_size.py
    scripts/audit_mcp_port_assignments.py
    scripts/audit_plan_code_consistency.py
)

# The pre-push-gate shell scripts (set -euo pipefail check + [-x]
# mismatch check). audit_script_hygiene.sh itself is in scope.
PRE_PUSH_GATE_SHELL=(
    scripts/pre_push_audit.sh
    scripts/audit_script_hygiene.sh
)

fail=0
note() { echo "FAIL $1: $2"; fail=1; }

# ---------- 1. set -euo pipefail in pre-push-gate shell scripts ----------
for f in "${PRE_PUSH_GATE_SHELL[@]}"; do
    [ -f "$f" ] || continue
    # Allow the line anywhere in the first 25 lines (after shebang
    # and the docstring comment block).
    if ! head -25 "$f" | grep -qF 'set -euo pipefail'; then
        note "$f" "missing 'set -euo pipefail' in the first 25 lines"
    fi
done

# ---------- 2. encoding="utf-8" on every read_text() ----------
for f in "${PRE_PUSH_GATE_AUDITS[@]}"; do
    [ -f "$f" ] || continue
    # Match any `.read_text(...)` call whose argument list does NOT
    # mention an `encoding=` keyword. Two cases caught:
    #   .read_text()                   -- no args at all
    #   .read_text(errors="replace")   -- args present but no encoding=
    # Allowed:
    #   .read_text(encoding="utf-8")
    #   .read_text("utf-8")            -- positional (deliberately
    #                                     not allowed here; force the
    #                                     kw form so intent is explicit)
    if grep -nE '\.read_text\([^)]*\)' "$f" \
        | grep -v 'encoding=' \
        | head -3 > /tmp/_hygiene_bad.$$; then
        if [ -s /tmp/_hygiene_bad.$$ ]; then
            bad=$(cat /tmp/_hygiene_bad.$$)
            note "$f" "read_text() without encoding=\"utf-8\" kwarg:"$'\n'"$bad"
        fi
    fi
    rm -f /tmp/_hygiene_bad.$$
done

# ---------- 3. startswith("/") as absolute-path check ----------
for f in "${PRE_PUSH_GATE_AUDITS[@]}"; do
    [ -f "$f" ] || continue
    if grep -nE '\.startswith\("/"\)' "$f" >/dev/null; then
        bad=$(grep -nE '\.startswith\("/"\)' "$f" | head -3)
        note "$f" "startswith(\"/\") is POSIX-only; use PurePath.is_absolute():"$'\n'"$bad"
    fi
done

# ---------- 4. [ -x SCRIPT.sh ] -> bash SCRIPT.sh mismatch ----------
for f in "${PRE_PUSH_GATE_SHELL[@]}"; do
    [ -f "$f" ] || continue
    # Find non-comment lines like `[ -x foo.sh ]` whose next 5 lines
    # contain `bash foo.sh`. awk because grep can't easily span lines.
    # Ignores lines starting with "#" so the docstring example doesn't
    # match against itself.
    # awk uses literal substring (index) for the followup "bash <script>"
    # check so script paths containing regex metacharacters (e.g. the "."
    # in foo.sh) cannot produce false matches.
    if awk '
        /^[[:space:]]*#/ { next }
        /\[[[:space:]]+-x[[:space:]]+[^][]+\.sh[[:space:]]+\]/ {
            script = $0
            sub(/.*\[[[:space:]]+-x[[:space:]]+/, "", script)
            sub(/[[:space:]]+\].*/, "", script)
            target = script
            needle = "bash " target
            for (i = 0; i < 5 && (getline line) > 0; i++) {
                if (index(line, needle) > 0) {
                    print NR ": [ -x " target " ] followed by bash " target
                    exit 1
                }
            }
        }
        END { exit 0 }
    ' "$f"; then :; else
        note "$f" "'[ -x <script> ]' guard followed by 'bash <script>' (exec-bit irrelevant; use [ -f ])"
    fi
done

if [ "$fail" -eq 0 ]; then
    echo "audit_script_hygiene: all checks passed"
fi
exit "$fail"
