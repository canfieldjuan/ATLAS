#!/usr/bin/env bash
# Run mechanical audit checks before opening or updating a PR.

set -euo pipefail

repo_root=""
script_root=""
current_pr_author="${ATLAS_CURRENT_PR_AUTHOR:-}"
python_bin="${PYTHON:-python3}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --repo-root)
            if [ "$#" -lt 2 ]; then
                echo "pre_push_audit.sh: --repo-root requires a path" >&2
                exit 2
            fi
            repo_root="$2"
            shift 2
            ;;
        --script-root)
            if [ "$#" -lt 2 ]; then
                echo "pre_push_audit.sh: --script-root requires a path" >&2
                exit 2
            fi
            script_root="$2"
            shift 2
            ;;
        --pr-author)
            if [ "$#" -lt 2 ]; then
                echo "pre_push_audit.sh: --pr-author requires a GitHub login" >&2
                exit 2
            fi
            current_pr_author="$2"
            shift 2
            ;;
        --help|-h)
            cat <<'EOF'
Usage: bash scripts/pre_push_audit.sh [--repo-root PATH] [--script-root PATH] [--pr-author LOGIN]

Run mechanical audit checks before opening or updating a PR. By default, both
roots are the current checkout. Trusted CI can execute scripts from
--script-root while inspecting --repo-root as data.

Pass --pr-author (or ATLAS_CURRENT_PR_AUTHOR) in trusted CI so the plan
admission policy preserves the explicit Dependabot exemption.
EOF
            exit 0
            ;;
        --*)
            echo "pre_push_audit.sh: unknown option: $1" >&2
            exit 2
            ;;
        *)
            echo "pre_push_audit.sh: unexpected argument: $1" >&2
            exit 2
            ;;
    esac
done

if [ -z "$repo_root" ]; then
    repo_root="$(git rev-parse --show-toplevel)"
fi
repo_root="$(cd "$repo_root" && pwd)"

if [ -z "$script_root" ]; then
    script_root="$repo_root"
fi
script_root="$(cd "$script_root" && pwd)"

cd "$repo_root"
export ATLAS_AUDIT_REPO_ROOT="$repo_root"
export ATLAS_AUDIT_SCRIPT_ROOT="$script_root"

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

run_check "CLAUDE.md MCP tool counts" "$python_bin" "$script_root/scripts/audit_claude_md_claims.py"
run_check "MCP port assignments" "$python_bin" "$script_root/scripts/audit_mcp_port_assignments.py"
run_check "MCP tool-name inventories" "$python_bin" "$script_root/scripts/audit_mcp_tool_names_match_docs.py"
run_check "Extracted manifest sync" "$python_bin" "$script_root/scripts/audit_extracted_manifests.py"
run_check "UI test:* CI enrollment" "$python_bin" "$script_root/scripts/audit_ui_test_enrollment.py"
run_check "PR-side docs/test consistency" "$python_bin" "$script_root/scripts/audit_pr_side_docs_test_consistency.py" --repo-root "$repo_root"
run_check "PR watcher safety" "$python_bin" "$script_root/scripts/audit_pr_watcher_safety.py" --repo-root "$repo_root"
plan_admission_args=("$script_root/scripts/audit_pr_plan_presence.py" "$base_ref")
if [ -n "$current_pr_author" ]; then
    plan_admission_args+=(--pr-author "$current_pr_author")
fi
run_check "Plan admission" "$python_bin" "${plan_admission_args[@]}"

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
diff_plan_docs=$(
    comm -23 \
        <(printf '%s\n' "$committed_plan_docs" | grep -v '^$' || true) \
        <(printf '%s\n' "$uncommitted_plan_docs" | grep -v '^$' || true) || true
)

if [ -n "$plan_docs" ]; then
    while IFS= read -r doc; do
        [ -z "$doc" ] && continue
        run_check "Plan shape: $doc" "$python_bin" "$script_root/scripts/audit_plan_doc.py" "$doc"
    done <<< "$plan_docs"

    while IFS= read -r doc; do
        [ -z "$doc" ] && continue
        run_check "Plan files touched: $doc" "$python_bin" "$script_root/scripts/audit_plan_doc_files_touched.py" "$doc" "$base_ref"
        run_check "Plan diff size: $doc" "$python_bin" "$script_root/scripts/audit_plan_doc_diff_size.py" "$doc" "$base_ref"
    done <<< "$diff_plan_docs"
else
    echo
    echo "==> Plan docs"
    echo "    SKIP (no plans/PR-*.md added or modified vs $base_ref or working tree)"
fi

if [ -f "$script_root/scripts/check_ascii_python.sh" ]; then
    run_check "ASCII Python policy" bash "$script_root/scripts/check_ascii_python.sh"
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
