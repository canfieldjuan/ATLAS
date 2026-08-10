#!/usr/bin/env bash
# Run the local mechanical review bundle before opening or updating a PR.

set -euo pipefail

base_ref="origin/main"
base_ref_set=0
allow_dirty=0
current_pr_body_file="${ATLAS_CURRENT_PR_BODY_FILE:-}"
current_pr_author="${ATLAS_CURRENT_PR_AUTHOR:-}"
repo_root=""
script_root=""
python_bin="${PYTHON:-python3}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --repo-root)
            if [ "$#" -lt 2 ]; then
                echo "local_pr_review.sh: --repo-root requires a path" >&2
                exit 2
            fi
            repo_root="$2"
            shift 2
            ;;
        --script-root)
            if [ "$#" -lt 2 ]; then
                echo "local_pr_review.sh: --script-root requires a path" >&2
                exit 2
            fi
            script_root="$2"
            shift 2
            ;;
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
        --pr-author|--current-pr-author)
            if [ "$#" -lt 2 ]; then
                echo "local_pr_review.sh: $1 requires a GitHub login" >&2
                exit 2
            fi
            current_pr_author="$2"
            shift 2
            ;;
        --help|-h)
            cat <<'EOF'
Usage: bash scripts/local_pr_review.sh [--allow-dirty] [--repo-root PATH] [--script-root PATH] [--current-pr-body-file PATH] [--pr-author LOGIN] [base-ref]

Run the local mechanical review bundle before opening or updating a PR.
By default, the worktree must be clean so committed-diff checks cannot
silently ignore uncommitted edits.

By default, both roots are the current checkout. Trusted CI can execute scripts
from --script-root while inspecting --repo-root as data.

When running before the GitHub PR exists, pass --current-pr-body-file
with the PR description you plan to use. The drift audit validates that
body's Slice phase against the branch plan. Installed pre-push hooks can
use ATLAS_CURRENT_PR_BODY_FILE=PATH for the same check.

Trusted CI should also pass --pr-author (or ATLAS_CURRENT_PR_AUTHOR=LOGIN)
so PR-body contract exemptions match the standalone body gate.
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

run_local_unit_gate_mirror() {
    local tmp_dir
    local base_baseline
    local selected
    local merge_base
    local status
    local -a unit_gate_env
    local -a unit_gate_env_values
    tmp_dir="$(mktemp -d)"
    base_baseline="$tmp_dir/base_baseline.txt"
    selected="$tmp_dir/selected.txt"
    unit_gate_env=(-u ATLAS_CURRENT_PR_BODY_FILE -u ATLAS_CURRENT_PR_AUTHOR)
    unit_gate_env_values=(
        # A push performed by a gated test would otherwise re-enter the
        # pre-push hook, which re-runs this script, which pushes again.
        ATLAS_SKIP_LOCAL_PR_REVIEW=1
        ATLAS_DB_CONNECTION_STRING=
        ATLAS_DB_HOST=127.0.0.1
        ATLAS_DB_PORT=1
        ATLAS_DB_DATABASE=atlas
        ATLAS_DB_USER=atlas
        ATLAS_DB_PASSWORD=atlas_dev_password
        ATLAS_DB_SOCKET_PATH=
    )
    while IFS= read -r git_env_var; do
        [ -z "$git_env_var" ] && continue
        unit_gate_env+=("-u" "$git_env_var")
    done < <(git rev-parse --local-env-vars)
    while IFS= read -r pytest_env_var; do
        [ -z "$pytest_env_var" ] && continue
        unit_gate_env+=("-u" "$pytest_env_var")
    done < <(compgen -e PYTEST_)

    if [ ! -f "$repo_root/scripts/check_unit_gate.py" ]; then
        echo "scripts/check_unit_gate.py is absent from this PR head; local unit gate mirror cannot verify the required gate"
        rm -rf "$tmp_dir"
        return 1
    fi
    if [ ! -f "$script_root/scripts/check_unit_gate.py" ]; then
        echo "trusted unit-gate checker unavailable; local unit gate mirror cannot verify the required gate"
        rm -rf "$tmp_dir"
        return 1
    fi

    if merge_base="$(git merge-base "$base_ref" HEAD 2>/dev/null)"; then
        git show "${merge_base}:tests/unit_gate_baseline.txt" > "$base_baseline" 2>/dev/null || : > "$base_baseline"
    else
        git show "${base_ref}:tests/unit_gate_baseline.txt" > "$base_baseline" 2>/dev/null || : > "$base_baseline"
    fi

    if [ ! -f "$repo_root/scripts/select_impacted_tests.py" ]; then
        echo "selector absent from this PR head; running FULL"
        echo "FULL" > "$selected"
    elif [ ! -f "$script_root/scripts/select_impacted_tests.py" ]; then
        echo "trusted selector unavailable; running FULL"
        echo "FULL" > "$selected"
    else
        # The unit gate workflow does not receive wrapper-only PR body env,
        # Git hook-local env, or local pytest option overrides.
        # Drop them so local pre-push mirrors CI.
        env "${unit_gate_env[@]}" \
            "$python_bin" "$script_root/scripts/select_impacted_tests.py" --base "$base_ref" > "$selected"
        status=$?
        if [ "$status" -ne 0 ]; then
            echo "selector failed while choosing local unit-gate tests"
            rm -rf "$tmp_dir"
            return "$status"
        fi
    fi

    if [ -f "$repo_root/requirements.unit_gate.txt" ]; then
        echo "installing unit-gate test dependencies from requirements.unit_gate.txt"
        "$python_bin" -m pip install -r "$repo_root/requirements.unit_gate.txt"
    fi

    echo "--- selection ---"
    cat "$selected"

    if [ "$(cat "$selected")" = "FULL" ]; then
        echo "running the FULL suite (selection escalated)"
        env "${unit_gate_env[@]}" \
            "${unit_gate_env_values[@]}" \
            "$python_bin" "$script_root/scripts/check_unit_gate.py" \
            --baseline tests/unit_gate_baseline.txt \
            --base-baseline "$base_baseline"
        status=$?
    elif [ ! -s "$selected" ]; then
        echo "no test is reachable from the changed files; growth guard only"
        env "${unit_gate_env[@]}" \
            "${unit_gate_env_values[@]}" \
            "$python_bin" "$script_root/scripts/check_unit_gate.py" \
            --baseline tests/unit_gate_baseline.txt \
            --base-baseline "$base_baseline" \
            --growth-only
        status=$?
    else
        mapfile -t selected_tests < "$selected"
        echo "running ${#selected_tests[@]} impacted test file(s)"
        env "${unit_gate_env[@]}" \
            "${unit_gate_env_values[@]}" \
            "$python_bin" "$script_root/scripts/check_unit_gate.py" \
            --baseline tests/unit_gate_baseline.txt \
            --base-baseline "$base_baseline" \
            --selected-files "$selected" \
            --pytest-args "${selected_tests[@]}" \
                -m "not integration and not e2e" \
                --continue-on-collection-errors -rfE --tb=no -q \
                -p no:cacheprovider
        status=$?
    fi
    rm -rf "$tmp_dir"
    return "$status"
}

body_uses_docs_only_marker() {
    [ -n "$current_pr_body_file" ] || return 1
    [ -f "$current_pr_body_file" ] || return 1
    "$python_bin" - "$current_pr_body_file" <<'PY'
import re
import sys
from pathlib import Path

body = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
first = next((line.strip() for line in body.splitlines() if line.strip()), "")
raise SystemExit(0 if re.fullmatch(r"Docs-only:\s*true", first, re.IGNORECASE) else 1)
PY
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

if [ -n "$current_pr_body_file" ]; then
    if [ -f "$script_root/scripts/audit_pr_body.py" ]; then
        body_audit_args=("$script_root/scripts/audit_pr_body.py" --repo-root "$repo_root")
        body_audit_args+=(--base-ref "$base_ref")
        if [ -n "$current_pr_author" ]; then
            body_audit_args+=(--pr-author "$current_pr_author")
        fi
        body_audit_args+=("$current_pr_body_file")
        run_check "PR body contract" "$python_bin" "${body_audit_args[@]}"
    else
        echo
        echo "==> PR body contract"
        echo "    SKIP (scripts/audit_pr_body.py not found)"
    fi

fi

pre_push_args=("$script_root/scripts/pre_push_audit.sh" --repo-root "$repo_root" --script-root "$script_root")
if [ -n "$current_pr_author" ]; then
    pre_push_args+=(--pr-author "$current_pr_author")
fi
run_check "Pre-push audit wrapper" bash "${pre_push_args[@]}"

if [ -f "$script_root/scripts/audit_extracted_pipeline_ci_enrollment.py" ]; then
    run_check "Extracted pipeline CI enrollment" \
        "$python_bin" "$script_root/scripts/audit_extracted_pipeline_ci_enrollment.py" --atlas-brain-tests-from "$base_ref"
else
    echo
    echo "==> Extracted pipeline CI enrollment"
    echo "    SKIP (scripts/audit_extracted_pipeline_ci_enrollment.py not found)"
fi

if [ -f "$script_root/scripts/audit_pr_session_drift.py" ]; then
    drift_args=("$script_root/scripts/audit_pr_session_drift.py" "$base_ref" --require-current-pr-body)
    if [ -n "$current_pr_body_file" ]; then
        drift_args+=(--current-pr-body-file "$current_pr_body_file")
    fi
    run_check "Cross-session PR drift" "$python_bin" "${drift_args[@]}"
else
    echo
    echo "==> Cross-session PR drift"
    echo "    SKIP (scripts/audit_pr_session_drift.py not found)"
fi

if [ -f "$script_root/scripts/audit_cross_layer_callers.py" ]; then
    run_check "Cross-layer caller hints" "$python_bin" "$script_root/scripts/audit_cross_layer_callers.py" "$base_ref"
else
    echo
    echo "==> Cross-layer caller hints"
    echo "    SKIP (scripts/audit_cross_layer_callers.py not found)"
fi

if [ -f "$script_root/scripts/audit_ai_reconciliation.py" ]; then
    reconcile_args=("$script_root/scripts/audit_ai_reconciliation.py")
    if [ -n "$current_pr_body_file" ]; then
        reconcile_args+=(--current-pr-body-file "$current_pr_body_file")
        current_pr_author_lc="${current_pr_author,,}"
        if [[ "$current_pr_author_lc" != "dependabot[bot]" && "$current_pr_author_lc" != "app/dependabot" && "$current_pr_author_lc" != "dependabot" ]] && ! body_uses_docs_only_marker; then
            reconcile_args+=(--require)
        fi
    fi
    run_check "AI reconciliation record" "$python_bin" "${reconcile_args[@]}"
else
    echo
    echo "==> AI reconciliation record"
    echo "    SKIP (scripts/audit_ai_reconciliation.py not found)"
fi

if [ -n "$current_pr_body_file" ]; then
    if [ -f "$script_root/scripts/audit_fix_loop_disposition.py" ]; then
        run_check "Fix-loop disposition preflight" \
            "$python_bin" "$script_root/scripts/audit_fix_loop_disposition.py" \
                --repo-root "$repo_root" \
                --base-ref "$base_ref" \
                --current-pr-body-file "$current_pr_body_file"
    else
        echo
        echo "==> Fix-loop disposition preflight"
        echo "    SKIP (scripts/audit_fix_loop_disposition.py not found)"
    fi
fi

if [ -f "$script_root/scripts/check_guard_class_closure.py" ]; then
    guard_class_args=("$script_root/scripts/check_guard_class_closure.py" --base "$base_ref" --strict)
    if [ -n "$current_pr_body_file" ]; then
        run_check "Guard class-closure lint" env \
            ATLAS_CURRENT_PR_BODY_FILE="$current_pr_body_file" \
            "$python_bin" "${guard_class_args[@]}"
    else
        run_check "Guard class-closure lint" "$python_bin" "${guard_class_args[@]}"
    fi
else
    echo
    echo "==> Guard class-closure lint"
    echo "    SKIP (scripts/check_guard_class_closure.py not found)"
fi

committed_plan_docs=$(
    git diff --name-only --diff-filter=AM "$base"...HEAD -- 'plans/PR-*.md' 2>/dev/null |
        sort -u |
        grep -v '^$' || true
)

if [ -n "$committed_plan_docs" ]; then
    while IFS= read -r doc; do
        [ -z "$doc" ] && continue
        if [ -f "$script_root/scripts/audit_plan_code_consistency.py" ]; then
            run_check "Plan/code consistency: $doc" \
                "$python_bin" "$script_root/scripts/audit_plan_code_consistency.py" \
                    --base-ref "$base_ref" \
                    "$doc"
        fi
        if [ -f "$script_root/scripts/audit_review_rules_triggered.py" ]; then
            run_check "Reviewer rules triggered: $doc" \
                "$python_bin" "$script_root/scripts/audit_review_rules_triggered.py" "$base_ref" --plan "$doc"
        fi
    done <<< "$committed_plan_docs"
else
    echo
    echo "==> Plan/code consistency"
    echo "    SKIP (no committed plans/PR-*.md changed vs $base_ref)"
fi

run_check "git diff --check" git diff --check

if [ "${GITHUB_ACTIONS:-}" = "true" ]; then
    echo
    echo "==> Local unit gate mirror"
    echo "    SKIP (GitHub Actions runs .github/workflows/unit_gate.yml as its own required check)"
elif [ "$failures" -ne 0 ]; then
    echo
    echo "==> Local unit gate mirror"
    echo "    SKIP ($failures earlier local review check(s) failed)"
elif [ -f "$script_root/scripts/check_unit_gate.py" ] || git cat-file -e "$base:scripts/check_unit_gate.py" 2>/dev/null; then
    run_check "Local unit gate mirror" run_local_unit_gate_mirror
else
    echo
    echo "==> Local unit gate mirror"
    echo "    SKIP (scripts/check_unit_gate.py not found)"
fi

# Advisory only: nudge to archive merged plan docs once the plans/ root grows
# past the threshold. Never affects the failure count -- archive_plans.py check
# always exits 0, and the guard keeps set -e happy if it is ever absent.
echo
echo "==> Plans archive backlog (advisory, non-blocking)"
if [ -f "$script_root/scripts/archive_plans.py" ]; then
    "$python_bin" "$script_root/scripts/archive_plans.py" check || true
    echo "    advisory only -- after a PR merges, switch to a local main synced to origin/main,"
    echo "    then move only that plan by name:"
    echo "    git mv plans/PR-<Slice>.md plans/archive/ && ${python_bin} scripts/archive_plans.py index"
else
    echo "    SKIP (scripts/archive_plans.py not found)"
fi

echo
if [ "$failures" -eq 0 ]; then
    echo "local PR review passed"
    echo
    echo "Next: open/update the PR; Codex connector review and live-reconciliation own reviewer feedback."
    exit 0
fi

echo "$failures local review check(s) failed"
exit 1
