#!/usr/bin/env bash
# Create the initial AGENTS.md seven-section PR plan scaffold.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/new_pr_plan.sh SLICE [--lane LANE] [--phase PHASE] [--force]

Creates plans/PR-<SLICE>.md with the required AGENTS.md plan sections.
SLICE may be passed with or without the PR- prefix.

Examples:
  bash scripts/new_pr_plan.sh Content-Ops-Thing --lane content-ops/example --phase "Vertical slice"
  bash scripts/new_pr_plan.sh PR-Dev-Workflow-Plan-Scaffold --lane dev-workflow/pr-prep-ergonomics --phase Workflow/process
EOF
}

die() {
    echo "new_pr_plan.sh: $*" >&2
    exit 2
}

slice=""
lane="TODO-ownership-lane"
phase="TODO-slice-phase"
force=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --force)
            force=1
            ;;
        --lane)
            shift
            [ "$#" -gt 0 ] || die "--lane requires a value"
            lane="$1"
            ;;
        --phase)
            shift
            [ "$#" -gt 0 ] || die "--phase requires a value"
            phase="$1"
            ;;
        --*)
            die "unknown option: $1"
            ;;
        *)
            if [ -n "$slice" ]; then
                die "expected one slice name, got extra argument: $1"
            fi
            slice="$1"
            ;;
    esac
    shift
done

[ -n "$slice" ] || {
    usage >&2
    die "missing slice name"
}

case "$slice" in
    */*|*\\*|.*|*..*)
        die "unsafe slice name: $slice"
        ;;
esac

case "$slice" in
    PR-*) plan_name="$slice" ;;
    *) plan_name="PR-$slice" ;;
esac

[ "$plan_name" != "PR-" ] || die "slice name must include text after PR-"

case "$plan_name" in
    *[!A-Za-z0-9._-]*)
        die "slice name may contain only letters, numbers, dot, underscore, and dash: $slice"
        ;;
esac

repo_root="$(git rev-parse --show-toplevel 2>/dev/null)" || die "not inside a git worktree"
plan_rel="plans/$plan_name.md"
plan_path="$repo_root/$plan_rel"

if [ -e "$plan_path" ] && [ "$force" -ne 1 ]; then
    die "plan already exists: $plan_rel (pass --force to overwrite)"
fi

mkdir -p "$repo_root/plans"
tmp="$(mktemp "$repo_root/plans/.new-pr-plan.XXXXXX")"
trap 'rm -f "$tmp"' EXIT

cat > "$tmp" <<EOF
# $plan_name

## Why this slice exists

TODO: Tie this slice to a concrete user request, prior plan, audit finding, or
review comment.

## Scope (this PR)

Ownership lane: $lane
Slice phase: $phase

1. TODO: Name the narrow behavior this PR changes.
2. TODO: Name the proof this PR adds.

### Files touched

- TODO: run \`python scripts/sync_pr_plan.py $plan_rel\` after implementation.

## Mechanism

TODO: Explain how the change works so the reviewer does not have to
reverse-engineer the diff.

## Intentional

- TODO: Name explicit trade-offs or rejected alternatives.

## Deferred

- TODO: Name follow-up work, or replace this with "None."

Parked hardening: none.

## Verification

- Pending before push: TODO.

## Estimated diff size

| File | LOC |
|---|---:|
| **Total** | **0** |
EOF

mv "$tmp" "$plan_path"
trap - EXIT
echo "created plan scaffold: $plan_rel"
