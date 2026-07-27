#!/usr/bin/env bash
# Create the initial AGENTS.md seven-section PR plan scaffold.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/new_pr_plan.sh SLICE --lane LANE [--phase PHASE] [--state-file PATH] [--force]

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
lane=""
lane_supplied=0
phase="TODO-slice-phase"
state_file=""
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
            lane_supplied=1
            ;;
        --state-file)
            shift
            [ "$#" -gt 0 ] || die "--state-file requires a value"
            state_file="$1"
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

if [ -z "$state_file" ]; then
    state_file="${ATLAS_SESSION_STATE_FILE:-$repo_root/SESSION_STATE.local.md}"
fi
case "$state_file" in
    /*) ;;
    *) state_file="$repo_root/$state_file" ;;
esac
[ -f "$state_file" ] || die "session state file not found: $state_file"

current_lane_count="$(awk '
    /^##[[:space:]]/ { exit }
    /^Current lane:/ { count += 1 }
    END { print count + 0 }
' "$state_file")"
[ "$current_lane_count" -eq 1 ] || die "session state must contain exactly one top-level Current lane: entry: $state_file"
current_lane="$(awk '
    /^##[[:space:]]/ { exit }
    /^Current lane:/ {
        sub(/^Current lane:[[:space:]]*/, "")
        print
        exit
    }
' "$state_file")"
[ -n "$current_lane" ] || die "session state Current lane: must be non-empty: $state_file"
case "$current_lane" in
    "<"*">"|TODO*|none|None)
        die "session state Current lane: must name an assigned lane: $state_file"
        ;;
esac
[ "$lane_supplied" -eq 1 ] || die "--lane is required and must match session Current lane: $current_lane"
[ "$lane" = "$current_lane" ] || die "lane mismatch: --lane $lane does not match session Current lane: $current_lane"

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

### Problem-derived contract

- Root cause: TODO: State what is actually wrong, and why, from the problem alone.
- Correct fix must touch/change: TODO: Name the modules, contracts, tests, and behaviors the fix must change to reach that cause.
- Must not change: TODO: Name modules, behaviors, product shape, contracts, and adjacent lanes this work must leave alone.

## Scope (this PR)

Ownership lane: $lane
Slice phase: $phase

1. TODO: Name the narrow behavior this PR changes.
2. TODO: Name the proof this PR adds.

### Review Contract

- Acceptance criteria: TODO: List outcomes the reviewer checks one by one. Each
  names a claim about the code, or the evidence that settles it (a file:line, a
  command + output, a CI job). Do NOT name a BARE risk category ("no TOCTOU",
  "no race conditions", "handles every malformed input") -- those name a hazard
  with nothing to look at, and the reviewer will fail authoring
  until the builder names the code claim or the evidence that settles it.
  Naming the evidence rescues it: "no unmasked email addresses in the audit
  export -- settled by tests/test_audit_export.py::test_masks_email_addresses"
  is fine. For open-input criteria, reference the 3k.3 evidence-gated mechanism;
  a sampled fixture list alone is not enough. For concurrency/open-execution
  criteria, reference the 3k.4 execution model and property-level invariant; a
  sampled concurrent test alone is not enough. See AGENTS.md 1a.
- Reachability proof: TODO: Name the real entrypoint and observable effect, or
  N/A with a reason for a surface-free change.
- Affected surfaces: TODO: Name the modules, workflows, contracts, and callers
  in scope.
- Risk areas: TODO: Name the regression or boundary risks the reviewer probes.
  Categories are fine here -- this field sets probe depth and is not
  dispositioned by the review matrix.
- Reviewer rules triggered: TODO: List the applicable R1-R14 rule IDs.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: TODO/N/A.
- Replaced-path behaviors: TODO/N/A.
- Guard-relevant fields: TODO/N/A.
- Caller x input shape: TODO/N/A.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: TODO/N/A.
- Explicit value probe: TODO/N/A.
- Absent value probe: TODO/N/A.
- Default-session/default-context probe: TODO/N/A.
- Side-effect ordering: TODO/N/A.

### Decision recording

Required when citing an operator decision that re-scopes an umbrella issue;
otherwise write "N/A - no re-scoping operator decision cited."

- Recorded decision URL: TODO/N/A.
- Umbrella issue: TODO/N/A.
- Scope effect: TODO/N/A.

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
- CI-equivalent command copied from enforcing workflow: TODO/N/A.
- Copied from enforcing workflow: TODO/N/A.
- No enforcing workflow applies: TODO/N/A.
- Closest local command: TODO/N/A.

## Estimated diff size

| File | LOC |
|---|---:|
| **Total** | **0** |
EOF

mv "$tmp" "$plan_path"
trap - EXIT
echo "created plan scaffold: $plan_rel"
