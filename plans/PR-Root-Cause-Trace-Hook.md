# PR-Root-Cause-Trace-Hook

## Why this slice exists

The operator reported that Claude keeps responding to review comments by
patching the pointed-at symptom instead of tracing the defect upstream. That
creates repeated review rounds: each downstream patch exposes the next leaf
case, and the PR grows without closing the underlying defect class.

The current repo already says "root cause, not symptoms" in `AGENTS.md` and
`CLAUDE.md`, but the active prevention is incomplete. The Claude edit hook only
checks the allowed file set, and the PR-body fix-loop audit only checks root
decision / blocking predicate / disposition / allowed files / max files. Neither
mechanically requires a symptom-to-source trace before edits are made.

This is over the 400 LOC target because the enforcement is deliberately tested
from both entrypoints: the real Claude hook payloads and the PR-body local
review audit. Splitting the tests from the hook would leave the prevention rule
unproven.

### Problem-derived contract

- Root cause: Fix-mode enforcement stops file-set drift, not symptom-patching.
  An active fix loop can carry an allowed file list and still let Claude edit
  downstream files without a machine-readable source trace that names the
  upstream defect and the files where that source is fixed.
- Correct fix must touch/change: add root-trace fields to the Claude fix-mode
  baton enforcement, re-inject those fields after resume/compaction, update the
  `/fix-mode` Claude skill / `CLAUDE.md` contract, and make
  `scripts/audit_fix_loop_disposition.py` reject PR-body fix records that lack
  source trace / upstream file evidence.
- Must not change: product code, reviewer severity policy, live reconciliation
  semantics, the guard-class closure detector, or unrelated EOM/Atlas runtime
  lanes.

## Scope (this PR)

Ownership lane: workflow/root-cause-trace
Slice phase: Workflow/process

1. Enforce root-trace completeness in active constrained Claude fix mode before
   normal edit/write tools can patch files.
2. Enforce the same source-trace fields in PR-body fix-loop dispositions and
   cover upstream-root, downstream-only, and symptom-only-deferred cases with
   synthetic tests.
Max files: 10

### Review Contract

- Acceptance criteria:
  - `.claude/hooks/check_edit_budget.py` denies normal edit/write targets when
    fix mode is active with an allowed set but the baton lacks root-trace
    fields.
  - `.claude/hooks/check_edit_budget.py` allows an in-budget edit when the baton
    names the symptom, root cause, source trace, fix strategy, and upstream
    files.
  - Symptom-only fix strategy remains possible only with an explicit rationale
    and follow-up pointer in the baton and PR-body preflight.
  - `scripts/audit_fix_loop_disposition.py` rejects `fixed-in` PR-body records
    that lack source trace / upstream files / fix strategy, or whose changed
    files do not touch the declared upstream source.
  - Existing no-findings, waiver, allowed-file, and max-file fix-loop behavior
    remains intact.
- Reachability proof: Hook entrypoint is the real Claude `PreToolUse` command
  from `.claude/settings.json`; tests invoke that hook payload directly. Local
  review entrypoint is `scripts/local_pr_review.sh`, which already calls
  `scripts/audit_fix_loop_disposition.py` when a PR body file is supplied.
- Affected surfaces: `.claude/hooks/check_edit_budget.py`,
  `.claude/hooks/inject_fix_mode.py`, `.claude/skills/fix-mode/SKILL.md`,
  `AGENTS.md`, `CLAUDE.md`, `docs/SESSION_STATE_TEMPLATE.md`,
  `scripts/audit_fix_loop_disposition.py`, and focused tests.
- Risk areas: hook false positives that lock Claude out of fix mode, stale
  baton shape after compaction, PR-body audit rejecting valid waivers, and
  downstream-only fixed-in records slipping through.
- Reviewer rules triggered: R1, R2, R10, R12, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: Claude fix-mode edit-admission hook and PR-body
  fix-loop audit.
- Replaced-path behaviors: existing allowed-file admission remains, with a new
  prerequisite that an active constrained fix loop carry root-trace evidence.
- Guard-relevant fields: `symptom`, `root_cause`, `source_trace`,
  `fix_strategy`, `upstream_files`, `symptom_only_reason`, `follow_up`.
- Caller x input shape: `PreToolUse` JSON payloads for Edit/Write/MultiEdit and
  Markdown PR-body fix-loop records.

#### Closure Declaration

- Required trace-field set: CLOSED. Membership is the literal
  `_REQUIRED_ROOT_TRACE_FIELDS` tuple in `.claude/hooks/check_edit_budget.py`
  and the `trace_contract_errors` field checks in
  `scripts/audit_fix_loop_disposition.py`; unlisted fields may exist but cannot
  satisfy the required root-trace gate.
- Fix strategy set: CLOSED. Membership is `upstream-root` and
  `symptom-only-deferred`, defined in both hook/audit strategy inventories.
  Any unlisted strategy is rejected before edit or before PR-body publication.
- Upstream file set: OPEN per fix loop. Membership comes from the
  machine/human baton or PR-body `Upstream files:` record for that root
  decision. For `upstream-root`, an unlisted downstream target cannot stand in
  for the declared source; the hook requires the source target first or an
  already-changed upstream file, and the PR-body audit requires a changed
  upstream file before certifying `fixed-in`.
- Support target set: CLOSED. Membership is the hook's explicit support paths
  and prefixes for tests, plans, Claude skill docs, and session-control docs.
  Unlisted allowed targets are treated as downstream symptom targets.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: `.claude/settings.json` keeps the existing
  `PreToolUse` and `SessionStart` hook commands.
- Explicit value probe: test an active baton with complete root-trace fields.
- Absent value probe: test an active constrained baton missing root-trace fields.
- Default-session/default-context probe: existing no/inactive/malformed baton
  tests continue to fail open.
- Side-effect ordering: the hook checks root-trace completeness before allowed
  file matching, so the model receives the upstream-trace denial before it
  spends edits on a symptom patch.

### Files touched

- `.claude/hooks/check_edit_budget.py`
- `.claude/hooks/inject_fix_mode.py`
- `.claude/skills/fix-mode/SKILL.md`
- `AGENTS.md`
- `CLAUDE.md`
- `docs/SESSION_STATE_TEMPLATE.md`
- `plans/PR-Root-Cause-Trace-Hook.md`
- `scripts/audit_fix_loop_disposition.py`
- `tests/test_audit_fix_loop_disposition.py`
- `tests/test_fix_mode_hook.py`

## Mechanism

The Claude fix-mode baton becomes the pre-edit root-cause hook. When active
with an `allowed` set, the hook requires a root trace before permitting normal
edits. The trace fields are deliberately small and machine-readable:

- `symptom`: the failing check or review claim being addressed.
- `root_cause`: the upstream defect, not the visible leaf symptom.
- `source_trace`: the chain from symptom to source.
- `fix_strategy`: `upstream-root` or `symptom-only-deferred`.
- `upstream_files`: repo-relative files where the upstream source is fixed.
- `symptom_only_reason` and `follow_up`: required only for
  `symptom-only-deferred`.

The PR-body fix-loop audit enforces the same shape for pushed review-fix
records. For `upstream-root`, at least one declared upstream file must be part
of the branch diff; otherwise a builder can claim an upstream source while only
patching downstream.

## Intentional

- This is a workflow/process slice because the defect is agent behavior, not
  product behavior.
- The hook fails open when no active constrained baton exists, preserving normal
  non-fix-mode editing and the existing emergency ability to edit baton/session
  control files.
- The PR-body audit remains tied to actionable AI reconciliation, so clean
  first-pass PRs and docs-only/no-finding bodies are not burdened with fix-loop
  trace records.

## Deferred

- None.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_fix_mode_hook.py tests/test_audit_fix_loop_disposition.py -q`
  -- 50 passed.
- `ATLAS_SESSION_STATE_FILE=SESSION_STATE.codex-root-cause-trace.local.md bash scripts/local_pr_review.sh --current-pr-body-file tmp/pr-root-cause-trace-hook-body.md`
  -- passed. Local unit gate escalated to FULL because `.claude/hooks/check_edit_budget.py`
  is not mapped by the selector; result was 160 baseline failing/errored nodes,
  `regressions=0`, `newly-passing=0`.

## Estimated diff size

| File | LOC |
|---|---:|
| `.claude/hooks/check_edit_budget.py` | 110 |
| `.claude/hooks/inject_fix_mode.py` | 7 |
| `.claude/skills/fix-mode/SKILL.md` | 13 |
| `AGENTS.md` | 13 |
| `CLAUDE.md` | 12 |
| `docs/SESSION_STATE_TEMPLATE.md` | 8 |
| `plans/PR-Root-Cause-Trace-Hook.md` | 199 |
| `scripts/audit_fix_loop_disposition.py` | 60 |
| `tests/test_audit_fix_loop_disposition.py` | 239 |
| `tests/test_fix_mode_hook.py` | 118 |
| **Total** | **779** |
