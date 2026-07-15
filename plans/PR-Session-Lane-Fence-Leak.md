# PR-Session-Lane-Fence-Leak

## Why this slice exists

Issue #2113 blocks the #2104 session-lane producer/enrollment path. The
session-lane parser currently accepts fenced example content as structural
plan metadata: a code block containing `## Scope (this PR)`, `Ownership lane:`,
and `Slice phase:` can satisfy `scripts/audit_pr_session_drift.py` even though
those lines are not real plan structure. That makes any future required
`session-lane` gate bypassable by fenced metadata.

This workflow/process slice fixes the parser class before producer or branch
protection enrollment continues. It is justified by #2113's live repro and by
#2110's explicit deferral that producer/enrollment must stay advisory or
blocked until this leak is closed.

### Problem-derived contract

- Root cause: `scope_lead_lines()` scans raw plan lines and treats any
  `## Scope`/`## <section>` line as structural, even while inside Markdown
  fences. Because `extract_plan_ownership_lanes()` and
  `extract_plan_slice_phases()` trust those raw Scope lead lines, a fenced fake
  Scope block can create a real lane and phase. The existing fixed-position
  metadata rule is sound only after the real Scope boundary is found; the
  boundary finder itself is not fence-aware.
- Correct fix must touch/change: Make Scope section boundary detection in
  `scripts/audit_pr_session_drift.py` ignore fenced headings by tracking the
  existing fence marker, and make the PR-body/session-drift fence helpers treat
  only real Markdown fences as structural fences: 0-3 leading spaces, valid
  backtick info strings, and bare closing fences. Preserve raw lead-line
  behavior after a real Scope heading is found. Add regression tests in
  `tests/test_audit_pr_session_drift.py` and `tests/test_audit_pr_body.py`
  proving a plan/body with only fenced metadata fails, both backtick and tilde
  fenced examples cannot create metadata, same-length inner fence openers do not
  reopen structural parsing, four-space indented code-block lookalikes and
  invalid backtick info strings do not hide valid real metadata, and closing
  fences with trailing text do not leak fenced sections. Enroll the
  session-drift test file in the standing pre-push tooling-test gate and add
  workflow coverage so the enrollment cannot silently disappear. Archive the
  merged #2110 plan by name and refresh the plan index.
- Must not change: Do not add the session-lane workflow, mutate branch
  protection, change `claude-review`, change PR-body header grammar or required
  sections, broaden Markdown parsing beyond the two lane-binding auditors, alter
  local session-state authorization, touch product behavior, touch protected S6
  or Dependabot/content lanes, or optimize Unit Gate runtime.

## Scope (this PR)

Ownership lane: dev-workflow/process-gate-enrollment
Slice phase: Workflow/process

1. Make `scope_lead_lines()` ignore `## Scope` and following heading-looking
   lines while a Markdown fence is open.
2. Restrict `FENCE_RE` to Markdown's 0-3-space fence indentation so indented
   code blocks do not open parser fences.
3. Reject backtick opening-fence lookalikes whose info string contains a
   backtick.
4. Make `closes_fence()` require a blank tail so inner opener lines such as
   code-fence-plus-language do not close the active fence.
5. Apply the same real-fence rules to the PR-body fenced-example skipper that
   issue #2111 identified as still live on main.
6. Keep fixed-position raw Scope lead semantics once a real Scope heading is
   found, including rejecting fences before the real lane declaration.
7. Add focused tests for the #2113/#2111 repros and fence variants.
8. Enroll `tests/test_audit_pr_session_drift.py` in the standing pre-push
   tooling-test workflow for both PR and main runs.
9. Move the merged Session Lane PreAdmission plan into `plans/archive/` and
   refresh the plan index.

### Review Contract

- Acceptance criteria:
  - [ ] A plan whose only Scope/lane/phase metadata appears inside a fenced
        block fails with missing real Scope metadata.
  - [ ] Backtick and tilde fences both prevent heading-looking fenced lines from
        becoming structural Scope boundaries.
  - [ ] Same-length inner fence opener lines with info strings do not close the
        active fence.
  - [ ] Four-space indented code-block fence lookalikes do not hide later real
        Scope metadata.
  - [ ] Backtick opening-fence lookalikes with backticks in their info strings
        do not hide later real metadata.
  - [ ] PR-body fenced sections do not leak through closing-fence lines with
        trailing text.
  - [ ] Valid real Scope lead metadata still passes.
  - [ ] Fenced content after a valid Scope lead remains non-duplicating.
  - [ ] Fences before the real lane declaration inside a real Scope still fail
        the fixed-position contract.
  - [ ] The session-drift regression file runs in the standing pre-push
        tooling-test workflow.
  - [ ] No session-lane workflow or branch-protection enrollment lands in this
        PR.
- Reachability proof: run the real `scripts/audit_pr_session_drift.py` CLI
  through the existing temporary-repo tests; the failing fixture must exercise
  the same branch-plan parsing path used by local/pre-push/session-lane audits.
- Affected surfaces: session-drift plan Scope boundary parsing, PR-body
  fenced-example skipping, focused tests, pre-push tooling-test enrollment, and
  plan archive housekeeping.
- Risk areas: accidentally accepting fenced fake headings, over-eliding raw
  Scope lead lines so fences before metadata no longer fail, or changing peer
  PR/body metadata behavior outside plan Scope parsing.
- Reviewer rules triggered: R1, R2, R10, R12, R13, R14.

### Files touched

- `.github/workflows/pre_push_audit.yml`
- `plans/INDEX.md`
- `plans/PR-Session-Lane-Fence-Leak.md`
- `plans/archive/PR-Session-Lane-PreAdmission.md`
- `scripts/audit_pr_body.py`
- `scripts/audit_pr_session_drift.py`
- `tests/test_audit_pr_body.py`
- `tests/test_audit_pr_session_drift.py`
- `tests/test_pre_push_audit_workflow.py`

## Mechanism

`scope_lead_lines()` uses the existing `FENCE_RE`, `open_fence_marker()`,
`opens_fence()`, and `closes_fence()` helpers while scanning raw lines.
`FENCE_RE` only matches fences indented by 0-3 spaces, matching Markdown fence
rules and excluding four-space indented code-block lookalikes. `opens_fence()`
rejects backtick-fence lookalikes whose info string contains a backtick. When
outside a fence, `scope_lead_lines()` may recognize `## Scope` and later section
headings. When inside a fence, it must ignore heading-looking lines for
section-boundary purposes and avoid collecting fenced lines as Scope lead
declarations unless a real Scope heading was already entered and the fence
itself is the raw lead content.

`closes_fence()` requires the matching delimiter to be followed only by
whitespace/end-of-line, so an inner opener such as code-fence-plus-language is
not treated as a close. `scripts/audit_pr_body.py` uses the same real-fence
rules so the #2111 live main PR-body leak closes alongside the session-drift
parser. This closes the fake-metadata class without changing lane grammar or
PR-body header grammar.

The regression tests use the existing CLI fixture harness so the proof goes
through branch-added plan discovery, not only a helper unit call. The merged
#2110 plan archive is a separate by-name teardown move in the same lane.

The pre-push tooling-test command now includes `tests/test_audit_pr_session_drift.py`
on both PR and main paths. `tests/test_pre_push_audit_workflow.py` asserts both
enrollments so this parser coverage remains attached to the standing gate.

## Intentional

- This is not the producer/enrollment PR. It only closes the parser leak that
  would make those future gates bypassable.
- The fix stays in the existing lightweight parser rather than introducing a
  new Markdown dependency for this narrow boundary.
- The code still treats fences before a real Scope lane declaration as content;
  that preserves the fixed-position contract from #2106.

## Deferred

- #2104/#2110 follow-up producer slice: add the actual session-lane workflow
  after this parser leak is closed.
- #2104 enrollment slice: add `session-lane` to branch protection only after
  this fix and producer burn-in.

Parked hardening: none.

## Verification

- New regression before fix: `test_cli_rejects_scope_metadata_inside_fenced_block` failed for backtick and tilde fences because fenced metadata produced a real lane and phase.
- Review regression before fix: `test_cli_rejects_scope_metadata_after_inner_fence_opener` failed for backtick and tilde fences because same-length inner opener lines closed the fence and reopened structural parsing.
- Review regression before fix: `test_cli_accepts_real_scope_after_indented_code_block_backticks` failed because a four-space indented code-block lookalike opened a parser fence and hid the later real Scope metadata.
- Review regression before fix: `test_cli_accepts_real_scope_after_invalid_backtick_fence_info` failed because an invalid backtick info string opened a parser fence and hid later Scope metadata.
- #2111 regressions before fix: `test_closing_fence_with_trailing_text_keeps_body_sections_fenced` and `test_backtick_fence_info_with_backtick_does_not_hide_body_sections` failed because the PR-body skipper accepted invalid closers and invalid backtick openers.
- `python -m pytest tests/test_audit_pr_session_drift.py::test_cli_rejects_scope_metadata_inside_fenced_block tests/test_audit_pr_session_drift.py::test_cli_ignores_fenced_headings_when_scoping_plan_lane tests/test_audit_pr_session_drift.py::test_cli_ignores_fenced_lane_example_after_scope_lead tests/test_audit_pr_session_drift.py::test_cli_rejects_fenced_slice_phase_in_plan_scope tests/test_audit_pr_session_drift.py::test_cli_rejects_scope_fence_before_lane_declaration tests/test_audit_pr_session_drift.py::test_cli_ignores_fenced_lane_example_outside_scope -q` — 8 passed.
- `python -m pytest tests/test_audit_pr_session_drift.py::test_cli_accepts_real_scope_after_indented_code_block_backticks tests/test_audit_pr_session_drift.py::test_cli_rejects_scope_metadata_after_inner_fence_opener tests/test_audit_pr_session_drift.py::test_cli_rejects_scope_metadata_inside_fenced_block tests/test_audit_pr_session_drift.py::test_cli_rejects_slice_phase_after_shorter_nested_fence tests/test_audit_pr_session_drift.py::test_cli_rejects_fenced_slice_phase_in_plan_scope tests/test_audit_pr_session_drift.py::test_cli_ignores_fenced_headings_when_scoping_plan_lane tests/test_audit_pr_session_drift.py::test_cli_ignores_fenced_lane_example_after_scope_lead -q` — 11 passed.
- `python -m pytest tests/test_audit_pr_session_drift.py tests/test_pre_push_audit_workflow.py -q` — 62 passed.
- `python -m pytest tests/test_audit_pr_session_drift.py::test_cli_accepts_real_scope_after_invalid_backtick_fence_info tests/test_audit_pr_body.py::test_closing_fence_with_trailing_text_keeps_body_sections_fenced tests/test_audit_pr_body.py::test_backtick_fence_info_with_backtick_does_not_hide_body_sections -q` — 5 passed.
- `python -m pytest tests/test_audit_pr_session_drift.py tests/test_audit_pr_body.py tests/test_pre_push_audit_workflow.py -q` — 120 passed.
- Direct #2113 helper repro now returns missing Ownership lane and missing Slice phase for fenced fake Scope metadata.
- Plan sync helper `scripts/sync_pr_plan.py` — passed and refreshed files/diff budget.
- `git diff --check` — passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/pre_push_audit.yml` | 4 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Session-Lane-Fence-Leak.md` | 191 |
| `plans/archive/PR-Session-Lane-PreAdmission.md` | 0 |
| `scripts/audit_pr_body.py` | 13 |
| `scripts/audit_pr_session_drift.py` | 26 |
| `tests/test_audit_pr_body.py` | 54 |
| `tests/test_audit_pr_session_drift.py` | 93 |
| `tests/test_pre_push_audit_workflow.py` | 6 |
| **Total** | **390** |
