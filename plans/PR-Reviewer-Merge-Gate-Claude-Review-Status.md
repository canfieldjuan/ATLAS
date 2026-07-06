# PR-Reviewer-Merge-Gate-Claude-Review-Status

## Why this slice exists

Operator request (2026-07-06): move the merge gate off the operator so the
active builder merges on green, and continues to the next slice, but only when
BOTH reviewers are clean -- Codex and Claude, not either alone.

**Problem-derived contract.** Codex's review is already a machine gate
(`live-reconciliation` reds while unaccounted bot threads exist). The Claude
Code reviewer runs as the operator's GitHub identity, so its verdict is prose
comments with no machine-checkable signal the builder can require. A correct fix
must: (a) promote the reviewer verdict to a per-SHA signal named `claude-review`
that a merge condition can require next to `live-reconciliation`; (b) make it
fail-closed by absence and reset on re-push; (c) state the two-gate requirement
in the builder merge condition (AGENTS.md 3c.1); (d) NOT itself grant merge
authority, flip branch protection, or change who merges -- those stay operator
actions.

## Scope (this PR)

Ownership lane: dev-workflow/reviewer-merge-gate
Slice phase: Workflow/process

1. Add the `claude-review` commit-status mechanism the reviewer sets after a
   review, and make the builder merge condition require it alongside
   `live-reconciliation`.
2. Prove it with unit tests over the status-argv builder and CLI, plus the
   AGENTS.md two-gate wording and a contract doc.

### Files touched

- `AGENTS.md`
- `docs/REVIEWER_MERGE_GATE.md`
- `plans/PR-Reviewer-Merge-Gate-Claude-Review-Status.md`
- `scripts/set_claude_review_status.py`
- `tests/test_set_claude_review_status.py`

## Mechanism

`set_claude_review_status.py` POSTs a commit status (`context=claude-review`) to
`repos/<repo>/statuses/<sha>` with `state` in {success, failure, pending}.
`success` = reviewed this head, no BLOCKER; `failure` = BLOCKER open; `pending`
= review in progress; absent = never reviewed (a new SHA from a re-push). The
context is hardcoded, so the tool can only set the reviewer gate. The builder
merge condition (AGENTS.md 3c.1 point 8) already requires "all
review/reconciliation gates clean"; this slice makes that mean BOTH
`live-reconciliation` and `claude-review`.

## Intentional

- Signal only. Making `claude-review` a required check (branch protection) and
  granting the builder standing merge authorization stay operator actions;
  until then the status is advisory (visible, non-gating). This keeps merge
  authority with the operator, per the constraint.
- Commit status (not a GitHub review APPROVE) because the reviewer shares the
  operator's GitHub identity, so an APPROVE would be indistinguishable from the
  operator's own; a named `claude-review` context is an unambiguous gate.
- Reviewer discipline is load-bearing: the status is only as honest as the
  reviewer setting it after an actual review. That is the same trust model as
  the existing prose LGTM, now made checkable.

## Deferred

- Surfacing `claude-review` inside `scripts/report_pr_watcher_state.py` output
  can come when the operator makes it a required check; it already appears in
  `gh pr checks` as a status context.

Parked hardening: none.

## Cold diff reconstruction

- `scripts/set_claude_review_status.py:1` adds a pure `build_gh_args` +
  `--dry-run` CLI that only ever emits `context=claude-review`.
- `tests/test_set_claude_review_status.py:1` proves the valid argv, rejects
  malformed state/repo/sha, and checks the dry-run CLI exits 0 / bad state
  exits 2.
- `AGENTS.md` 3c.1 point 8 now requires BOTH `live-reconciliation` and
  `claude-review`, and notes the re-push reset.
- `docs/REVIEWER_MERGE_GATE.md:1` documents the two-gate contract and the two
  remaining operator-owned steps.

## Verification

- Ran pytest over `tests/test_set_claude_review_status.py` -> 16 passed (0.15s).
- Dry-run of the setter emits the expected gh api POST to
  repos/<repo>/statuses/<sha> with context=claude-review and no network call.
- Ran `scripts/audit_pr_body.py` against this PR body before push -> PASS.

## Estimated diff size

| File | LOC |
|---|---:|
| `AGENTS.md` | 14 |
| `docs/REVIEWER_MERGE_GATE.md` | 65 |
| `plans/PR-Reviewer-Merge-Gate-Claude-Review-Status.md` | 105 |
| `scripts/set_claude_review_status.py` | 140 |
| `tests/test_set_claude_review_status.py` | 113 |
| **Total** | **437** |
