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

### Review Contract

Acceptance criteria (reviewer checks one by one):
- The setter only ever emits `context=claude-review` (hardcoded); it cannot set
  another check. Rejects invalid state/repo, and any non-full-40-char SHA.
- `--dry-run` makes no network call; a missing `gh` binary exits 2, not a raw
  traceback.
- `tests/test_set_claude_review_status.py` is enrolled in
  `.github/workflows/pre_push_audit.yml` in BOTH the PR and push paths.
- AGENTS.md 3c.1 point 8 requires BOTH `live-reconciliation` and `claude-review`
  and states the forgeability trust boundary; the reviewer flow (AGENTS.md 1)
  now tells the reviewer to set the status after the verdict.

Affected surfaces: dev-workflow tooling only (a script, a doc, AGENTS.md
merge-condition + reviewer-flow text, and the CI pytest list). No runtime,
product, billing, delivery, report, or public contract surface.

Risk areas: the status is forgeable by any `status:write` token (documented as
the trust boundary; real enforcement deferred to a distinct reviewer identity).
The setter shells out to `gh`.

Reviewer rule IDs: guard-shaped input validation on the setter -> boundary /
second-side checks on the state/repo/sha validators; the merge-gate wording is a
contract change -> verify the two-gate and trust-boundary claims against the
code, not the description.

Reachability proof: the setter was run live against #2028 and #2026 head SHAs;
the `claude-review` status appeared in `gh pr checks` with the posted state,
proving the gate surface is wired end to end.

### Files touched

- `.github/workflows/pre_push_audit.yml`
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
- `.github/workflows/pre_push_audit.yml` enrolls
  `tests/test_set_claude_review_status.py` in both the PR-path and push-path
  pytest lists, so a future break of the setter reds the required tooling
  workflow instead of passing silently (the test file was omitted before).
- `scripts/set_claude_review_status.py` tightens the SHA guard to a full
  40-char hex (the GitHub statuses API rejects abbreviations), fixing a footgun
  that let an abbreviated `--sha` pass validation and then fail the live call.
- `scripts/set_claude_review_status.py` now catches `FileNotFoundError` when
  `gh` is absent and exits 2 with a message, instead of a raw traceback (P3).
- `AGENTS.md` 3c.1 point 8 adds the forgeability trust boundary: `claude-review`
  is a plain commit status, forgeable by any `status:write` token, so it is a
  coordination signal until posted from a distinct reviewer identity; the
  operator must not grant a forge-capable builder merge authority on it (P2).
- `AGENTS.md` reviewer flow (section 1) now tells the reviewer to set
  `claude-review` after the verdict, and `docs/REVIEWER_MERGE_GATE.md` adds the
  trust boundary + the distinct-identity step as the real-enforcement
  prerequisite.
- The plan's Scope now carries the required `### Review Contract` block.

## Verification

- Ran pytest over `tests/test_set_claude_review_status.py` -> 21 passed (0.20s),
  now including abbreviated-/wrong-length-SHA rejection and the missing-`gh`
  exit-2 path.
- Dry-run of the setter emits the expected gh api POST to
  repos/<repo>/statuses/<sha> with context=claude-review and no network call.
- Confirmed `tests/test_set_claude_review_status.py` now appears in both
  pre_push_audit.yml pytest invocations.
- Ran `scripts/audit_pr_body.py` against this PR body before push -> PASS.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/pre_push_audit.yml` | 4 |
| `AGENTS.md` | 30 |
| `docs/REVIEWER_MERGE_GATE.md` | 84 |
| `plans/PR-Reviewer-Merge-Gate-Claude-Review-Status.md` | 153 |
| `scripts/set_claude_review_status.py` | 150 |
| `tests/test_set_claude_review_status.py` | 175 |
| **Total** | **596** |
