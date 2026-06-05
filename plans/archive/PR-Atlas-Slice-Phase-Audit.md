# PR-Atlas-Slice-Phase-Audit

## Why this slice exists

PR-Atlas-Workflow-Phase-Contract made `Slice phase` part of the Atlas
PR contract, but enforcement is still manual. That leaves builders and
reviewers dependent on memory exactly where the contract is supposed to
reduce cross-session drift.

This slice closes the deferred follow-up by teaching the existing
cross-session PR audit to validate phase metadata for new plan docs and
the current PR body.

The diff is over the 400 LOC soft cap because the audit change needs
fixture coverage for plan parsing, current-PR body parsing, mismatch
handling, other-PR advisory behavior, real-form normalization, existing
ownership-lane behavior, and the code-block false-positive that surfaced
in #904.

## Scope (this PR)

Ownership lane: atlas-workflow

Slice phase: Workflow/process.

1. Validate that newly added PR plan files contain a recognized
   `Slice phase` in `Scope (this PR)`.
2. Validate the current PR body, when one exists, contains a recognized
   `Slice phase` matching the branch plan.
3. Keep other open PR body phase issues as warnings so legacy/open PRs do
   not block unrelated branches.
4. Add fixture tests for missing, invalid, mismatched, and valid phase
   metadata, plus the existing whole-doc ownership-lane behavior.

### Files touched

- `scripts/audit_pr_session_drift.py`
- `tests/test_audit_pr_session_drift.py`
- `plans/PR-Atlas-Slice-Phase-Audit.md`

## Mechanism

The existing drift audit already:

- collects new plan docs,
- extracts ownership metadata,
- reads open PR bodies through `gh`, and
- runs inside `scripts/local_pr_review.sh`.

This slice extends that flow with `Slice phase` parsing. Plan-doc
metadata is extracted only from the Scope section so illustrative examples
elsewhere in the plan cannot trigger false failures. Current PR body
metadata is blocking because it is the branch under review; other open PR
body phase problems are warnings to avoid blocking a new branch on older
PRs that predate this contract.

## Intentional

- No new audit script. Reusing the existing drift audit avoids another
  local-review step and keeps PR metadata checks together.
- No hard failure for other open PR bodies that are missing phase metadata.
  That would make a builder responsible for legacy PRs outside the slice.
- Existing ownership-lane parsing stays whole-document in this slice. Moving
  the lane parser to Scope-only would be a separate migration, not part of
  phase enforcement.
- No structured front-matter migration in this slice. The current contract
  is line-based; the richer metadata block remains a future option.

## Deferred

- Future PR: consider replacing first-line Scope metadata with a structured
  metadata block if more labels are added.
- Parked hardening: none.

## Verification

- `python -m pytest tests/test_audit_pr_session_drift.py -q` - 22 passed.
- `bash scripts/local_pr_review.sh --allow-dirty` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Audit script | ~130 |
| Tests | ~260 |
| Plan doc | ~84 |
| **Total** | ~474 |
