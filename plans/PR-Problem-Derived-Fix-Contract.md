# PR-Problem-Derived-Fix-Contract

## Why this slice exists

The operator set a new builder discipline: before coding, derive the correct
solution from the problem alone, write that contract down, build only to it,
then reconstruct the finished diff cold before calling the slice done. The
existing reconstruction docs already pushed reviewers in that direction, but
the builder-side scaffold and PR-body gate did not force the method. That made
the rule easy to remember in one session and easy to drop after compaction or a
fresh handoff.

### Problem-derived contract

- Root cause: the builder workflow had reconstruction guidance, but it did not
  make the problem-derived contract and cold diff reconstruction a hard,
  generated, PR-body-enforced step before coding and before push.
- Correct fix must touch/change: update the builder workflow contract, the
  reconstruction guidance, the fresh-session bootstrap, the plan scaffold, and
  the PR-body audit/tests so future non-Dependabot PRs must carry the
  `## Cold diff reconstruction` receipt; wire that audit into the local
  push/open wrappers so invalid bodies fail before push or PR publication.
- Must not change: runtime code, product shape, report/snapshot/email/PDF
  behavior, billing/checkout behavior, reviewer verdict semantics, merge
  authority, watcher behavior, or any active product lane.

## Scope (this PR)

Ownership lane: workflow-process
Slice phase: Workflow/process

1. Codify the problem-derived pre-code contract in the builder workflow docs.
2. Update the generated PR plan scaffold so new plans start with the contract
   block.
3. Require `## Cold diff reconstruction` in non-Dependabot PR bodies.
4. Add focused tests proving the scaffold and PR-body audit enforce the new
   shape.
5. Run the PR-body audit from the push/open wrappers before push or PR
   create/edit, with tests proving invalid bodies stop before external actions.

### Review Contract

- `AGENTS.md` names the pre-code contract and before-push cold reconstruction
  as builder requirements.
- `docs/CODING_FOR_RECONSTRUCTION_REVIEW.md` matches the operator's 1-2-3
  method rather than the previous softer three-way self-check.
- `docs/SESSION_BOOTSTRAP.md` reminds fresh sessions to use the same method.
- `scripts/new_pr_plan.sh` generates a `### Problem-derived contract` block.
- `scripts/audit_pr_body.py` requires `## Cold diff reconstruction` for
  non-Dependabot PR bodies, with tests proving both pass and fail behavior.
- `scripts/push_pr.sh` and `scripts/open_pr.sh` run the PR-body audit before
  push, PR create, or PR edit, with tests proving invalid bodies do not reach
  fetch/local-review/push or `gh`.
- No runtime/product files or user-facing product shape move.
Reviewer rules triggered: R1 requirements match, R2 test evidence, R10
checker/gate predicates, R14 codebase verification.

### Files touched

- `AGENTS.md`
- `docs/CODING_FOR_RECONSTRUCTION_REVIEW.md`
- `docs/SESSION_BOOTSTRAP.md`
- `plans/PR-Problem-Derived-Fix-Contract.md`
- `scripts/audit_pr_body.py`
- `scripts/new_pr_plan.sh`
- `scripts/open_pr.sh`
- `scripts/push_pr.sh`
- `tests/test_audit_pr_body.py`
- `tests/test_new_pr_plan.py`
- `tests/test_open_pr_wrapper.py`
- `tests/test_push_pr_wrapper.py`

## Mechanism

- Tighten the builder reconstruction doc so it requires the exact sequence:
  derive contract before code, build only to it, then reconstruct the diff cold
  with cited gaps before done.
- Mirror the rule into AGENTS and the bootstrap prompt so both ongoing and
  fresh sessions see it.
- Add the contract placeholder to `new_pr_plan.sh` so the plan scaffold creates
  the contract area before implementation begins.
- Add `Cold diff reconstruction` to the PR-body audit's required section list
  so local/CI PR-body checks fail if the self-reconstruction receipt is absent.
- Call the same audit in the push/open wrappers immediately after the body file
  exists check, so the documented local path fails before external side effects.

## Intentional

- Do not change `docs/PR_RECONSTRUCTION_PROTOCOL.md`; it already defines the
  reviewer-side independent reconstruction flow. This slice tightens the
  builder side.
- Do not add a deeper semantic parser for the cold reconstruction text. The PR
  body gate enforces section presence; reviewers still judge whether the
  content is honest and cited.

## Deferred

- A future hardening slice may add a richer audit that validates the
  `Problem-derived contract` block content, but this slice stops the current
  omission loop by generating the block and enforcing the PR-body receipt.

Parked hardening: none.

## Verification

- Command: python -m pytest tests/test_new_pr_plan.py tests/test_audit_pr_body.py -q (27 passed)
- Command: python -m pytest tests/test_new_pr_plan.py tests/test_audit_pr_body.py tests/test_push_pr_wrapper.py tests/test_open_pr_wrapper.py -q (44 passed)
- Command: python scripts/audit_pr_body.py /tmp/pr-problem-derived-fix-contract-body.md (passed)
- Command: ATLAS_CURRENT_PR_BODY_FILE=/tmp/pr-problem-derived-fix-contract-body.md bash scripts/local_pr_review.sh (passed)

## Estimated diff size

| File | LOC |
|---|---:|
| `AGENTS.md` | 39 |
| `docs/CODING_FOR_RECONSTRUCTION_REVIEW.md` | 59 |
| `docs/SESSION_BOOTSTRAP.md` | 1 |
| `plans/PR-Problem-Derived-Fix-Contract.md` | 127 |
| `scripts/audit_pr_body.py` | 5 |
| `scripts/new_pr_plan.sh` | 6 |
| `scripts/open_pr.sh` | 2 |
| `scripts/push_pr.sh` | 2 |
| `tests/test_audit_pr_body.py` | 26 |
| `tests/test_new_pr_plan.py` | 4 |
| `tests/test_open_pr_wrapper.py` | 93 |
| `tests/test_push_pr_wrapper.py` | 84 |
| **Total** | **448** |
