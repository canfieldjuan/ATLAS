# PR-Open-PR-Final-Admission-Gate

## Why this slice exists

PR #2251 turned one pre-open review gap into a proof-file/push-parser handshake.
This replacement keeps the root fix in `open_pr.sh`: review before
`gh pr create/edit`, then mutate only the published branch that was reviewed.
This workflow/process slice is admitted now because it fixes a concrete
merge-safety blocker: without the final admission gate, Atlas PRs can enter
GitHub review and required checks with a head, body, or target that the local
mechanical gate did not review.
The slice is over the soft LOC budget because the reviewed snapshot, target
binding, real-review tests, and plan/body contract must land together; splitting
them would leave the mutating helper half-guarded.

### Problem-derived contract

- Root cause: `open_pr.sh` could mutate GitHub after only a body audit.
- Correct fix must touch/change: `open_pr.sh` audits the body, rejects target
  selectors, derives repo identity from normal GitHub `origin` URL forms,
  verifies and pins head/body/PR identity before/after review, enforces a
  single local writer around the reviewed mutation window, then calls
  `gh pr create/edit` with explicit trusted `--repo` and `--base main`. Tests
  cover each gate.
- Must not change: `push_pr.sh`, pre-push hook dispatch, CI, product code, #2251
  branch content, or other PR lanes.

## Scope (this PR)

Ownership lane: workflow/open-pr-final-admission
Slice phase: Workflow/process

1. Add the final admission checks to `scripts/open_pr.sh`.
2. Add focused wrapper tests.

### Review Contract

- Acceptance criteria:
  1. Missing, stale, or post-review-changed `origin/<branch>`/`HEAD` blocks fake `gh`.
  2. Post-review body or full PR-identity drift blocks fake `gh`.
  3. Fake and real `local_pr_review.sh` failures block create and edit mutation.
  4. `--head`, `--repo`, non-main `--base`, and `GH_REPO` are rejected.
  5. Create/edit still pass body over stdin and never pass the body path to `gh`.
  6. Existing PR edits require matching `headRefName`, `headRefOid` equal to
     the reviewed head, same `headRepository` by case-insensitive owner/repo
     identity, `baseRefName=main`, and non-cross-repo identity.
  7. `scripts/push_pr.sh` is unchanged.
  8. Normal GitHub SSH/HTTPS origin forms, GitHub owner/repo casing differences,
     and single-branch fetch refspecs are accepted without opening
     target-selection holes.
- Reachability proof: tests execute real `scripts/open_pr.sh` with fake `gh`;
  two paths run real `scripts/local_pr_review.sh` and assert no mutation.
- Affected surfaces: `scripts/open_pr.sh`, `tests/test_open_pr_wrapper.py`, plan.
- Risk areas: skipped review, stale branch, target leakage, ambiguous PR
  identity, docs-only handling, and #2251 proof-surface regrowth.
- Reviewer rules triggered: R1, R2, R10, R12, R13, R14.

### Boundary-change enumeration

- Boundary path/seam: `scripts/open_pr.sh` mutation admission and PR lookup.
- Replaced-path behaviors: create/edit now require publication, target rejection,
  local review, exact pre/post-mutation snapshot recheck, single-local-writer
  admission, and explicit repo/base.
- Guard-relevant fields: branch, `HEAD`, `origin/<branch>`, `origin/main`, body
  file/hash, create args, `GH_REPO`, origin repo URL, local review exit, PR
  number, `headRefName`, `headRefOid`, `headRepository.nameWithOwner`,
  `baseRefName`, `isCrossRepository`, local mutation lock.
- Closure declaration: CLOSED. The target-selector denylist is exactly `GH_REPO`,
  `--head`/`-H`, `--repo`/`-R`, and non-main `--base`/`-B`; membership comes
  from `gh pr create` target-selection flags used by this wrapper. Inputs outside
  this set are admitted only as non-target create args and still must pass body,
  publication, review, and exact-identity gates.
- Caller x input shape: create, edit, docs-only, stale/changing snapshot,
  target/env override, bad PR identity, single-branch fetch refspec, normal
  GitHub origin URL forms, and review failure.

### Deployed-config probing

- N/A local wrapper guard. Published fixture branch permits fake create/edit;
  missing/stale branch, review failure, changed post-review snapshot, implicit
  target config, or bad PR identity blocks it.

### Files touched

- `plans/PR-Open-PR-Final-Admission-Gate.md`
- `scripts/open_pr.sh`
- `tests/test_open_pr_wrapper.py`

## Mechanism

`open_pr.sh` rejects target overrides, derives the repo from raw `origin`, pins
the reviewed head/body/PR snapshot, takes a local mutation lock, runs
`local_pr_review.sh`, rechecks the same snapshot immediately before mutation,
then mutates GitHub with explicit `--repo` and `--base main`. Because GitHub
does not expose a PR body compare-and-swap operation, the enforced execution
model is single local writer plus post-mutation verification: a competing writer
that changes the branch or PR identity during the mutation window makes the
wrapper fail after the mutation and requires human inspection before continuing.
No proof file or `push_pr.sh` parser.

## Intentional

- Supersedes #2251; no proof-file handshake.
- Cross-repo PRs, non-main bases, and explicit `--head` stay out of this helper.
- The final GitHub mutation is not claimed to be atomic at the API layer; the
  wrapper enforces single local writer and detects post-mutation drift because
  `gh pr create/edit` has no reviewed-SHA compare-and-swap.

## Deferred

- Close #2251 as superseded after this replacement PR is published.
- Ownership-wrapper and draft-consent guards remain separate slices.
- Parking predicate: findings outside `open_pr.sh` final admission semantics,
  wrapper tests, or this plan contract are parked unless they prove the helper
  can still publish an unreviewed head/body/target or block a valid normal
  Atlas push/open flow.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_open_pr_wrapper.py -q` - 25 passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Open-PR-Final-Admission-Gate.md` | 131 |
| `scripts/open_pr.sh` | 285 |
| `tests/test_open_pr_wrapper.py` | 470 |
| **Total** | **886** |
