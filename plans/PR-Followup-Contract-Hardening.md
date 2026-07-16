# PR-Followup-Contract-Hardening

## Why this slice exists

The draft-only customer-follow-up contract (`atlas_brain/schemas/followup_workflow.py`,
merged in #2126) left three machine-progress refinements deferred to #2129 under the
3-round Codex cap. They are latent on an unwired contract, but they close the last
gaps between what the worker can CLAIM in the machine-progress fields and what those
fields canonically mean, so downstream consumers of the contract never receive a
non-canonical or approval-adjacent claim. This slice clears #2129: it makes
`stages_completed` on a failure the exact ordered pre-failure prefix (fail-closed,
like `error_code`), and forbids a drafted result from claiming the server-owned
approval stage completed. The third item is a plan-doc reviewer-rules declaration
(R3), addressed by this slice's Review Contract below.

### Problem-derived contract

From the #2129 findings, a correct fix must:
- Reject a drafted (successful) result whose `stages_completed` includes `approval`:
  the approval stage is server-owned and runs after the worker's draft, so the worker
  may not claim it completed (even though `resolve_approval` already ignores the
  claim, so nothing is granted).
- Require `stages_completed` on a failure to be the exact ordered prefix of the stages
  before `failed_stage` -- a missing, reordered, duplicated, or at/after entry must all
  fail closed, not merely entries at or after the failed stage.
- Declare R3 on the Review Contract: the server-owned approval boundary is an
  authorization surface.

## Scope (this PR)

Ownership lane: followup-workflow
Slice phase: production hardening

Two guard closures in the existing validator plus their proving tests. No new module,
no runtime caller, no send path. The contract is still unwired; this only tightens
its fail-closed machine-progress rules.

### Review Contract

- Acceptance criteria:
  - [ ] A drafted result whose `stages_completed` includes `approval` is rejected; a
        drafted result reporting only pre-approval stages still validates.
  - [ ] A failure validates only when `stages_completed` equals the exact ordered
        prefix before `failed_stage`; missing / reordered / duplicated / at-or-after
        entries are all rejected.
  - [ ] All prior #2126 invariants still hold (no regression in the 56 existing tests).
- Reachability proof: N/A -- contract + validator only, no runtime caller and no send
  path. Proof is the test suite.
- Affected surfaces: one existing schema module and its test file; no new file; nothing
  imports the module yet.
- Risk areas: the server-owned approval boundary (a drafted result may not claim the
  approval stage; approval remains derived only via `resolve_approval`); machine-progress
  canonicality of `stages_completed` on failures.
- Reviewer rules triggered: R2, R3, R14. R3 is declared because the change tightens the
  server-owned approval authorization boundary.

### Files touched

- `atlas_brain/schemas/followup_workflow.py`
- `plans/PR-Followup-Contract-Hardening.md`
- `tests/test_followup_workflow.py`

## Mechanism

Both closures live in the existing `_fail_closed` model-validator. In the drafted
branch, `"approval" in stages_completed` now raises (the stage is server-owned). In the
failure branch, the previous "reject stages at/after `failed_stage`" loop is replaced
with an exact-equality check against `list(STAGE_ORDER[:index(failed_stage)])`, so
`stages_completed` must be precisely the ordered pre-failure prefix. Both are consistent
with the contract's existing fail-closed treatment of the canonical `error_code`.

## Intentional

- Failure `stages_completed` is enforced (fail closed) rather than normalized/derived:
  a worker that cannot report the canonical pre-failure prefix is malfunctioning, so the
  whole result is rejected -- the same posture as the exact canonical `error_code` check,
  not a silent overwrite of a contradictory claim.
- Drafted `stages_completed` is only blocked from claiming `approval`; a drafted result
  may still report a subset of the pre-approval stages (`lookup`/`select`/`compose`).
  The finding was specifically the approval-stage claim; drafted pre-approval stages are
  informational and not an authorization surface.

## Deferred

- None. This slice closes all three #2129 items (the R3 item is a Review Contract
  declaration, above).

## Verification

```
python -m pytest tests/test_followup_workflow.py -q
```
60 tests pass (56 existing + 4 new): a drafted result claiming the approval stage is
rejected while a drafted result reporting pre-approval stages validates; a failure's
`stages_completed` must equal the exact ordered pre-failure prefix (missing / reordered /
duplicated / at-or-after all rejected; empty prefix required for a first-stage failure);
all prior invariants unchanged.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/schemas/followup_workflow.py` | 29 |
| `plans/PR-Followup-Contract-Hardening.md` | 112 |
| `tests/test_followup_workflow.py` | 47 |
| **Total** | **188** |
