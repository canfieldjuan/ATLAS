# PR-Content-Factory-Gate-Wiring

## Why this slice exists

#2135 built the deterministic copy-verification gate (`verify_copy`) but nothing called
it, so the #2116 promote guard still ran on the worker's OWN `copy_verification` claim --
a model could self-report `verdict: "pass"` and promote overclaiming copy. This slice
(part of #2136, Phase 4.2) wires the gate into the stage runner: when the editor stage
produces an `editorial_audit.v1`, the runner recomputes `copy_verification` deterministically
from the edited copy and overwrites the worker's value before validation. Now the promote
decision is gated on the gate's verdict, not the worker's, so a model cannot self-promote.

### Problem-derived contract

A correct fix must:
- Recompute `copy_verification` deterministically (via `verify_copy` on the edited copy)
  for an editor audit and discard any worker-supplied value.
- Do this before the artifact is validated/persisted, so #2116's promote-requires-pass
  guard sees the deterministic verdict.
- Fail closed: a worker that asserts `promote` on overclaiming/PII copy is rejected (not
  persisted); a `revise` recommendation still persists with the recorded hits.
- Touch only the editor stage; other stages are unaffected.

## Scope (this PR)

Ownership lane: content-factory
Slice phase: vertical slice

One helper in `content_factory_runner` called from `run_stage`, plus tests. No new module,
no change to the contracts or the gate. Live end-to-end validation against the model is a
separate step (needs LM Studio serving the qualified model).

### Review Contract

- Acceptance criteria:
  - [ ] An editor audit's `copy_verification` is the deterministic `verify_copy` result of
        its `edited_body_markdown`, regardless of what the worker reported.
  - [ ] A worker asserting `promote` (and a passing verdict) on overclaiming copy is
        rejected and nothing is persisted; a `revise` recommendation persists with the
        fail verdict + hits.
  - [ ] Clean edited copy may promote.
  - [ ] Non-editor stages get no `copy_verification` injected.
- Reachability proof: `run_stage` is the pipeline's stage entry point; the store persists
  what it returns. Proof is the test suite (worker mocked, real store + contracts + gate).
- Affected surfaces: `content_factory_runner.run_stage` (one added helper call) and its
  test file.
- Risk areas: the deterministic override actually replacing the worker's claim; fail-closed
  on self-promote; only the editor stage touched.
- Reviewer rules triggered: R14 (wires a safety gate into the enforcement path).

### Files touched

- `atlas_brain/services/content_factory_runner.py`
- `plans/PR-Content-Factory-Gate-Wiring.md`
- `tests/test_content_factory_runner.py`

## Mechanism

`_enforce_copy_verification(artifact)` sets `artifact["copy_verification"]` to
`verify_copy(edited_body_markdown).model_dump()` when the artifact's schema is
`editorial_audit.v1`, discarding any worker value. `run_stage` calls it after extracting
the JSON and before `write_artifact`. Because #2116's `EditorialAudit` rejects
`recommendation == "promote"` unless the verdict is `"pass"`, a self-promoting overclaim
now fails validation in the store and is never persisted.

## Intentional

- The override is unconditional for the editor stage: the worker's `copy_verification` is
  advisory at best and never trusted, matching the "model cannot self-promote" invariant.
- Enforcement reuses the existing #2116 guard rather than adding new gating: injecting the
  real verdict is enough; the contract already rejects promote-without-pass.
- The gate verifies `edited_body_markdown` (the copy the editor would promote); if the
  editor leaves it empty, there is nothing to verify (a degenerate case a later slice can
  tighten by also checking the parent draft).

## Deferred

- Live end-to-end validation through the model (needs LM Studio serving the qualified
  model) -- an ops step, tracked in #2136.
- An OWUI Editor Filter enforcing the same gate at the OWUI boundary, and verifying the
  parent draft body when the audit's edited copy is empty -- later, #2136.

## Verification

```
python -m pytest tests/test_content_factory_runner.py -q
```
21 tests pass (16 existing + 5 new): an editor audit's copy_verification is the
deterministic verdict of its edited copy; a worker cannot self-promote overclaiming copy
(rejected, nothing persisted) while a revise recommendation persists with the fail verdict
and hits; clean copy may promote; a worker-claimed verdict is overridden; non-editor stages
get no copy_verification.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/services/content_factory_runner.py` | 29 |
| `plans/PR-Content-Factory-Gate-Wiring.md` | 104 |
| `tests/test_content_factory_runner.py` | 61 |
| **Total** | **194** |
