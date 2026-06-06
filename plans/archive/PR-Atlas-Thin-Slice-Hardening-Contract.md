# PR-Atlas-Thin-Slice-Hardening-Contract

## Why this slice exists

Atlas builder sessions have been moving quickly across overlapping Content Ops
areas. The operator wants a stricter default: build the thinnest real
end-to-end slice, fix only what the slice cannot function without, and park
non-blocking discoveries instead of expanding the PR.

This slice adds that workflow contract and a root hardening tracker without
changing product code.

## Scope (this PR)

Ownership lane: atlas-workflow

1. Add the thin-slice and inline-fix rules to `AGENTS.md`.
2. Add the required `HARDENING.md` parking format.
3. Add PR-body and final-report expectations for parked hardening.
4. Preserve the existing seven-section plan-doc shape.
5. Clarify reviewer feedback on security/authz test coverage and hardening
   queue grooming.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Atlas-Thin-Slice-Hardening-Contract.md` | Plan doc for this workflow-contract slice. |
| `AGENTS.md` | Documents thin-slice behavior, hardening triage, exceptions, and reporting expectations. |
| `HARDENING.md` | Root tracker for non-blocking hardening items parked during slices. |

## Mechanism

The builder workflow will explicitly separate required slice work from
non-blocking hardening. Required work is anything needed for the slice's real
flow, AGENTS contract, tests, CI, security, or data truthfulness. Other
discoveries get appended to `HARDENING.md` with location, description, why it
matters, rough effort, category, and source slice.

The plan shape stays unchanged. Parked hardening is reported inside the existing
`Deferred` section and mirrored in the PR body/final report.

## Intentional

- No automation in this first slice. Reviewer enforcement and the written
  contract are enough to test the workflow before adding audit scripts.
- Root `HARDENING.md` instead of package-local files so cross-session discoveries
  are visible in one place.
- Existing plan docs are not retroactively updated.

## Deferred

- A future audit can enforce a `Deferred`/`HARDENING.md` link if builders forget
  to report parked items.
- A future debt-register integration can copy closed hardening items into
  `docs/technical-debt/` if that becomes useful.
- Parked hardening for this slice: none.

## Verification

- Git whitespace check - passed.
- Manual section sweep for `AGENTS.md`, `HARDENING.md`, and this plan - passed.
- Local PR review initially caught a missing Ownership lane; plan updated.
- Local PR review against `origin/main` - passed.
- Reviewer update: added security/authz guard test coverage to the non-parkable
  list and documented the `HARDENING.md` pickup loop.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~75 |
| AGENTS workflow contract | ~55 |
| HARDENING tracker | ~20 |
| **Total** | ~150 |
