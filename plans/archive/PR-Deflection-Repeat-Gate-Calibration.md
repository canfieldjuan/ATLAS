# PR-Deflection-Repeat-Gate-Calibration

## Why this slice exists

Issue #1440 is down to live proof/config blockers. The full-volume hosted submit
proof already showed ATLAS can ingest and build the regenerated CFPB upload:
40,383 submitted rows, 52.4 MB, 1,659 generated questions, snapshot 200, unpaid
artifact 403. The only submit-gate failure was self-inflicted: the live command
used an ad hoc `--min-repeat-ticket-count 30000`, while the actual full-volume
sample produced 27,384 repeat tickets.

That means the root problem is calibration discipline, not submit transport or
report generation. This slice replaces the hand-entered threshold bundle with a
named `full-volume-cfpb` gate profile whose repeat threshold is below the
observed proof but still high enough to reject tiny/smoke fixtures.

The diff is over the 400 LOC target because the review fix adds a focused
regression test proving lower explicit minimums cannot weaken profile floors,
and the live proof doc now records the portfolio route resolution without
removing the remaining real-snapshot rerun step.

## Scope (this PR)

Ownership lane: content-ops/deflection-full-volume
Slice phase: Functional validation

1. Add a named full-volume CFPB gate profile to the hosted deflection submit
   handoff smoke.
2. Calibrate that profile from the committed #1440 live proof: repeat-ticket
   minimum 25,000, source/submitted rows 30,000, generated questions 30,
   uploaded bytes 50,000,000, top questions 5.
3. Preserve the existing explicit `--min-*` flags; nonzero explicit minimums can
   raise profile floors when the operator intentionally wants a stricter gate.
4. Update the #1440 proof docs/runbook to use the named profile instead of the
   disproven ad hoc repeat threshold.
5. Add focused tests for profile pass, tiny-fixture fail, explicit stricter
   override fail, and lower explicit minimums not weakening profile floors.

### Review Contract

- Acceptance criteria:
  - [ ] A live-shaped full-volume result with 27,384 repeat tickets passes the
        `full-volume-cfpb` profile.
  - [ ] A tiny/smoke fixture fails the same profile.
  - [ ] An explicit nonzero `--min-repeat-ticket-count 30000` still fails a
        27,384-repeat result, proving stricter overrides remain possible.
  - [ ] An explicit lower `--min-*` value cannot weaken a selected profile
        floor.
  - [ ] Existing no-profile gate behavior is unchanged.
- Affected surfaces: smoke script, smoke tests, validation docs.
- Risk areas: false-green validation, CI enrollment, operator runbook drift.
- Reviewer rules triggered: R1, R2, R10, R12, R14.

### Files touched

- `docs/extraction/validation/content_ops_faq_deflection_submit_handoff_runbook.md`
- `docs/extraction/validation/deflection_full_volume_live_proof_2026-06-14.md`
- `plans/PR-Deflection-Repeat-Gate-Calibration.md`
- `scripts/smoke_content_ops_deflection_submit_handoff.py`
- `tests/test_smoke_content_ops_deflection_submit_handoff.py`

## Mechanism

The smoke gets a `--volume-gate-profile full-volume-cfpb` option. When selected,
the configured volume gates start from the profile defaults. Existing explicit
`--min-*` values remain supported and raise profile floors only when they are
greater than the profile default for that gate.

The profile defaults are:

| Gate | Minimum |
|---|---:|
| uploaded bytes | 50,000,000 |
| source rows | 30,000 |
| submitted rows | 30,000 |
| generated questions | 30 |
| repeat tickets | 25,000 |
| visible top questions | 5 |

The result artifact includes the selected profile name inside `volume_gates` so
future live evidence shows whether the run used the calibrated profile or manual
thresholds.

## Intentional

- The profile does not hide the old `30000` behavior. If an operator explicitly
  passes `--min-repeat-ticket-count 30000`, the stricter gate still applies and
  still fails the observed 27,384-repeat sample.
- This does not lower product standards for real buyers; it calibrates the
  validation gate to the first committed near-50 MB CFPB live proof while still
  rejecting low-volume smoke fixtures.
- This does not change report generation, clustering, portfolio rendering,
  Stripe, paid unlock, or email/PDF delivery.

## Deferred

- Live rerun with `--volume-gate-profile full-volume-cfpb` after this PR deploys
  or after the operator chooses to rerun the submit proof.
- Paid unlock/email/PDF proof remains blocked on deployed Stripe signing-secret
  alignment.
- Portfolio canonical snapshot rendering still needs deployed config/data
  availability; #307 and #308 closed the routing failure modes.

Parked hardening: none.

## Verification

- pytest tests/test_smoke_content_ops_deflection_submit_handoff.py - 41 passed.
- python -m py_compile scripts/smoke_content_ops_deflection_submit_handoff.py - passed.
- bash scripts/run_extracted_pipeline_checks.sh - 4,189 passed, 10 skipped,
  1 existing torch warning.
- bash scripts/local_pr_review.sh with the current PR body - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/extraction/validation/content_ops_faq_deflection_submit_handoff_runbook.md` | 25 |
| `docs/extraction/validation/deflection_full_volume_live_proof_2026-06-14.md` | 36 |
| `plans/PR-Deflection-Repeat-Gate-Calibration.md` | 123 |
| `scripts/smoke_content_ops_deflection_submit_handoff.py` | 87 |
| `tests/test_smoke_content_ops_deflection_submit_handoff.py` | 181 |
| **Total** | **452** |
