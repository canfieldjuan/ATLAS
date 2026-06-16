# PR-Deflection-Full-Report-QA-Redaction-Policy

## Why this slice exists

#1612 starts the full-report delivery QA epic, but its initial proof-bundle
shape named live `report_model.json`, `email.eml`, `result_page.html`,
`report.pdf`, and `evidence_export.csv/jsonl` fixtures. That repeats the #1572
risk: a real paid-report `request_id`/result URL is a live capability, and the
evidence export is the uncapped customer ticket surface.

Root cause: the QA harness planned committed artifacts before defining the
artifact safety boundary. This PR fixes that root for the epic with the
redaction policy and reusable self-check. Later harness slices must commit only
a sanitized scorecard for live runs unless the data and IDs are synthetic.

The amended PR exceeds the 400 LOC soft cap because review found leak-gate
false negatives in the checker itself: snippets could echo raw evidence, paths
were not scanned/redacted, PDFs and missing bundles could pass as clean, and
the detector test was not enrolled in CI. Those are part of the same safety
boundary, so splitting them would leave the gate knowingly unsafe.

## Scope (this PR)

Ownership lane: content-ops/deflection-full-report-qa
Slice phase: Production hardening

1. Update #1612 with the slice-0 artifact policy.
2. Add a proof-bundle redaction checker for future QA artifacts.
3. Emit an assertions-JSON-shaped result with explicit fields for request
   IDs, result URLs, customer emails, local paths, Stripe IDs, raw evidence,
   source-ID lists, and private notes.
4. Fail closed on any forbidden committed-bundle content.
5. Reject missing, empty, PDF, binary, and non-UTF-8 bundles rather than
   treating unscannable artifacts as clean.
6. Scan and redact artifact path names as data.
7. Test every detector plus safe synthetic/example near-misses and enroll the
   checker test in CI.

### Files touched

- `.github/workflows/pre_push_audit.yml`
- `plans/PR-Deflection-Full-Report-QA-Redaction-Policy.md`
- `scripts/check_deflection_full_report_proof_bundle.py`
- `tests/test_check_deflection_full_report_proof_bundle.py`
- `tests/test_pre_push_audit_workflow.py`

### Review Contract

Acceptance criteria:

- #1612 records the rule: live proof runs commit only sanitized scorecards, not
  real email/page/PDF/export bundles.
- The checker scans proof files and reports named rows for every forbidden
  class, including live capabilities, PII-ish fields, Stripe IDs, raw evidence,
  source-ID lists, and private notes.
- Missing, empty, PDF, binary, non-UTF-8, and identifier-named bundles fail
  closed.
- Serialized findings do not echo raw evidence/private-note bodies or raw
  request IDs from path metadata.
- Every detector has a failing negative fixture, and safe synthetic/example
  identifiers remain allowed.
- This PR introduces no live identifiers, customer data, private notes, or raw
  evidence fixtures.

Affected surfaces: future #1612 proof-bundle validation and issue-tracker
contract.

Risk areas: broad regexes can false-positive; narrow regexes can repeat #1572;
generic `secrets_committed` checks are not specific enough.

- Reviewer rules triggered: R1, R2, R3, R9, R10, R12, R13, R14.

## Mechanism

The checker walks a proof-bundle file or directory, scans artifact names and
UTF-8 text content with named detectors, and rejects missing, empty, PDF,
binary, and non-UTF-8 inputs as unreadable. Each detector returns redacted path,
label, and snippet metadata. The CLI prints JSON and exits non-zero on any
violation.

The result object is shaped like future assertions JSON: every sensitive class
is present whether it passes or fails, so omission cannot masquerade as safety.

## Intentional

- This PR blocks the unsafe artifact path before building the full
  email/page/PDF/export harness.
- Tests use tiny synthetic snippets with example domains and fake IDs.
- The checker is fail-closed and conservative for committed bundles. If a
  future live run needs richer local artifacts, they stay uncommitted and only
  the sanitized scorecard is committed.

## Deferred

- PR-Deflection-Full-Report-QA-Scorecard: shared scorecard and model-anchored
  count assertions.
- PR-Deflection-Full-Report-QA-Deterministic-Harness: fake-transport
  email/page/PDF/export consistency tests.
- PR-Deflection-Full-Report-QA-Live-Runner: live Zendesk-shaped proof runner
  that commits only sanitized scorecards.

Parked hardening: none.

## Verification

- python -m pytest tests/test_check_deflection_full_report_proof_bundle.py -q
  (30 passed)
- python -m pytest tests/test_pre_push_audit_workflow.py -q
  (4 passed)
- python -m compileall -q scripts/check_deflection_full_report_proof_bundle.py tests/test_check_deflection_full_report_proof_bundle.py tests/test_pre_push_audit_workflow.py
  (passed)
- python scripts/sync_pr_plan.py plans/PR-Deflection-Full-Report-QA-Redaction-Policy.md --check
  (passed)
- Pending before push: push-wrapper local review.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/pre_push_audit.yml` | 2 |
| `plans/PR-Deflection-Full-Report-QA-Redaction-Policy.md` | 125 |
| `scripts/check_deflection_full_report_proof_bundle.py` | 288 |
| `tests/test_check_deflection_full_report_proof_bundle.py` | 242 |
| `tests/test_pre_push_audit_workflow.py` | 6 |
| **Total** | **663** |
