# PR-Deflection-Live-Proof-Report-Golden

## Why this slice exists

Issue #1434 found that the committed resolution-evidence live-proof
`report.md` fixture drifted after #1429 made paid deflection report prose
brand-neutral. The generator now emits "This report found..." and "These were
mined...", while the committed sample still says "ATLAS found..." and "ATLAS
mined...".

The current proof test only checks summary equality plus a report substring, so
that buyer-facing fixture drift stayed green. This slice refreshes the fixture
against the current deterministic generator and strengthens the test so future
report prose changes cannot silently leave the committed proof stale.

## Scope (this PR)

Ownership lane: deflection/clustering-raw-data
Slice phase: Production hardening

1. Regenerate the committed resolution-evidence proof artifacts from the
   existing `source.csv` on current `origin/main`.
2. Commit the refreshed `report.md` fixture, and include `summary.json` /
   `result.json` only if the generator changes them.
3. Strengthen the proof test to assert full `report.md` byte-for-byte equality
   between regenerated output and the committed fixture.
4. Do not change deflection report generation semantics, source rows, PDF,
   email delivery, checkout, or live payment behavior.

### Review Contract

- Acceptance criteria:
  - [ ] Regenerating from the committed source CSV matches the committed
        `report.md` fixture exactly.
  - [ ] The test would fail on the #1434 stale-branding drift.
  - [ ] Existing publishable-answer and no-proven-answer lane assertions stay
        intact.
  - [ ] The refreshed fixture stays deterministic and no LLM/local-model route
        is introduced.
- Affected surfaces: one proof fixture and its focused regression test.
- Risk areas: accidentally changing source evidence, weakening the lane proof,
  or drifting into delivery/payment work.
- Reviewer rules triggered: R1, R2, R10, R13.

### Files touched

- `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/report.md`
- `plans/PR-Deflection-Live-Proof-Report-Golden.md`
- `tests/test_content_ops_deflection_resolution_live_proof.py`

## Mechanism

The existing proof test already invokes `scripts/build_content_ops_deflection_report.py`
against the committed source CSV and temporary output files. This PR adds the
missing golden assertion:

```python
assert output.read_text(encoding="utf-8") == REPORT.read_text(encoding="utf-8")
```

The fixture is regenerated with the same CLI used by the original proof. Since
the source rows and deterministic generator are unchanged, the expected
behavioral metrics stay the same while the report prose sample catches up to
current shipped output.

## Intentional

- This does not broaden #1419's proof fixture or source shape. The issue is
  stale output locking, not missing data coverage.
- This does not add a looser substring/contains matcher. The point is to make
  any future report Markdown change explicit through a fixture refresh.
- This does not touch live upload, Stripe, PDF, or email paths.

## Deferred

- Sanitized real-provider export fixtures remain deferred under #1384.
- Operator-supplied production upload/payment/delivery proof remains separate
  from this deterministic fixture lock.

Parked hardening: none.

## Verification

- Regenerated the proof fixture with
  `scripts/build_content_ops_deflection_report.py`.
  - Result: passed; summary still reports `drafted_answer_count=2`,
    `no_proven_answer_count=2`, and
    `support_ticket_resolution_evidence_present=true`.
- Focused proof pytest for
  `tests/test_content_ops_deflection_resolution_live_proof.py`.
  - Result: `3 passed in 0.08s`.
- Full extracted pipeline CI mirror through
  `scripts/run_extracted_pipeline_checks.sh`.
  - Result: `3571 passed, 10 skipped, 1 warning in 60.16s`.
- Local PR review with `scripts/local_pr_review.sh`.
  - Result: passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/report.md` | 4 |
| `plans/PR-Deflection-Live-Proof-Report-Golden.md` | 105 |
| `tests/test_content_ops_deflection_resolution_live_proof.py` | 7 |
| **Total** | **116** |
