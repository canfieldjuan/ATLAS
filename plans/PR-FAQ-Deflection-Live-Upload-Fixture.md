# PR-FAQ-Deflection-Live-Upload-Fixture

## Why this slice exists

The deflection intake path is now the portfolio upload page: customer CSV ->
private Vercel Blob -> portfolio JSON submit -> ATLAS multipart submit -> free
snapshot and locked paid artifact. The code path is wired, but operator live
validation still depends on an ad hoc CSV assembled outside the repo.

This slice adds a small checked support-ticket CSV fixture and updates the
handoff runbook so the live upload/snapshot test uses a stable input that
matches the current private-Blob and account-less result-page contracts.

## Scope (this PR)

Ownership lane: content-ops/deflection-report-gating
Slice phase: Functional validation

1. Add a representative support-ticket CSV fixture for live FAQ deflection
   upload and snapshot generation.
2. Update the hosted handoff runbook to point operators at the checked fixture
   and the current account-less result-page expectations.
3. Add a focused fixture test that guards the CSV header, row count, and
   repeated FAQ themes needed for a meaningful snapshot.

### Files touched

- `plans/PR-FAQ-Deflection-Live-Upload-Fixture.md`
- `docs/extraction/validation/content_ops_faq_deflection_submit_handoff_runbook.md`
- `docs/extraction/validation/fixtures/faq_deflection_live_upload_sample.csv`
- `tests/test_faq_deflection_live_upload_fixture.py`

## Mechanism

The fixture uses non-sensitive synthetic support-ticket rows with the fields the
upload page and ATLAS submit path accept:

```text
ticket_id,created_at,subject,message,status,tags
```

The focused test parses the fixture with Python's `csv` module and requires at
least 12 rows, the exact header, unique ticket ids, and repeated themes for
exports, billing, security, and team/admin questions. That keeps the live
snapshot input useful without baking generated snapshot output into the repo.

## Intentional

- This does not run the hosted live upload; credentials, production URLs, and
  Vercel Blob state stay operator-owned.
- This does not add another network smoke. It makes the existing operator
  smokes reproducible with a checked fixture.
- The fixture is synthetic and contains no customer data.

## Deferred

- Recording a new hosted live upload/snapshot artifact remains an operator run
  after this fixture PR lands.
- Stripe paid-unlock live validation remains gated behind a successful unpaid
  submit/result-page smoke.
- Parked hardening: none.

## Verification

- Python compile check for `tests/test_faq_deflection_live_upload_fixture.py` - passed.
- Focused pytest for `tests/test_faq_deflection_live_upload_fixture.py` - 2 passed.
- Local PR review with the prepared PR body file - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 85 |
| Fixture CSV | 13 |
| Runbook updates | 25 |
| Fixture test | 50 |
| **Total** | **173** |

Under the 400 LOC soft cap.
