# PR-FAQ-Deflection-Submit-Deploy-Parity

## Why this slice exists

The source tree now has the multipart FAQ deflection submit route and the
hosted smoke for the portfolio handoff, but the redeployed ATLAS host still
returned the old FastAPI/Pydantic JSON-body validation error for a multipart
CSV submit on May 30, 2026:

`detail[0].type == "model_attributes_type"` at `loc == ["body"]`.

That proves auth and reachability are not the blocker; the deployed runtime is
serving a stale JSON-only submit contract or importing stale route code. The
next slice needs a fail-closed deploy-parity diagnostic so the operator can tell
whether the live host is actually running the multipart contract before moving
on to paid unlock validation.

## Scope (this PR)

Ownership lane: content-ops/faq-deflection
Slice phase: Functional validation

1. Teach the hosted submit smoke to recognize the stale JSON-body FastAPI 422
   that occurs when a multipart CSV hits the old submit route.
2. Return a clear, non-secret diagnostic in the result artifact while keeping
   snapshot/artifact probes skipped after the failed submit.
3. Add focused failure fixtures so the detector catches the stale-route shape
   and does not fire for JSON-mode or unrelated 422 responses.
4. Document the diagnostic and remediation in the submit handoff runbook.

### Files touched

- `plans/PR-FAQ-Deflection-Submit-Deploy-Parity.md`
- `docs/extraction/validation/content_ops_faq_deflection_submit_handoff_runbook.md`
- `scripts/smoke_content_ops_deflection_submit_handoff.py`
- `tests/test_smoke_content_ops_deflection_submit_handoff.py`

## Mechanism

The smoke already parses HTTP error payloads. This slice adds a small predicate
over the submit response:

```python
if submit_mode == "multipart" and status == 422 and detail includes
model_attributes_type at loc ["body"]:
    emit "deployed submit route rejected multipart as JSON body ..."
```

The diagnostic is appended to the existing submit errors. The smoke still
requires submit `200` before it hydrates snapshot or artifact routes, so a stale
deploy cannot create a false-green request id or paid-gate proof.

## Intentional

- This slice does not change the submit route implementation. The current
  source already has the multipart `Request` route, Docker image copy entries,
  and `python-multipart` dependency; the observed failure is deploy/runtime
  parity.
- The detector is restricted to multipart mode and the exact FastAPI
  `model_attributes_type` body error so ordinary validation errors are not
  mislabeled as stale deploys.
- The smoke does not print raw response bodies or uploaded CSV contents in the
  diagnostic.

## Deferred

- Parked hardening: none. Existing `HARDENING.md` entries do not touch this
  smoke or submit handoff lane.
- Live green proof remains deferred until the deployed host serves the
  multipart route and the operator provides a representative CSV input.
- Stripe paid-unlock live proof remains the next validation step after submit
  returns a real `request_id`.

## Verification

- `python -m py_compile scripts/smoke_content_ops_deflection_submit_handoff.py tests/test_smoke_content_ops_deflection_submit_handoff.py` - passed.
- `python -m pytest tests/test_smoke_content_ops_deflection_submit_handoff.py -q` - 20 passed.
- Live diagnostic run against `https://atlas-brain.tailc7bd29.ts.net` with the
  B2B JWT/account env from the local root `.env` and a temporary CSV fixture -
  expected fail, exit 1: submit `422` with
  `deployed submit route rejected multipart as a JSON body`; snapshot/artifact
  skipped with `submit_failed`. This confirms the current deployed host is
  reachable/authenticated but still serving the stale JSON-body submit route.
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/faq-deflection-submit-deploy-parity-pr-body.md` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 83 |
| Smoke script | 35 |
| Tests | 75 |
| Runbook | 20 |
| **Total** | **213** |
