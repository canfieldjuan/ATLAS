# PR-Deflection-Live-Artifact-Model-Export-Contract

## Why this slice exists

#1612 is the report delivery proof arc, and #1671 finally ran the live paid
artifact proof against a real hosted request. That proof succeeded through paid
unlock and PDF extraction, then exposed the upstream contract failure that still
blocks the testing arc: the hosted artifact route returned a legacy-shaped
artifact without `evidence_export`, and the hosted report-model route returned
404 for the same fresh paid request.

Root cause: the modern paid artifact contract is only pinned on adjacent
generation and renderer surfaces. The buyer-facing portfolio submit route, the
path used by the live proof, tests locked/paid artifact availability but does
not require persisted paid artifacts to carry `report_model` and
`evidence_export`, nor does it prove that the report-model route works for a
fresh submit-generated request. That let a stale or incomplete hosted contract
escape the repo gates.

This change fixes the root test/contract gap for the submit path first. If the
new regression fails, the implementation fix belongs upstream in the submit
artifact persistence boundary, not in the live QA runner.

## Scope (this PR)

Ownership lane: content-ops/deflection-full-report-qa
Slice phase: Functional validation

1. Pin the portfolio submit paid-flow contract: a fresh submit-generated paid
   artifact must expose a supported `deflection.v1` `report_model` and an
   object `evidence_export`.
2. Pin the paired route contract: the report-model route must return the same
   model persisted inside the paid artifact for the same fresh request.
3. Keep the existing locked teaser/paywall behavior unchanged.

### Review Contract

- Acceptance criteria:
  - [ ] A fresh full-thread submit request persists a paid artifact containing
        object-valued `report_model` and `evidence_export`.
  - [ ] The report-model route returns HTTP-200-equivalent data for that fresh
        paid submit request and matches the artifact's `report_model`.
  - [ ] The locked pre-payment artifact route still returns 403 and does not
        leak the full report.
  - [ ] The existing paid-flow route test through execute remains green.
- Affected surfaces: API, paid report persistence, tests, hosted proof
  contract.
- Risk areas: public API backcompat, authorization boundary, stale hosted
  contract, false-green validation.
- Reviewer rules triggered: R1, R2, R3, R5, R6, R10, R12, R14.

### Files touched

- `plans/PR-Deflection-Live-Artifact-Model-Export-Contract.md`
- `tests/test_extracted_content_deflection_submit.py`

## Mechanism

The slice adds focused assertions to the existing submit-path regression that
already exercises the real full-thread blob importer, the portfolio submit
route, the in-memory paid report store, the paid marker route, and the artifact
route. After the request is marked paid, the test fetches the paid artifact and
the report-model route for the same request id, then asserts the shared model
contract and the presence of the uncapped evidence export.

If those assertions fail, the repair will be made where the submit route stores
the generated artifact. The live runner and portfolio pages are downstream
consumers; they should not compensate for a missing producer contract.

## Intentional

- This slice does not rebuild the live QA harness. #1671 already proved the
  harness can expose the failure; the missing proof is the repo-side submit
  contract.
- This slice does not close the parked handoff-smoke teaser false positive in
  `HARDENING.md`. That is adjacent checker cleanup, while this PR targets the
  paid artifact/report-model contract that blocked the live proof.

## Deferred

- Rerun the live #1612 paid artifact proof after this contract lands and is
  deployed.
- Drain the parked handoff-smoke teaser false positive separately unless it
  blocks the rerun.

Parked hardening: none.

## Verification

- Passed:
  - `python -m pytest tests/test_extracted_content_deflection_submit.py::test_deflection_submit_accepts_zendesk_full_thread_blob -q` - 1 passed.
  - Python compile check for `tests/test_extracted_content_deflection_submit.py` - passed.
  - `python -m pytest tests/test_extracted_content_deflection_submit.py::test_deflection_submit_accepts_zendesk_full_thread_blob tests/test_atlas_billing_content_ops_deflection_paid_flow.py -q` - 3 passed, 1 warning from `torch.cuda` importing deprecated `pynvml`.
- Passed after rebase:
  - `python -m pytest tests/test_extracted_content_deflection_submit.py::test_deflection_submit_accepts_zendesk_full_thread_blob tests/test_atlas_billing_content_ops_deflection_paid_flow.py -q` - 3 passed, 1 warning from `torch.cuda` importing deprecated `pynvml`.
  - Python compile check for `tests/test_extracted_content_deflection_submit.py` - passed.
  - `python scripts/sync_pr_plan.py plans/PR-Deflection-Live-Artifact-Model-Export-Contract.md --check` - passed.
- Pending before push:
  - `bash scripts/push_pr.sh <pr-body-file>` local review wrapper before push.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Deflection-Live-Artifact-Model-Export-Contract.md` | 107 |
| `tests/test_extracted_content_deflection_submit.py` | 19 |
| **Total** | **126** |
