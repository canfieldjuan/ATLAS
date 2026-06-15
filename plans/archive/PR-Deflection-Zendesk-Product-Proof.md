# PR-Deflection-Zendesk-Product-Proof

## Why this slice exists

Issue #1419 needs product-shaped evidence that the paid FAQ deflection report
can ingest a real Zendesk full-thread export and produce publishable answer
drafts from public agent replies. The previous Zendesk proof was a four-row
fixture; it proved the object shape but not a seeded trial tenant report output
that can be inspected as buyer-shaped proof.

#1567 now owns the Zendesk exporter URL fix and the committed 50-ticket
product-proof corpus. This PR no longer changes exporter behavior. It keeps the
sanitized proof artifacts generated from the seeded Zendesk trial export and
documents exactly what that report proof did and did not validate. A review pass
found the committed samples also show buyer-facing label defects (`[Atlas seed
N]` subject-prefix pollution and repeated `What should I do about atla?`
labels), so this PR now records those as explicit output-quality boundaries
instead of overstating buyer-readiness.

The diff is above the 400 LOC target because the slice commits sanitized proof
artifacts (`summary.json` plus report excerpt). Those extra lines are the
reviewable evidence the slice exists to produce.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Functional validation

1. Commit sanitized proof artifacts from the live Zendesk trial export and
   deterministic FAQ deflection report generation.
2. Add a proof doc that separates Zendesk product/integration evidence from
   CFPB full-volume stress evidence and names the remaining portfolio-wrapper
   blockers.
3. Extend CI-facing tests to pin the proof boundaries, including private-note
   non-leakage.
4. Explicitly defer exporter URL behavior to #1567.
5. Pin the output-quality boundary: generated samples include synthetic subject
   prefixes and the degraded `What should I do about atla?` label, so this is
   not a buyer-ready FAQ quality pass.

### Review Contract

Acceptance criteria:

- The committed Zendesk product proof records 50 tickets, 126 comments, more
  than zero answer drafts, status diagnostics, CSAT diagnostics, and no
  private-note markers in the sanitized report excerpt.
- The doc does not claim raw export, paid unlock, email delivery, or the public
  portfolio wrapper were proven.
- The plan and PR body do not claim ownership of Zendesk exporter URL behavior;
  #1567 remains the canonical exporter fix.
- The proof doc names the output-quality defects visible in the committed
  artifacts and does not claim buyer-ready question-label quality.

Affected surfaces:

- Deflection validation docs and sanitized proof artifacts.

Risk areas:

- Accidentally committing raw Zendesk export data.
- Overclaiming product proof as full-volume or paid-funnel proof.
- Overclaiming generated report samples as buyer-ready despite degraded labels.
- Private internal notes leaking into committed report excerpts.

Reviewer rules triggered: R1, R2, R7, R8, R10, R12, R13, R14.

### Files touched

- `docs/extraction/validation/deflection_zendesk_product_proof_2026-06-14.md`
- `docs/extraction/validation/fixtures/deflection_zendesk_product_proof_20260614/report_excerpt.md`
- `docs/extraction/validation/fixtures/deflection_zendesk_product_proof_20260614/summary.json`
- `plans/PR-Deflection-Zendesk-Product-Proof.md`
- `tests/test_content_ops_deflection_source_proof_docs.py`

## Mechanism

The live proof run used the exporter directly against the seeded Zendesk trial
tenant, normalized the `{ticket, comments}` artifact through the full-thread
importer, built the support-ticket input package, and generated the deterministic
FAQ deflection report artifact. Only sanitized scalar metrics and short answer
excerpts are committed; the raw export remains under `tmp/` and outside git.

Exporter URL behavior is intentionally out of this PR after #1567. #1567 is the
canonical fix for the live-compatible Zendesk cursor endpoint and the committed
50-ticket product-proof corpus; this PR is the report-proof artifact slice.

## Intentional

- No raw Zendesk export is committed. The committed artifacts are summary JSON
  and a short report excerpt only.
- This PR does not mutate Stripe state, unlock paid reports, or send email.
- This PR does not change the Zendesk exporter. The earlier `per_page` follow-up
  was dropped after #1567 became the canonical exporter fix.
- This PR keeps the degraded report samples instead of regenerating or editing
  them, because the slice is proof honesty: the artifacts should show the real
  output and the doc should name the boundary.
- This PR does not claim the portfolio wrapper was proven. The hosted
  `ATLAS_API_BASE_URL` export route was stale during the proof window, and the
  local portfolio env lacked the Blob/export access-token prerequisites.
- The proof records `output_checks.resolution_evidence_scoped: false` as a
  boundary because two one-ticket seeded clusters had missing question scope.
  That is not fixed in this functional-validation slice.

## Deferred

- Re-run the public portfolio wrapper once the deployed Atlas route accepts
  `/api/v1/content-ops/zendesk-export/full-thread` and local/deployed env has
  `BLOB_READ_WRITE_TOKEN` plus `ATLAS_DEFLECTION_ZENDESK_EXPORT_ACCESS_TOKEN`.
- Keep CFPB as the full-volume stress proof; this Zendesk trial export is the
  product-shaped proof corpus, not a scale corpus.
- Investigate the two one-ticket missing-question-scope clusters if that
  quality boundary persists on future seeded exports.
- Any exporter URL or page-size validation belongs to #1567 follow-up work, not
  this proof-artifact PR.
- Fix synthetic subject-prefix pollution and the repeated degraded `atla` label
  in the next output-quality slice before claiming buyer-ready FAQ quality.

Parked hardening: none.

## Verification

- `pytest tests/test_content_ops_deflection_source_proof_docs.py -q` - passed.
- `scripts/validate_extracted_content_pipeline.sh` - passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` - passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - passed.
- `scripts/check_ascii_python.sh` - passed.
- `scripts/run_extracted_pipeline_checks.sh` - 4200 passed, 10 skipped.
- After rebasing on #1567 and dropping exporter changes:
  `pytest tests/test_content_ops_deflection_source_proof_docs.py -q` - 4 passed.
- After adding the output-quality boundary:
  `pytest tests/test_content_ops_deflection_source_proof_docs.py -q` - 5 passed.
  `tests/test_content_ops_deflection_source_proof_docs.py` compiled with
  `py_compile` - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/extraction/validation/deflection_zendesk_product_proof_2026-06-14.md` | 153 |
| `docs/extraction/validation/fixtures/deflection_zendesk_product_proof_20260614/report_excerpt.md` | 32 |
| `docs/extraction/validation/fixtures/deflection_zendesk_product_proof_20260614/summary.json` | 178 |
| `plans/PR-Deflection-Zendesk-Product-Proof.md` | 145 |
| `tests/test_content_ops_deflection_source_proof_docs.py` | 56 |
| **Total** | **564** |
