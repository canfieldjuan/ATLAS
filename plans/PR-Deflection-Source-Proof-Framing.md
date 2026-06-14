# PR-Deflection-Source-Proof-Framing

## Why this slice exists

Issue #1440 is now past the CFPB repeat-gate calibration. The important
operator decision from that review is that CFPB remains a full-volume stress
source, not the product-quality source for Zendesk-like support tickets.

The current validation docs mention CFPB full-volume proof and Zendesk
full-thread proof, but they do not make the source-role boundary explicit
enough. That creates a downstream risk: future slices can accidentally treat
the CFPB stress gate as buyer-readiness or Zendesk-product calibration. This
slice closes that framing gap before the next funnel proof builds on the wrong
source assumption.

## Scope (this PR)

Ownership lane: content-ops/deflection-full-volume
Slice phase: Functional validation

1. Update the deflection handoff runbook to name CFPB as stress/scale evidence
   and Zendesk-shaped full-thread inputs as product-quality/integration
   evidence.
2. Update the CFPB live proof to state that the `full-volume-cfpb` gates are
   stress-proof thresholds, not Zendesk/product-quality thresholds.
3. Update the Zendesk full-thread proof to state what it proves and what it
   does not prove because the committed fixture is small.
4. Add a docs contract test that fails if the source-role language disappears.
5. Enroll the test in the extracted pipeline check script and make docs-only
   validation changes trigger the extracted workflow.

### Review Contract

- Acceptance criteria:
  - [ ] CFPB docs explicitly call the corpus a stress/scale proof source, not
        Zendesk/product-quality calibration.
  - [ ] Zendesk full-thread docs explicitly call the fixture API-shaped
        integration/product-quality evidence, not full-volume proof.
  - [ ] The runbook tells operators when to use each source class.
  - [ ] A committed test protects those source-role claims.
  - [ ] The new test is enrolled in `scripts/run_extracted_pipeline_checks.sh`.
  - [ ] Docs-only edits under `docs/extraction/validation/**` trigger the
        extracted workflow so the contract test runs when the protected docs
        change.
- Affected surfaces: validation docs, docs contract test, extracted check
  enrollment.
- Risk areas: source-role drift, false product-quality claims, CI enrollment.
- Reviewer rules triggered: R1, R2, R10, R12, R14.

### Files touched

- `.github/workflows/extracted_pipeline_checks.yml`
- `docs/extraction/validation/content_ops_faq_deflection_submit_handoff_runbook.md`
- `docs/extraction/validation/deflection_full_volume_live_proof_2026-06-14.md`
- `docs/extraction/validation/deflection_zendesk_full_thread_proof_2026-06-13.md`
- `plans/PR-Deflection-Source-Proof-Framing.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_content_ops_deflection_source_proof_docs.py`

## Mechanism

This is a framing and validation-contract slice. It does not change submit,
generation, clustering, Stripe, portfolio rendering, delivery, or Zendesk
import code.

The docs get a source-role section that separates:

- CFPB full-volume stress proof: useful for hosted upload, parser scale,
  generation scale, snapshot/artifact gates, and delivery plumbing.
- Zendesk-shaped product/integration proof: useful for ticket/comment object
  shape, public requester/agent wording, private-note exclusion, and
  publishable-answer evidence.

The test reads the three docs as text and asserts the key claims remain present.
That is intentionally a contract test for proof language, not a parser or
runtime behavior test.

The extracted workflow path filters include `docs/extraction/validation/**` so
a docs-only PR that weakens the protected language still runs the contract
test.

## Intentional

- This does not replace the active #1563 Zendesk full-thread work. That PR owns
  the importer/control-surface path and remains off-limits to this session.
- This does not remove CFPB from #1440. CFPB still has value as stress/scale
  evidence; the fix is to stop treating it as Zendesk-like product-quality
  evidence.
- This does not use live Zendesk credentials. The live Zendesk API is a future
  integration proof after the source-role contract is explicit and after #1563
  lands.

## Deferred

- Live Zendesk API smoke remains future work after #1563 or a successor slice
  owns the full-thread import/control-surface path.
- Full buyer funnel proof remains blocked on portfolio snapshot config/data
  availability and deployed Stripe signing-secret alignment.

Parked hardening: none.

## Verification

- pytest tests/test_content_ops_deflection_source_proof_docs.py - 3 passed.
- python scripts/audit_extracted_pipeline_ci_enrollment.py - OK, 179 matching
  tests are enrolled.
- bash scripts/run_extracted_pipeline_checks.sh - 4,199 passed, 10 skipped,
  1 existing torch warning.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/extracted_pipeline_checks.yml` | 2 |
| `docs/extraction/validation/content_ops_faq_deflection_submit_handoff_runbook.md` | 23 |
| `docs/extraction/validation/deflection_full_volume_live_proof_2026-06-14.md` | 9 |
| `docs/extraction/validation/deflection_zendesk_full_thread_proof_2026-06-13.md` | 9 |
| `plans/PR-Deflection-Source-Proof-Framing.md` | 121 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_content_ops_deflection_source_proof_docs.py` | 43 |
| **Total** | **208** |
