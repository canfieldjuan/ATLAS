# PR-Content-Ops-Report-Claim-Id-Ledger-Validation

## Why this slice exists

PR #1403 closed the live report proof gap by ensuring evidence-backed report
sections never persist empty `claim_ids`. Its review left one non-blocking but
valid hardening NIT: model-supplied `claim_ids` are still copied through without
checking whether they exist in the narrative-plan claim ledger. That makes the
field prompt-trusted rather than verifiable. This slice turns the NIT into code:
claim IDs copied from the model must either be verified against the same
narrative-plan section or replaced by an explicitly tagged fallback.

## Scope (this PR)

Ownership lane: content-ops/report-traceability-hardening
Slice phase: Production hardening

1. Validate model-supplied report section `claim_ids` against that same
   section's narrative-plan `claim_ids` when they exist.
2. Tag verified model claim IDs with `claim_id_source=model_verified`.
3. Drop unverified model claim IDs and record them in metadata instead of
   silently preserving dangling references.
4. Keep the existing fallback order after validation: section-level
   narrative-plan claim IDs, then deterministic section-local derived IDs for
   evidence-backed sections.
5. Add focused generator tests for verified, mixed, invalid, wrong-section, and
   no-ledger model-supplied IDs.

### Review Contract
- Acceptance criteria:
  - [ ] A model-supplied claim ID that exists in the same narrative-plan
        section is persisted and tagged `claim_id_source=model_verified`.
  - [ ] A model-supplied claim ID that exists only in a different
        narrative-plan section is not persisted as an authoritative ID for the
        current section.
  - [ ] A model-supplied claim ID that is not in the section ledger is not
        persisted as an authoritative claim ID.
  - [ ] Dropped unverified model IDs are visible in section metadata.
  - [ ] When model IDs are invalid, the existing fallback order still applies:
        same-section narrative-plan IDs first, then derived section-local IDs
        when evidence exists.
  - [ ] If there is no narrative-plan ledger, arbitrary model-supplied IDs are
        not trusted; evidence-backed sections get the deterministic derived
        fallback instead.
- Affected surfaces: extracted content pipeline report generation.
- Risk areas: traceability truthfulness, backward compatibility, extracted
  package behavior.
- Reviewer rules triggered: R1, R2, R10.

### Files touched

- `extracted_content_pipeline/report_generation.py`
- `plans/PR-Content-Ops-Report-Claim-Id-Ledger-Validation.md`
- `tests/test_extracted_report_generation.py`

## Mechanism

The #1403 helper returns the narrative-plan claim IDs keyed by section id.
Section normalization then handles parsed `claim_ids` in this order:

1. Clean the model-supplied IDs.
2. If same-section narrative-plan claim IDs exist, keep only model IDs in that
   section's claim-id set and tag the section `claim_id_source=model_verified`.
3. Record any rejected model IDs under `dropped_unverified_claim_ids` so the
   audit trail is explicit.
4. If no verified IDs remain, copy the same-section narrative-plan IDs and tag
   `claim_id_source=narrative_plan`.
5. If no plan IDs remain and `evidence_ids` is non-empty, derive the existing
   deterministic section-local ID and tag `claim_id_source=derived_section`.

When no same-section ledger exists, model-supplied IDs are treated as
unverified and the same fallback path applies. This keeps every persisted claim
ID either same-section verified, directly copied from the narrative plan, or
clearly derived.

## Intentional

- No new prompt change is included. The prompt already tells the model not to
  invent upstream claim-ledger IDs; this slice makes that rule load-bearing in
  code.
- No quality-gate blocker is added. Invalid model IDs are normalized into a
  safe persisted shape instead of failing generation, because the report still
  has evidence-backed traceability after fallback.
- No new claim ledger product is introduced. The only authoritative ledger for
  this slice is the same-section claim list in the already-attached
  `canonical_reasoning.narrative_plan`.

## Deferred

- Full `extracted_evidence_to_story` claim-ledger wiring remains deferred; this
  slice only validates against the report's existing same-section
  narrative-plan ledger.

Parked hardening: none.

## Verification

- Command: pytest tests/test_extracted_report_generation.py -- 33 passed.
- Command: bash scripts/run_extracted_pipeline_checks.sh -- validation/import/audit
  preflight passed; extracted reasoning core 295 passed; extracted content
  pipeline 3498 passed, 10 skipped.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/report_generation.py` | 37 |
| `plans/PR-Content-Ops-Report-Claim-Id-Ledger-Validation.md` | 110 |
| `tests/test_extracted_report_generation.py` | 154 |
| **Total** | **301** |
