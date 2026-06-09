# PR-Content-Ops-Report-Claim-Ids

## Why this slice exists

PR #1399 proved the report generator can run through Gate A live coverage, but
the exported report sections carried populated `evidence_ids` and empty
`claim_ids`. That leaves the report artifact partially traceable: renderers and
review tools can see which evidence was cited, but not which report-level claims
the section is asserting. This production-hardening slice closes that live-proof
gap without changing the larger quality gate or introducing a new claim-ledger
product.

## Scope (this PR)

Ownership lane: content-ops/report-traceability-hardening
Slice phase: Production hardening

1. Ensure `ReportGenerationService` persists non-empty section `claim_ids` for
   evidence-backed sections even when the LLM omits the field.
2. Preserve model/reasoning-supplied `claim_ids` exactly after string
   normalization; do not rewrite authoritative IDs from a narrative plan.
3. Mark runtime-derived claim IDs in section metadata so they are transparent
   section-local traceability IDs, not fake upstream claim-ledger IDs.
4. Tighten the report prompt so future LLM output emits the same fallback shape
   the runtime enforces.
5. Add focused report-generation regression tests for supplied IDs, missing IDs,
   and blank section IDs.

### Review Contract
- Acceptance criteria:
  - [ ] A section with supplied `claim_ids` persists those IDs unchanged.
  - [ ] A section with `evidence_ids` but no `claim_ids` persists a stable,
        non-empty section-local claim ID.
  - [ ] A section whose `id` is blank still gets a deterministic fallback claim
        ID based on section order.
  - [ ] Runtime-derived claim IDs carry section metadata that makes the source
        explicit.
  - [ ] The report prompt instructs the model to emit claim IDs for every
        section and names the same section-local fallback.
- Affected surfaces: extracted content pipeline report generation and package
  prompt.
- Risk areas: traceability truthfulness, backward compatibility, extracted
  package sync discipline.
- Reviewer rules triggered: R1, R2, R10.

### Files touched

- `extracted_content_pipeline/report_generation.py`
- `extracted_content_pipeline/skills/digest/report_generation.md`
- `plans/PR-Content-Ops-Report-Claim-Ids.md`
- `tests/test_extracted_report_generation.py`

## Mechanism

`ReportGenerationService._build_draft` will normalize each parsed section
through a small helper before constructing `ReportSection`.

- Existing `claim_ids` are stripped, empty values are dropped, and order is
  preserved.
- If a section has no remaining `claim_ids` and the attached narrative plan has
  claim IDs for the same section id, those plan IDs fill the gap.
- If a section still has no remaining `claim_ids` but does have `evidence_ids`,
  the helper derives one stable section-local ID from the section's own `id`
  plus one-based section index. Blank/unsafe ids fall back to the section
  index.
- When a claim ID is derived, the section metadata receives
  `claim_id_source=derived_section` so consumers can distinguish runtime
  traceability from an upstream evidence-to-story claim ledger.
- Sections with neither evidence nor claim ids remain unchanged; the existing
  report quality pack already blocks no-reference outputs when quality gates are
  enabled.

The prompt change asks the model to emit `claim_ids` for every section and,
when no reasoning-context claim IDs exist, use the same section-local pattern
that the runtime fallback enforces.

## Intentional

- No quality-gate blocker is added in this slice. The quality pack already
  enforces evidence references; this slice hardens persisted traceability
  without turning historical prompt drift into a new generation blocker.
- No new top-level claim ledger is invented. Derived IDs are section-local and
  explicitly marked in metadata until `extracted_evidence_to_story` is wired
  into reports.
- No host/API/UI changes are included; the persisted `ReportSection` shape
  already exposes `claim_ids` through `as_dict()`.
- Cross-layer caller hints for `ReportGenerationService` and `_build_draft`
  were inspected. Host factories, the live execute harness, and report export
  consume the same `ReportDraft`/`ReportSection` shape; no caller-layer code
  change is needed because `claim_ids` already serialize through
  `ReportSection.as_dict()`.

## Deferred

- Full evidence-to-story claim-ledger integration remains deferred until
  `extracted_evidence_to_story` is wired into report generation. That future
  slice should replace derived section-local IDs with source-backed claim
  ledger IDs.

Parked hardening: none.

## Verification

- Command: pytest tests/test_extracted_report_generation.py -- 29 passed.
- Command: bash scripts/run_extracted_pipeline_checks.sh -- validation/import/audit
  preflight passed; extracted reasoning core 295 passed; extracted content
  pipeline 3485 passed, 10 skipped.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/report_generation.py` | 78 |
| `extracted_content_pipeline/skills/digest/report_generation.md` | 3 |
| `plans/PR-Content-Ops-Report-Claim-Ids.md` | 117 |
| `tests/test_extracted_report_generation.py` | 109 |
| **Total** | **307** |
