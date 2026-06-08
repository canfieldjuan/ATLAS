# PR - Content-Ops Quality Gate Coverage Rows

## Why this slice exists

Issue #1353 says the deterministic quality packs and brand-voice audit already
exist; the missing seam is turning their findings into `ContentPR` coverage
rows. Without that adapter, the verdict engine stays dependent on hand-written
coverage even when deterministic gate evidence is already available.

This slice lands only the pure adapter. MCP transport, LLM-assisted review,
service threading, and generated-asset mutation stay deferred.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Vertical slice

Add deterministic coverage-row adapters:

1. Convert typed or decoded quality reports into resolved `CoverageRow` values.
2. Convert brand-voice audit mappings into resolved `CoverageRow` values.
3. Emit unresolved required rows for missing or malformed evidence.
4. Keep warning/info quality findings visible as optional resolved rows.

### Review Contract

- Acceptance criteria:
  - [ ] Passing quality reports emit a required `pass` row.
  - [ ] Blocker quality findings emit required `fail` rows.
  - [ ] Warning/info quality findings emit optional resolved rows.
  - [ ] Failed reports without blocker findings emit required `fail` rows.
  - [ ] Missing/malformed reports emit unresolved required rows.
  - [ ] Brand-voice warnings and banned terms emit required `fail` rows.
  - [ ] Brand-voice pass emits a required `pass` row.
  - [ ] No MCP, LLM, DB migration, tenant binding, or asset mutation is added.
- Affected surfaces: extracted package pure adapter, CI enrollment.
- Risk areas: silent approval, decoded input robustness, backcompat.
- Reviewer rules triggered: R1, R2, R5, R10, R12.

### Files touched

- `extracted_content_pipeline/coverage_rows.py`
- `extracted_content_pipeline/manifest.json`
- `tests/test_extracted_content_coverage_rows.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `plans/PR-Content-Ops-Quality-Gate-Coverage-Rows.md`

## Mechanism

The adapter reads quality reports structurally, so both value objects and
decoded mapping payloads work. A passed report gets one required pass row.
Blocker findings become required failures. Warning/info findings become optional
resolved rows. A failed report with no structured blocker still becomes a
required failure.

Brand-voice audit mappings follow the same fail-closed shape: `passed: true`
becomes a required pass; warnings and banned terms become required failures;
missing or malformed audits become unresolved rows so `review_verdict` blocks.

## Intentional

- This does not run quality packs; callers still choose the asset-specific
  pack.
- This does not store or attach rows to a `ContentPR`; it only builds rows.
- Missing evidence blocks as incomplete review evidence, not content failure.
- Warning/info quality findings remain visible without blocking by themselves.

## Deferred

- `PR-Content-Ops-Review-Service-Gate-Rows`: thread these rows through the host
  review workflow service.
- `PR-Content-Ops-Tenant-Binding-Bridge`: reconcile connector tenant binding
  with `TenantScope`.
- `PR-Marketer-Verification-MCP`: expose verify-only marketer tools after the
  remaining seams are wired.
- Parked hardening: none expected.

## Verification

- Focused coverage-row and Content-PR pytest command -- 32 passed.
- Extracted pipeline CI enrollment audit command -- 156 matching tests are
  enrolled.
- Extracted package guardrail commands -- validation, Atlas reasoning import
  ban, standalone audit, and ASCII policy passed.
- Local PR review command with a prepared PR body file -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/coverage_rows.py` | 144 |
| `tests/test_extracted_content_coverage_rows.py` | 154 |
| `plans/PR-Content-Ops-Quality-Gate-Coverage-Rows.md` | 95 |
| manifest + CI enrollment | 4 |
| **Total** | **397** |
