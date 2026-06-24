# PR-Deflection-Report-Paid-Consumer

## Why this slice exists

This is #1805 slice 4 after the ATLAS paid `report_projection` contract was
published (#1815), runtime-enforced (#1817), and turned into the in-repo
frontend TypeScript artifact (#1821). The remaining in-repo paid browser
surface still reads `artifact.report_model.sections` with hand-coded section
IDs inside `portfolio-ui/api/content-ops/deflection/result-page.js`.

Root cause: the paid result-page QA observer is still consumer-inferred even
though ATLAS now owns the paid report-model contract. Snapshot proxy code
already consumes a generated JS contract, but the paid result page does not
have the matching generated API-side report-model contract, so section IDs and
data assumptions can drift from `report_projection`.

This change fixes the root for the in-repo API renderer by generating the
report-model API contract from `report_projection` and using it in the paid
result-page report-model lookup. It does not wire `atlas-portfolio` yet, and it
does not expose the paid model to the free React snapshot surface.

This slice is slightly over the 400 LOC soft cap because the generated
report-model API contract enumerates every paid section and field. Splitting
the generated artifact from its consumer would leave the paid renderer on
hand-coded assumptions for another PR.

## Scope (this PR)

Ownership lane: deflection/report-contract-1805
Slice phase: Production hardening

Max files: 8

1. Extend the existing frontend contract generator to emit a JS report-model
   API contract for the paid result-page renderer, parallel to the existing
   snapshot API contract.
2. Use the generated report-model section IDs in the paid result-page
   `report_model` section lookup instead of hard-coded string-only section
   assumptions.
3. Keep the full `DeflectionStructuredReport` contract paid-only: do not import
   or parse it in the free React snapshot page.
4. Add generator and result-page tests proving the contract is current,
   consumed by the paid renderer, and absent from the free/locked snapshot
   surface.
5. Enroll the generated API contract in the deflection product-surface
   manifest and test that generated API contracts stay enrolled.

### Review Contract
- Acceptance criteria:
  - [ ] `portfolio-ui/api/content-ops/deflection/report-model-contract.js` is
        generated from `report_projection`, not hand-written.
  - [ ] `scripts/generate_deflection_frontend_contract_types.py --check`
        fails when the report-model API contract is stale.
  - [ ] `result-page.js` consumes the generated report-model section IDs for
        paid report-model lookup.
  - [ ] The free React snapshot page continues to import only
        `DeflectionResultPageSnapshot`, and does not import or parse the full
        paid `DeflectionStructuredReport`.
  - [ ] Locked/free result-page renders do not embed paid-only report-model
        fields such as `source_ids`, `evidence_quotes`, or `top_evidence`.
  - [ ] Generated deflection API contract outputs are present in
        `tests/maturity_sweep/deflection_product_surface_manifest.json`.
- Affected surfaces: contract generator, generated in-repo API contract,
  paid result-page renderer, portfolio-ui deflection result tests, deflection
  report CI path enrollment.
- Risk areas: paid/free boundary, stale generated contracts, result-page QA
  observation drift, consumer shape assumptions.
- Reviewer rules triggered: R1, R2, R3, R10, R12, R14; boundary-probe required
  because this extends and consumes a generated contract/gate on a paid/free
  privacy boundary.

### Files touched

- `.github/workflows/atlas_content_ops_deflection_report_checks.yml`
- `plans/PR-Deflection-Report-Paid-Consumer.md`
- `portfolio-ui/api/content-ops/deflection/report-model-contract.js`
- `portfolio-ui/api/content-ops/deflection/result-page.js`
- `portfolio-ui/scripts/faq-deflection-result-page.test.mjs`
- `scripts/generate_deflection_frontend_contract_types.py`
- `tests/maturity_sweep/deflection_product_surface_manifest.json`
- `tests/test_generate_deflection_frontend_contract_types.py`

## Mechanism

The generator already reads the backend deflection report-model contract and
produces snapshot TS, snapshot API JS, and paid report-model TS. This slice
adds a fourth output:

1. read `report_projection.sections`;
2. emit `DEFLECTION_REPORT_MODEL_SCHEMA_VERSION`,
   `DEFLECTION_REPORT_SECTION_IDS`, `DEFLECTION_REPORT_CONDITIONAL_SECTION_IDS`,
   and per-section field tuples in API-friendly JS;
3. include the output in `main(... --check)`; and
4. import `DEFLECTION_REPORT_SECTION_IDS` in `result-page.js` so
   `reportModelSectionData()` validates requested section IDs against the
   generated contract before reading paid `report_model.sections`; and
5. fail closed at module load if the generated section or field constants no
   longer contain the members the paid QA observer reads.

The paid/free boundary stays at `artifact_status === "unlocked"` and the free
React page remains snapshot-only.

## Intentional

- This PR does not render new paid UI sections from the generated type. It
  makes the existing paid report-model observer consume a generated contract
  first, keeping the slice reviewable.
- The API contract is JS rather than TS because the paid result-page renderer is
  a Vercel API module running as ESM JavaScript.

## Deferred

- `atlas-portfolio` cross-repo paid report-model consumption remains the next
  repo/lane after this in-repo paid renderer consumes the ATLAS-owned contract.
- Hosted-safe construction from `hosted_consumer_safe_fields` and runtime
  parity for free-surface consumption remains a separate consumer-boundary
  slice. This PR deliberately does not feed the full paid model to free/hosted
  surfaces.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_generate_deflection_frontend_contract_types.py tests/test_deflection_product_surface_manifest.py -q`
  - 17 passed.
- `python scripts/generate_deflection_frontend_contract_types.py --check`
  - all four generated artifacts current.
- `npm --prefix portfolio-ui run test:deflection-result`
  - 30 result-page tests passed.
- Python py_compile for `scripts/generate_deflection_frontend_contract_types.py`,
  `tests/test_generate_deflection_frontend_contract_types.py`, and
  `tests/test_deflection_product_surface_manifest.py`
  - passed.
- `git diff --check`
  - passed.
- `scripts/validate_extracted_content_pipeline.sh`
  - passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline`
  - passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt`
  - passed.
- `scripts/check_ascii_python.sh`
  - passed.
- `scripts/check_deflection_product_surface_manifest.py`
  - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_content_ops_deflection_report_checks.yml` | 2 |
| `plans/PR-Deflection-Report-Paid-Consumer.md` | 159 |
| `portfolio-ui/api/content-ops/deflection/report-model-contract.js` | 104 |
| `portfolio-ui/api/content-ops/deflection/result-page.js` | 51 |
| `portfolio-ui/scripts/faq-deflection-result-page.test.mjs` | 103 |
| `scripts/generate_deflection_frontend_contract_types.py` | 71 |
| `tests/maturity_sweep/deflection_product_surface_manifest.json` | 1 |
| `tests/test_generate_deflection_frontend_contract_types.py` | 90 |
| **Total** | **581** |
