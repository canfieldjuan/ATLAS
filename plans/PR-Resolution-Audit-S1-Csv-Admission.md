# PR-Resolution-Audit-S1-Csv-Admission

## Why this slice exists

Issue #1993 and the merged current-code tracker name S1 as the first real
Resolution Audit CSV remediation slice after the tracker work. The live path
already has a richer CSV loader than the old audit wording implied, but it
still admits some unsafe shapes too quietly: duplicate headers overwrite data,
data-looking first rows can be mistaken for fallback headers, legacy encoding
fallback can be hard to distinguish from clean UTF-8 input, and missing stable
ticket IDs fall back to row-order IDs without any trust signal.

Root cause: CSV admission and support-ticket identity are split across the
generic CSV loader and support-ticket package normalization, so the loader can
silently preserve ambiguous rows while the package later treats row-order
fallback identity as normal source identity. This change fixes the root for
this slice by making unsafe CSV header shapes fail closed, carrying explicit
parser warnings for accepted-but-risky CSV inputs, and surfacing missing-ID
identity diagnostics where repeat/cost counts are produced.

Diff size note: the first pass was under the 400 LOC target. Review found
three same-class boundary gaps in the real paths, so this revision exceeds the
soft cap to cover the full variant class: downstream alias duplicate headers,
short-prose fallback headers, and submit-response diagnostics.

## Scope (this PR)

Ownership lane: resolution-audit-csv
Slice phase: Functional validation

1. Tighten real CSV loader admission for duplicate headers and clearly
   data-looking low-confidence fallback headers.
2. Preserve existing successful CSV export paths while surfacing accepted
   legacy-encoding and low-confidence-header warnings.
3. Add support-ticket missing-ID diagnostics so row-order fallback identity is
   visible before repeat/cost counts are trusted.
4. Forward missing-ID diagnostics through the Resolution Audit submit response.
5. Update the #1993 tracker checklist with this S1 PR once opened.

### Review Contract

- Real loader path is exercised: `load_csv_source_rows_result_from_file` and
  `build_support_ticket_input_package`, not hand-rolled parser fakes.
- Duplicate headers fail closed with a structured
  `CsvCustomerDataParseError`.
- Headerless/data-looking CSV does not silently treat the first data row as a
  trusted header.
- Accepted legacy encoding and low-confidence fallback-header cases return
  explicit warnings instead of pretending to be clean UTF-8/hinted CSV.
- Missing stable source IDs are counted in package metadata/warnings without
  changing buyer-facing report/snapshot/email/PDF shape.
- Resolution Audit submit forwards the missing-ID warning/count when present.
- No user-facing report, snapshot, landing, email, PDF, pricing, or checkout
  shape changes.
Reviewer rules triggered: R1 requirements match, R2 test evidence, R10
detector/gate predicates, R14 codebase verification.

### Files touched

- `docs/audits/resolution-audit-csv/CURRENT_CODE_REMEDIATION_ARC.md`
- `extracted_content_pipeline/api/control_surfaces.py`
- `extracted_content_pipeline/campaign_customer_data.py`
- `extracted_content_pipeline/support_ticket_input_package.py`
- `plans/PR-Resolution-Audit-S1-Csv-Admission.md`
- `tests/test_extracted_campaign_customer_data.py`
- `tests/test_extracted_campaign_source_adapters.py`
- `tests/test_extracted_content_deflection_submit.py`
- `tests/test_extracted_support_ticket_input_package.py`

## Mechanism

- Add a small header-admission check after CSV header selection and before row
  materialization. Duplicate normalized header names fail with a structured
  parse error; downstream-known compact alias duplicates also fail so the
  loader and support-ticket consumers agree. No-hint fallback headers that
  look like customer prose fail as low-confidence/headerless input.
- Keep fallback headers possible for current compatibility, but attach a
  warning when the accepted header has no known support/customer-data hint.
- Make legacy CP1252/Latin-1 fallback emit an explicit warning even when the
  file decodes cleanly.
- In support-ticket normalization, count rows that use row-order fallback IDs
  and surface that count/examples in package metadata and warnings.
- In the Resolution Audit submit endpoint, forward missing-ID diagnostics when
  they exist; omit zero-count metadata to keep unrelated submit payloads stable.
- Add fixtures against the real loader/package entrypoints for duplicate
  headers, downstream alias duplicates, meaningful punctuation near-misses,
  data-looking/headerless input, short-prose headerless input, over-wide rows,
  legacy encoding warning propagation, submit diagnostics, and missing-ID
  diagnostics.

## Intentional

- Do not reject every no-hint CSV header. Current evidence tooling has
  intentionally accepted no-hint shapes that later reject as zero-usable rows;
  this slice only blocks headers that clearly look like data/prose and warns on
  the remaining no-hint path.
- Do not hard-reject every question-mark header cell. A no-hint header like
  `Escalated?` / `Resolved?` is suspicious enough to warn, not enough by
  itself to prove the first row is customer prose.
- Do not deduplicate missing-ID rows in this slice. Row identity policy can
  affect repeat counts and therefore buyer-visible report semantics; this slice
  surfaces the risk as diagnostics and leaves the policy decision to the next
  clustering/counting slice if needed.
- Do not change any report/snapshot/email/PDF output shape.

## Deferred

- S2 remains row-key normalization cache.
- S5 may consume the missing-ID diagnostics when it calibrates clustering and
  repeat-count behavior.

Parked hardening: none.

## Verification

- Command: python -m py_compile extracted_content_pipeline/campaign_customer_data.py extracted_content_pipeline/support_ticket_input_package.py extracted_content_pipeline/api/control_surfaces.py
- Command: python -m pytest tests/test_extracted_support_ticket_input_package.py -q (83 passed)
- Command: python -m pytest tests/test_extracted_campaign_source_adapters.py -q (150 passed)
- Command: python -m pytest tests/test_extracted_content_deflection_submit.py -q (92 passed)
- Command: python -m pytest tests/test_evaluate_csv_admission_threshold_evidence.py -q (10 passed)
- Command: python -m pytest tests/test_smoke_content_ops_support_ticket_package.py -q (24 passed)
- Command: python scripts/maturity_sweep.py extracted_content_pipeline --tests-root tests --baseline tests/maturity_sweep/baseline_extracted_content_pipeline.json (ratchet passed)
- Command: bash scripts/run_extracted_pipeline_checks.sh (5117 passed, 21 skipped)
- Command: python scripts/sync_pr_plan.py plans/PR-Resolution-Audit-S1-Csv-Admission.md
- Command: bash scripts/local_pr_review.sh --allow-dirty (passed)

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/audits/resolution-audit-csv/CURRENT_CODE_REMEDIATION_ARC.md` | 2 |
| `extracted_content_pipeline/api/control_surfaces.py` | 9 |
| `extracted_content_pipeline/campaign_customer_data.py` | 171 |
| `extracted_content_pipeline/support_ticket_input_package.py` | 23 |
| `plans/PR-Resolution-Audit-S1-Csv-Admission.md` | 140 |
| `tests/test_extracted_campaign_customer_data.py` | 4 |
| `tests/test_extracted_campaign_source_adapters.py` | 26 |
| `tests/test_extracted_content_deflection_submit.py` | 39 |
| `tests/test_extracted_support_ticket_input_package.py` | 180 |
| **Total** | **594** |
