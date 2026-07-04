# PR-Resolution-Audit-S2-Parsing-Clustering

## Why this slice exists

Slice 2 of the Resolution Audit CSV audit arc (epic #1956, slice issue #1958) --
the empirical core. Phase 0 (#1957, merged) mapped the pipeline and predicted a
fragmentation failure in clustering. This slice attacks the product's central
claim ("repeat questions cluster correctly") and the parser's robustness by
building adversarial fixtures and running them through the REAL pipeline code,
then records severity-rated findings with reproductions. It confirmed the
fragmentation failure is real and customer-money-wrong.

## Scope (this PR)

Ownership lane: workflow/process
Slice phase: workflow/process

1. Phase 1 -- empirically attack the CSV parser (encodings, structure, content,
   Zendesk-specific), quantifying crash / silent-drop / silent-corruption.
2. Phase 2 -- empirically attack clustering (fragmentation, over-merge, threshold
   sweep, singletons, junk-as-representative, determinism).
3. Record findings in `docs/audits/resolution-audit-csv/FINDINGS.md`
   (severity-rated, file:line + reproduction fixture each).
4. No product code changes; all fixtures/runners stay in gitignored `_audit_scratch/`.

### Files touched

- `docs/audits/resolution-audit-csv/FINDINGS.md`
- `plans/PR-Resolution-Audit-S2-Parsing-Clustering.md`

### Review Contract

- Acceptance criteria:
  - `FINDINGS.md` records every defect with severity, `file:line`, and a
    reproduction; the CRITICAL findings (C1, C3, F1, F2, F7) were independently
    re-run by the reviewer against `origin/main`.
  - No product code is modified; `_audit_scratch/` stays gitignored and out of the diff.
- Affected surfaces: documentation only. No runtime, API, auth, billing, delivery,
  webhook, or UI surface is introduced, so no reachability proof applies.
- Rule mapping: the diff triggers no review-rule classes (no auth/token/permission,
  no webhooks/jobs, no defect-class review comments) -- it is a read-only audit deliverable.

## Mechanism

49 CSV fixtures (Phase 1) and ~15 clustering fixtures (Phase 2) built under
`_audit_scratch/` and run through the real `_load_csv_dict_rows_result`,
`build_support_ticket_input_package`, and `FAQDeflectionReportService.generate ->
build_ticket_faq_markdown`. Numbers read from the rendered `report_model`, not
computed by hand. Reviewer independently reproduced the CRITICALs.

## Intentional

- Findings only; refactor recommendations are Slice 4 (#1960).
- Product code untouched; experiments confined to gitignored `_audit_scratch/`.
- Clustering run WITHOUT embeddings (the production default; F7 shows the wrapper
  discards `embedding_port`), with a WITH-embeddings control that changed nothing.

## Deferred

- Performance findings (`FINDINGS.md` append) -- Slice 3 (#1959).
- `REFACTORS.md`, `PRESENTATION.md`, `SUMMARY.md` -- Slice 4 (#1960). SUMMARY leads
  with the F1/F2 clustering criticals and C1/C3 parser criticals.

## Verification

- Every finding produced by a real run against `origin/main`; reviewer re-ran C1
  (5->4 rows), C3 (`02-01-2026 -> 2026-02-01`), F1 (12 -> 1 cluster of 5 + 7
  non_repeat), F2 (junk `ticket_count=6` ranked #1, annualized $1,782), F7
  (`del kwargs`, embedding_port not forwarded).
- `git diff --check` clean; ASCII-only markdown.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/audits/resolution-audit-csv/FINDINGS.md` | 196 |
| `plans/PR-Resolution-Audit-S2-Parsing-Clustering.md` | 78 |
| **Total** | **274** |
