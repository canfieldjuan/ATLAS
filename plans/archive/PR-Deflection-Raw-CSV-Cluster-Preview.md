# PR-Deflection-Raw-CSV-Cluster-Preview

## Why this slice exists

#1386 names the next launch-gating deflection slice: report quality on raw,
untagged CSVs plus #1384 CSV ingestion/preview discipline. The active public
funnel in `atlas-portfolio` does not parse CSV rows client-side; it uploads the
raw CSV to private Blob, then posts those bytes to ATLAS
`/content-ops/deflection-reports/submit`. That means the report-quality fix
belongs in the ATLAS deflection submit/report path, not as a detached frontend
cleanup.

PR #1391 already landed the P0 CSV parser hardening (BOM, cp1252 fallback,
delimiter sniffing, fail-loud parse errors). The remaining production risk is
semantic but still deterministic: untagged real help-desk exports fragment into
singletons because the support-ticket package preview clusters by explicit
`pain_category` or exact title, and the FAQ report fallback can also use exact
subject text. A buyer can then see a weak or sparse preview before checkout, and
the paid report can rank near-duplicate questions as separate issues.

This PR exceeds the 400 LOC soft cap because the launch-gating behavior is one
end-to-end contract: the same deterministic grouping must be created in the
support-ticket input package, consumed by the paid FAQ report, exposed in the
deflection submit diagnostics/snapshot path, and proven by paired regression
tests. Splitting the helper from the report/submit tests would leave an
unreviewable half-fix on the paid funnel path.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Production hardening

1. Add deterministic, no-LLM support-ticket clustering for the deflection
   submit/report path. Explicit categories still win, but untagged rows get a
   token-set/keyword cluster hint based on normalized customer wording instead
   of exact subject text.
2. Thread the cluster hint into `TicketFAQMarkdownService` so the locked
   snapshot and paid FAQ report group the same raw-ticket clusters that the
   input-provider preview reports.
3. Strip common HTML ticket-body markup during support-ticket normalization so
   Zendesk/Help Scout/Freshdesk exports do not feed tags into clustering,
   customer wording examples, or report evidence.
4. Add messy untagged CSV/report tests that prove repeated raw questions group
   together and that reply-less tickets still produce non-empty review-needed
   answer drafts.

### Review Contract

- Acceptance criteria:
  - [ ] Active deflection path is named and preserved:
        `atlas-portfolio` raw CSV bytes -> ATLAS multipart
        `/deflection-reports/submit` -> `build_support_ticket_input_package`
        -> `TicketFAQMarkdownService`.
  - [ ] No LLM, embedding, Ollama, or model-backed clustering is introduced;
        clustering is deterministic token/keyword logic.
  - [ ] Untagged raw support-ticket rows with varied wording but shared intent
        group into the same preview cluster and the same FAQ report item.
  - [ ] Explicit `pain_category`/category labels remain authoritative for
        tagged rows.
  - [ ] HTML body text is normalized to readable customer wording before
        clustering/evidence output.
  - [ ] Sparse/reply-less tickets still generate non-empty
        `draft_needs_review` answers rather than empty paid-report sections.
  - [ ] Parser hardening from #1391 remains unchanged except where reused by
        tests.
- Affected surfaces: extracted package support-ticket normalization,
  deterministic FAQ report generation, deflection submit diagnostics/snapshot
  proof; no DB, billing, Stripe, MCP, or portfolio UI mutation in this PR.
- Risk areas: public API backcompat, report quality, deterministic grouping
  false positives, extracted-package CI enrollment.
- Reviewer rules triggered: R1, R2, R5, R10, R12.

### Files touched

- `extracted_content_pipeline/api/control_surfaces.py`
- `extracted_content_pipeline/manifest.json`
- `extracted_content_pipeline/support_ticket_clustering.py`
- `extracted_content_pipeline/support_ticket_input_package.py`
- `extracted_content_pipeline/ticket_faq_markdown.py`
- `plans/PR-Deflection-Raw-CSV-Cluster-Preview.md`
- `tests/test_extracted_content_deflection_submit.py`
- `tests/test_extracted_support_ticket_input_package.py`
- `tests/test_extracted_ticket_faq_markdown.py`
- `tests/test_smoke_content_ops_support_ticket_package.py`

## Mechanism

Introduce a small package-owned clustering helper that:

1. converts HTML-ish ticket text to plain text with the standard library;
2. tokenizes customer wording with stopword removal, light synonym folding, and
   simple singularization;
3. prefers explicit category labels when present;
4. groups untagged rows by deterministic token-set overlap; and
5. exposes a stable `support_ticket_cluster`/`support_ticket_cluster_key` hint
   for downstream FAQ generation.

`build_support_ticket_input_package` applies the helper after row
normalization. Its `top_ticket_clusters` preview uses the same grouping that it
threads into each normalized `source_material` row. `TicketFAQMarkdownService`
then prefers that hint before falling back to exact source title, so the free
snapshot and paid report rank the same repeated raw-ticket cluster rather than
splitting near-duplicate subjects.

The tests use realistic messy rows (HTML body, varied subjects, no
`pain_category`, no agent resolution) rather than the tagged demo fixture.

## Intentional

- No LLM, embedding, or model repair pass: the deflection trust story remains
  deterministic and no customer ticket text leaves the stack for clustering.
- This PR does not mutate Stripe, checkout, delivery, or portfolio UI code. The
  active pre-check surface is the existing locked snapshot/results page after
  submit and before checkout; a follow-up can improve how the portfolio renders
  any extra backend diagnostics.
- The clustering logic is conservative token overlap, not full semantic
  clustering. False positives are riskier than a few separate clusters in a
  paid support report.

## Deferred

- Portfolio rendering polish: if the product wants a separate cluster-quality
  widget before the locked results page, build it in `atlas-portfolio` after
  this backend response is proven.
- Sanitized real provider exports: this PR adds messy representative fixtures;
  a later validation slice should add sanitized Zendesk/Freshdesk/Help Scout
  exports when available from real accounts.

Parked hardening: none.

## Verification

- Command: pytest tests/test_extracted_support_ticket_input_package.py tests/test_extracted_ticket_faq_markdown.py tests/test_extracted_content_deflection_submit.py tests/test_smoke_content_ops_support_ticket_package.py
  - 214 passed
- Command: pytest tests/test_content_ops_deflection_report.py
  - 36 passed
- Command: pytest tests/test_smoke_content_ops_support_ticket_package.py
  - 9 passed
- Command: bash scripts/validate_extracted_content_pipeline.sh
  - passed
- Command: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
  - passed
- Command: python scripts/audit_extracted_standalone.py --fail-on-debt
  - `Atlas runtime import findings: 0`
- Command: bash scripts/check_ascii_python.sh
  - passed
- Command: git diff --check
  - passed
- Command: bash scripts/run_extracted_pipeline_checks.sh
  - 3515 passed, 10 skipped, 1 warning

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/api/control_surfaces.py` | 2 |
| `extracted_content_pipeline/manifest.json` | 3 |
| `extracted_content_pipeline/support_ticket_clustering.py` | 609 |
| `extracted_content_pipeline/support_ticket_input_package.py` | 59 |
| `extracted_content_pipeline/ticket_faq_markdown.py` | 28 |
| `plans/PR-Deflection-Raw-CSV-Cluster-Preview.md` | 166 |
| `tests/test_extracted_content_deflection_submit.py` | 65 |
| `tests/test_extracted_support_ticket_input_package.py` | 197 |
| `tests/test_extracted_ticket_faq_markdown.py` | 50 |
| `tests/test_smoke_content_ops_support_ticket_package.py` | 4 |
| **Total** | **1183** |
