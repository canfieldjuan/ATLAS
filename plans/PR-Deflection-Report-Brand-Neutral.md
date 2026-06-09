# PR-Deflection-Report-Brand-Neutral

## Why this slice exists

Issue #1382 is a go-live polish fix for the paid deflection report artifact.
The generated customer-facing Markdown currently says "ATLAS" in the support
tax summary, the missing-date-window caveat, and the help-desk SEO wording
section. The paid product should read as a Deflection Snapshot / Full
Deflection Report deliverable, not expose the internal engine brand.

## Scope (this PR)

Ownership lane: go-live-deflection-cleanup
Slice phase: Product polish

1. Replace the three buyer-facing `ATLAS` mentions in
   `render_deflection_report()` output with product-neutral report wording.
2. Preserve the existing support-cost caveat and the no keyword-volume,
   search-rank, or traffic-promise doctrine.
3. Refresh the pinned frontend report example so the documented contract still
   matches the producer output.
4. Add a generated Markdown assertion that the deflection report example
   contains no `ATLAS` or `Atlas` engine branding.

### Review Contract

- Acceptance criteria:
  - [ ] Generated paid deflection report Markdown has no buyer-facing `ATLAS`
        or `Atlas` engine branding.
  - [ ] The support-cost estimate caveat remains present and still avoids a
        savings guarantee.
  - [ ] The help-desk SEO wording section still avoids keyword volume, search
        rank, and traffic promises.
  - [ ] The frontend contract example remains producer-matched.
- Affected surfaces: generated Markdown artifact, frontend contract fixture,
  report contract tests.
- Risk areas: buyer-facing copy drift, fixture drift.
- Reviewer rules triggered: R1, R2, R10.

### Files touched

- `docs/frontend/content_ops_faq_deflection_report_example.json`
- `extracted_content_pipeline/faq_deflection_report.py`
- `plans/PR-Deflection-Report-Brand-Neutral.md`
- `tests/test_content_ops_faq_report_contract_docs.py`

## Mechanism

The report renderer keeps the same sections, counts, cost math, ranked
questions, publishable answers, and evidence appendix. Only the three literal
phrases named in #1382 change:

- `ATLAS found ...` becomes `This report found ...`.
- `ATLAS did not receive ...` becomes `This report did not receive ...`.
- `ATLAS mined them ...` becomes `These were mined ...`; the sentence still
  says the report does not claim keyword volume, search rank, or traffic.

The existing producer-vs-docs contract test already regenerates the report
payload and compares it to the JSON fixture. This slice extends that test with
explicit `ATLAS`/`Atlas` absence assertions on the generated Markdown.

## Intentional

- This slice does not rename backend modules, package names, config, logs, or
  internal docs. #1382 is limited to buyer-facing deflection report Markdown.
- The word `Gartner` and the support-cost benchmark copy stay unchanged; this
  PR removes engine branding only, not the estimate model.

## Deferred

None.

Parked hardening: none.

## Verification

- `pytest tests/test_content_ops_faq_report_contract_docs.py -q` - 5 passed.
- `scripts/validate_extracted_content_pipeline.sh` - passed via bash.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` - passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - passed, 0 findings.
- `scripts/check_ascii_python.sh` - passed via bash.
- Pending before push: local PR review.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/frontend/content_ops_faq_deflection_report_example.json` | 2 |
| `extracted_content_pipeline/faq_deflection_report.py` | 10 |
| `plans/PR-Deflection-Report-Brand-Neutral.md` | 92 |
| `tests/test_content_ops_faq_report_contract_docs.py` | 2 |
| **Total** | **106** |
