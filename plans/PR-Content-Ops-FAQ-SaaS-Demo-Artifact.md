# PR-Content-Ops-FAQ-SaaS-Demo-Artifact

## Why this slice exists

PR-Content-Ops-FAQ-SaaS-Demo-Corpus added a labeled synthetic B2B SaaS support
ticket corpus and proved the real FAQ generator can process it, but the repo
still lacks a checked-in generated FAQ report that another session can inspect
or hand to a landing-page demo without rerunning the generator.

This slice closes that handoff gap with a static Markdown artifact generated
from the synthetic corpus through the existing FAQ producer. It stays small
because the corpus already exists and the generated output is deterministic.

## Scope (this PR)

Ownership lane: content-ops/faq-generator
Slice phase: Functional validation

1. Add a checked-in Markdown FAQ report generated from the synthetic B2B SaaS
   support-ticket corpus.
2. Extend the corpus test to regenerate the Markdown through the real producer
   and compare it to the checked artifact.
3. Keep runtime APIs, generator behavior, search projection, and frontend code
   unchanged.

### Files touched

| File | Purpose |
|---|---|
| `plans/PR-Content-Ops-FAQ-SaaS-Demo-Artifact.md` | Plan contract for this artifact handoff slice. |
| `extracted_content_pipeline/examples/support_ticket_saas_demo_faq.md` | Static generated FAQ report from the synthetic B2B SaaS corpus. |
| `tests/test_content_ops_faq_saas_demo_corpus.py` | Regeneration guard proving the static artifact matches the real producer output. |

## Mechanism

The Markdown artifact is generated with the existing CLI-compatible producer
settings used by the prior corpus proof:

```bash
python scripts/build_extracted_ticket_faq_markdown.py \
  extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv \
  --title "Synthetic B2B SaaS Support FAQ Demo" \
  --max-items 12 \
  --max-evidence-per-item 3 \
  --support-contact https://example.com/support \
  --require-output-checks \
  --output extracted_content_pipeline/examples/support_ticket_saas_demo_faq.md
```

The test reads the CSV, normalizes it with `source_rows_to_campaign_opportunities`,
runs `build_ticket_faq_markdown`, and asserts the committed Markdown file is an
exact match. The existing leakage guard still scans generated Markdown for the
consumer-finance terms that should not appear in this SaaS demo lane.

## Intentional

- The artifact is synthetic and labeled in the title; this is not represented
  as design-partner or anonymized customer data.
- The generated answer steps remain draft-review placeholders because the
  synthetic corpus has no resolution evidence. That is data-truthful for the
  current FAQ generator contract.
- No result JSON is checked in; the Markdown is the artifact needed by the
  landing-page/demo handoff, and the test already covers producer fidelity.

## Deferred

- Future PR: seed this corpus and generated FAQ into the hosted FAQ search demo
  route once the demo session is ready to consume it.
- Future PR: replace or supplement the synthetic artifact with an anonymized
  real SaaS export when one exists.
- Parked hardening: none.

## Verification

- python -m pytest tests/test_content_ops_faq_saas_demo_corpus.py -q - 3 passed.
- python scripts/audit_extracted_pipeline_ci_enrollment.py . - passed.
- python scripts/audit_plan_code_consistency.py plans/PR-Content-Ops-FAQ-SaaS-Demo-Artifact.md - passed.
- git diff --check - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/examples/support_ticket_saas_demo_faq.md` | 143 |
| `plans/PR-Content-Ops-FAQ-SaaS-Demo-Artifact.md` | 87 |
| `tests/test_content_ops_faq_saas_demo_corpus.py` | 37 |
| **Total** | **267** |
