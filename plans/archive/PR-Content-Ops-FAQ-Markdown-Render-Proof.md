# PR-Content-Ops-FAQ-Markdown-Render-Proof

## Why this slice exists

The FAQ Markdown output is now wired into Content Ops generation, storage,
export, review, and the offline execution smoke. The remaining proof gap is
whether the generated Markdown is actually renderable as a readable FAQ from
real packaged support-ticket rows. Discovery found the current bold labels and
bullets render as paragraphs rather than HTML lists because the Markdown lacks
blank lines before the bullet blocks. This slice fixes that at the source and
locks the rendered structure.

## Scope (this PR)

1. Fix FAQ Markdown spacing so action and source bullets render as lists.
2. Add a render-proof regression test for FAQ Markdown generated from the
   packaged support-ticket CSV.
3. Render the generated Markdown through the already declared Python Markdown
   dependency and assert the customer-visible structure survives rendering.
4. Leave FAQ generation content and grouping behavior unchanged.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-FAQ-Markdown-Render-Proof.md` | Plan doc for this slice. |
| `docs/extraction/coordination/inflight.md` | Claim this in-flight slice. |
| `.github/workflows/extracted_pipeline_checks.yml` | Install the declared Markdown renderer in extracted CI. |
| `extracted_content_pipeline/ticket_faq_markdown.py` | Add Markdown-compatible blank lines before FAQ bullet lists. |
| `tests/test_extracted_ticket_faq_markdown.py` | Add render-proof coverage for generated FAQ Markdown. |

## Mechanism

The production renderer now emits a blank line after each bold list label
before the bullets. That keeps the visible Markdown shape the same while
allowing standard Markdown renderers to emit real unordered lists.

The extracted CI workflow installs the same Markdown package already declared
in `requirements.txt` so the render proof runs in CI instead of passing only in
developer environments that happen to have the package installed.

The test loads the existing support-ticket CSV fixture through the same source
adapter used by the FAQ CLI, builds the FAQ Markdown with the production
builder, renders it with the declared Markdown package, and inspects the HTML
with the standard-library HTML parser.

The assertions cover the three output checks:

1. User vocabulary survives in rendered headings and body text.
2. Similar rows are condensed into the generated FAQ items rather than
   duplicated beyond the builder result.
3. The rendered FAQ includes a clear "What to do next" action section with
   bullet items, plus source links as rendered list items.

## Intentional

- No model call is added. The deterministic builder should cover the first
  useful version of a grounded FAQ; LLM polishing can be a later opt-in layer.
- No production HTML renderer is added. Hosts can render Markdown however they
  prefer; this slice verifies the generated Markdown is compatible with the
  declared Markdown renderer.
- The test inspects rendered HTML structure, not exact HTML formatting, because
  renderer whitespace and tag formatting are implementation details.

## Deferred

- A visual/browser rendering check can be added if a hosted FAQ page UI is
  introduced.
- An optional LLM rewrite or semantic clustering pass can be evaluated after
  deterministic FAQ output is validated against real customer data.

## Verification

- pytest tests/test_extracted_ticket_faq_markdown.py - 15 passed.
- python -m py_compile extracted_content_pipeline/ticket_faq_markdown.py tests/test_extracted_ticket_faq_markdown.py - passed.
- bash scripts/validate_extracted_content_pipeline.sh - passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline - passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt - passed.
- bash scripts/check_ascii_python.sh - passed.
- git diff --check - passed.
- bash scripts/run_extracted_pipeline_checks.sh - 1494 passed, 1 existing torch/pynvml warning.
- bash scripts/local_pr_review.sh - passed after CI dependency amendment.

## Estimated diff size

| File | Estimated LOC |
|---|---:|
| `plans/PR-Content-Ops-FAQ-Markdown-Render-Proof.md` | +93 |
| `docs/extraction/coordination/inflight.md` | +1 / -1 |
| `.github/workflows/extracted_pipeline_checks.yml` | +1 / -1 |
| `extracted_content_pipeline/ticket_faq_markdown.py` | +2 |
| `tests/test_extracted_ticket_faq_markdown.py` | +83 |
| Total | ~184 |
