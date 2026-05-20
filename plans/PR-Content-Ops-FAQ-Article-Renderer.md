Plan: PR-Content-Ops-FAQ-Article-Renderer

## Why this slice exists

PR #667 proved the FAQ output checks, but the generated Markdown still reads
like a thin evidence summary. A sellable FAQ article should give a user a clear
answer and concrete next steps while staying grounded in the source tickets.

This slice upgrades the deterministic FAQ renderer at the source: richer
article sections, numbered steps, support escalation guidance, and cited ticket
evidence. It does not add an LLM or invent product-specific UI instructions.

## Scope (this PR)

1. Extend FAQ item payloads with summary, steps, escalation guidance, and
   evidence quote fields.
2. Render those fields as richer Markdown sections.
3. Update tests for the richer Markdown shape and grounding.
4. Update docs/status to describe the article-style output.
5. Claim the slice in the coordination ledger.

### Files touched

- `plans/PR-Content-Ops-FAQ-Article-Renderer.md`
- `docs/extraction/coordination/inflight.md`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/STATUS.md`
- `extracted_content_pipeline/ticket_faq_markdown.py`
- `tests/test_extracted_ticket_faq_markdown.py`

## Mechanism

The existing grouping/ranking/dedup logic stays intact. `_item(...)` will add
derived article fields from the already-selected evidence rows:

- `summary`: concise answer grounded in ticket volume and snippets.
- `steps`: numbered next steps using the existing action-rule policy.
- `when_to_contact_support`: generic escalation guidance.
- `evidence_quotes`: compact ticket quotes with source ids/titles.

`_render(...)` then emits those fields as Markdown. The old `answer` and
`action_items` keys remain populated for compatibility.

## Intentional

- No provider call. The FAQ remains deterministic, fast, and auditable.
- No product-specific click paths are invented. Steps stay generic unless
  source text/action rules support the category.
- Existing public function signatures stay unchanged.

## Deferred

- LLM-assisted FAQ rewriting remains separate from this deterministic renderer.
- Help-center publishing UI remains separate from the Markdown generation path.
- Real customer help desk exports remain the trigger for more source-specific
  answer policies.

## Verification

- pytest tests/test_extracted_ticket_faq_markdown.py - 42 passed
- python -m py_compile extracted_content_pipeline/ticket_faq_markdown.py tests/test_extracted_ticket_faq_markdown.py - passed
- python scripts/build_extracted_ticket_faq_markdown.py extracted_content_pipeline/examples/support_ticket_sources.csv --source-format csv --require-output-checks --output /tmp/support_ticket_faq_article.md - passed
- git diff --check - passed
- bash scripts/run_extracted_pipeline_checks.sh - 1530 passed, 1 existing torch/pynvml warning
- bash scripts/local_pr_review.sh - passed

## Estimated diff size

| File | Estimated LOC |
|---|---:|
| `plans/PR-Content-Ops-FAQ-Article-Renderer.md` | +65 |
| `docs/extraction/coordination/inflight.md` | +2 / -1 |
| `extracted_content_pipeline/README.md` | +5 / -2 |
| `extracted_content_pipeline/STATUS.md` | +5 / -2 |
| `extracted_content_pipeline/ticket_faq_markdown.py` | +65 / -10 |
| `tests/test_extracted_ticket_faq_markdown.py` | +45 / -15 |
| **Total** | **217** |

This is below the 400 LOC review budget.
