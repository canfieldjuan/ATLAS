# Content Ops FAQ Human Action Language

## Why this slice exists

The richer FAQ Markdown renderer shipped article-style answers, but the default
action copy still used vague phrases such as "self-service" and could only say
"contact support" without a phone number, email, or help URL. That weakens the
customer-facing output: the FAQ should lead the user through clear next steps
and use a real support contact only when the host provides one.

## Scope (this PR)

1. Add optional support contact configuration for FAQ
   Markdown generation.
2. Thread the contact through the standalone CLI, generation plan, and Content
   Ops execution dispatch.
3. Replace vague action language with concrete, plain-English steps for login,
   reporting/export, workflow, and fallback issues.
4. Update focused tests for the builder, CLI, plan, and execution seams.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-FAQ-Human-Action-Language.md` | Plan doc for this slice. |
| `docs/extraction/coordination/inflight.md` | Claim this in-flight slice for coordination. |
| `extracted_content_pipeline/ticket_faq_markdown.py` | Add support-contact config and human action copy. |
| `extracted_content_pipeline/generation_plan.py` | Thread FAQ support contact into plan config. |
| `extracted_content_pipeline/content_ops_execution.py` | Pass support contact to the FAQ service. |
| `scripts/build_extracted_ticket_faq_markdown.py` | Add support contact CLI flag. |
| `extracted_content_pipeline/README.md` | Document support-contact usage. |
| `tests/test_extracted_ticket_faq_markdown.py` | Builder/CLI regressions for contact and language. |
| `tests/test_extracted_content_generation_plan.py` | Plan config coverage. |
| `tests/test_extracted_content_ops_execution.py` | Execution wiring coverage. |

## Mechanism

The renderer keeps deterministic, extractive grouping. It now accepts an
optional contact string and passes it through the FAQ item builder, article-step
builder, and escalation-guidance builder. When the contact is present, the final
action step names it directly. When absent, the output still says to contact
support but does not invent a phone number or URL.

The language update removes "self-service" from summaries, next steps, and
escalation guidance. Instead, each FAQ entry gives concrete steps such as
checking account settings, looking for export/download controls, checking role
or plan permissions, and then contacting the configured support channel when
those steps fail.

## Intentional

- No hard-coded support phone number. Hosts supply the CLI support-contact
  option or FAQ support-contact input; otherwise the FAQ says "contact support".
- No LLM generation. The FAQ path stays deterministic and grounded in source
  rows.
- No signature-breaking changes. Existing callers keep working because the new
  argument is optional.

## Deferred

- Platform-specific action templates remain out of scope until a real host
  export or product guide tells us the exact settings/export paths.

## Verification

- Focused pytest for FAQ Markdown, FAQ plan mapping, and FAQ execution wiring:
  45 passed.
- Python compile for touched Python files and tests: passed.
- Packaged support-ticket FAQ CLI with a support contact and required output
  checks: passed.
- Local PR review: passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Content-Ops-FAQ-Human-Action-Language.md` | 70 |
| `docs/extraction/coordination/inflight.md` | 4 |
| `extracted_content_pipeline/ticket_faq_markdown.py` | 90 |
| `extracted_content_pipeline/generation_plan.py` | 4 |
| `extracted_content_pipeline/content_ops_execution.py` | 2 |
| `scripts/build_extracted_ticket_faq_markdown.py` | 8 |
| `extracted_content_pipeline/README.md` | 8 |
| `tests/test_extracted_ticket_faq_markdown.py` | 55 |
| `tests/test_extracted_content_generation_plan.py` | 3 |
| `tests/test_extracted_content_ops_execution.py` | 2 |
| **Total** | **246** |
