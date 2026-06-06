# PR: Support-Ticket Descriptive Blog Contract

## Why this slice exists

PR #981 proved the support-ticket landing path can save a grounded draft, but
the blog path still fails closed on no-outcome/no-resolution support-ticket
inputs. The failure is not just model weakness: the long-form blog contract
still pulls the model toward benefit claims and answer steps that uploaded
question-only tickets cannot support.

The prior PR intentionally deferred this follow-up: make support-ticket blogs
with no measured outcomes and no resolution evidence explicitly descriptive so
the system can produce a passing draft without copying guardrail text, while
the deterministic evaluator still blocks unsupported outcomes and invented
procedural answers.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-provider-descriptive-blog-contract
Slice phase: Functional validation

1. Update support-ticket blog prompt guidance so no-outcome/no-resolution
   inputs ask for an observed-pattern article: clusters, customer wording,
   review workflow, placeholder answer shells, publication checklist, and
   measurement plan.
2. Tighten support-ticket generated-content evaluation so descriptive language
   is allowed when it is not promising support-volume, retention, speed, or
   self-service outcomes.
3. Keep existing blockers for unsupported outcome claims and concrete answer
   steps without resolution evidence.
4. Add focused tests for the reviewer-probed false-positive examples and for
   the blog generator saving a descriptive support-ticket draft.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Support-Ticket-Descriptive-Blog-Contract.md` | Plan doc for this slice. |
| `atlas_brain/skills/digest/blog_post_generation.md` | Source prompt guidance for descriptive support-ticket blogs. |
| `extracted_content_pipeline/skills/digest/blog_post_generation.md` | Synced extracted prompt copy. |
| `extracted_content_pipeline/support_ticket_generated_content_eval.py` | Scope false-positive-prone outcome and answer-step patterns. |
| `tests/test_evaluate_support_ticket_generated_content.py` | Fixtures proving descriptive copy passes while unsafe claims still fail. |
| `tests/test_extracted_blog_generation.py` | Service-level proof that a descriptive support-ticket blog draft saves. |

## Mechanism

The prompt gets a support-ticket-specific path for question-only uploads:
describe what the uploaded tickets show, draft review-needed FAQ shells, and
tell the reader what to verify before publishing. It explicitly avoids a broad
benefits section unless measured outcomes exist, and avoids concrete answer
steps unless resolution examples exist.

The evaluator keeps the deterministic fail-closed backstop, but narrows the
patterns that caused descriptive drafts to fail:

- bare phrases like "fewer tickets" or "faster resolution for customers" no
  longer fail by themselves; they fail when framed as a result, promise, or
  publication outcome
- imperative answer-step verbs still fail, but ordinary descriptive phrases
  such as "customers go to billing questions first" do not
- generic third-party capability wording no longer fails as a product-specific
  support answer step; product-specific exported/updated/changed claims still
  fail without resolution evidence

## Intentional

- This is not another live-LLM validation slice. The previous live run already
  proved the failing shape; this slice makes the contract capable of accepting
  a descriptive draft and proves that deterministically.
- This does not take over FAQ generator ownership. It only constrains how blog
  generation may describe FAQ opportunities from uploaded tickets.
- The evaluator still errs toward blocking unsupported customer-facing claims;
  this slice removes false positives only where the sentence is descriptive
  rather than promissory or procedural.

## Deferred

- Parked hardening: none.
- A fresh live Haiku proof that the model now naturally produces a passing
  descriptive blog belongs in the next validation slice after this contract
  change lands.
- Future product slice: promote extracted customer wording from support
  tickets into a shared search-language contract for blog and landing-page
  keywords, then add a first-class FAQ Article output that creates
  help-center-style, search-visible articles from FAQ Report clusters without
  branching into one-off generator patches.
- Broader acceptance testing across many customer CSV shapes remains a later
  robust-testing slice.

## Verification

Completed:

- `python -m pytest tests/test_evaluate_support_ticket_generated_content.py -q`
  - `42 passed`.
- `python -m pytest tests/test_extracted_blog_generation.py::test_generate_saves_descriptive_support_ticket_blog_without_outcome_or_resolution_evidence -q`
  - `1 passed`.
- `python -m pytest tests/test_evaluate_support_ticket_generated_content.py tests/test_extracted_blog_generation.py -q`
  - `99 passed`.
- `bash scripts/local_pr_review.sh --current-pr-body-file <PR body file>`
  - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~105 |
| Prompt guidance | ~10 |
| Evaluator and tests | ~120 |
| **Total** | **~235** |
