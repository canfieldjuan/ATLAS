# PR: Support-Ticket Blog Descriptive Contract

## Why this slice exists

PR-Support-Ticket-SaaS-Demo-Generated-Content-Acceptance proved the 36-row
SaaS demo landing-page path, but deliberately did not accept the broader blog
path. The live blog runs kept drifting from observed support-ticket clusters
into unsupported benefit and outcome claims. The evaluator now catches the
known-bad examples, but that is still downstream defense.

This slice promotes the parked hardening item
`Support-ticket blog generation needs contract-level descriptive mode before
SaaS demo acceptance` and fixes the source contract: when support-ticket data
has no measured outcomes and no resolution evidence, the blog generator should
receive an explicit descriptive/no-outcome mode that tells it what claims are
allowed, what claims are forbidden, and how draft FAQ shells should be framed.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-provider
Slice phase: Production hardening

1. Add a structural support-ticket blog contract for no-outcome/no-resolution
   support-ticket blog contexts.
2. Thread that contract through the live smoke support-ticket blog blueprint
   payload so the SaaS demo shape gets the same source-of-truth instructions
   the generator uses.
3. Make `BlogPostGenerationService` enrich support-ticket blog blueprints with
   the descriptive contract before prompting and repair.
4. Update the blog prompt to consume the contract fields instead of relying
   only on long banned-phrase prose.
5. Add focused deterministic tests for the contract fields and repair prompt
   wiring.
6. Mark the promoted HARDENING item resolved in this branch while leaving the
   cost telemetry item parked.

### Files touched

- `plans/PR-Support-Ticket-Blog-Descriptive-Contract.md` - Plan doc for the contract hardening slice.
- `HARDENING.md` - Remove the resolved descriptive-mode parked item.
- `scripts/smoke_content_ops_live_generation.py` - Include the descriptive contract in support-ticket blog blueprint payloads.
- `extracted_content_pipeline/blog_generation.py` - Add and apply the support-ticket descriptive blog contract before prompt/repair.
- `atlas_brain/skills/digest/blog_post_generation.md` - Prompt consumes the explicit support-ticket descriptive contract fields.
- `extracted_content_pipeline/skills/digest/blog_post_generation.md` - Synced extracted prompt.
- `tests/test_smoke_content_ops_live_generation.py` - Assert the live smoke blueprint carries the contract.
- `tests/test_extracted_blog_generation.py` - Assert generator prompt/repair wiring and descriptive mode behavior.

## Mechanism

The contract is a small data-context block, not another detector layer. For a
support-ticket blog context where both `has_measured_outcomes` and
`support_ticket_resolution_evidence_present` are false or missing, the generator
adds:

- `support_ticket_blog_mode: "descriptive_no_outcome"`
- `allowed_claims`: observed clusters, customer wording, review-needed FAQ
  shells, support-team verification work, and metrics to measure after publish.
- `forbidden_claims`: ticket reduction, churn/retention outcomes, faster
  resolution, prevented tickets, capacity gains, and concrete answer steps
  without resolution evidence.
- `draft_answer_guidance`: a review-needed placeholder sentence for FAQ shells.

`_blog_generation_prompts` already serializes the enriched blueprint into the
prompt, and `_repair_quality_once` rebuilds the same base prompt for repair, so
putting the contract on the blueprint data context makes it load-bearing for
both first generation and repair. The deterministic evaluator remains as a
backstop.

## Intentional

- This does not run a new live Haiku acceptance loop. The next validation slice
  should do that after this contract lands.
- This does not change FAQ Markdown/article ownership. It only improves how
  support-ticket input is consumed by blog generation.
- This keeps the broad outcome detector in place. Narrowing false positives for
  the eventual accepted-blog fixture is deferred unless the new deterministic
  tests require it.
- This keeps the LLM usage schema mismatch parked because it is unrelated to
  making the blog contract truthful.
- Cross-layer caller hints for `BlogPostGenerationService` are expected because
  it is the shared blog generator. This slice covers the relevant support-ticket
  caller with `tests/test_support_ticket_provider_landing_blog_execute.py`.

## Deferred

- Future PR: run one Haiku live retry against
  `support_ticket_saas_demo_sources.csv` and, if it passes, commit an accepted
  SaaS demo blog fixture.
- Future PR: narrow broad outcome-pattern false positives that block legitimate
  neutral measurement language in the accepted descriptive-blog fixture.
- Future PR: add a scripted regression gate once the accepted SaaS demo blog
  fixture exists.
- Parked hardening:
  - LLM usage storage schema mismatch hides per-run cost telemetry.

## Verification

- Command: python -m pytest tests/test_extracted_blog_generation.py tests/test_smoke_content_ops_live_generation.py -q
  - Passed, 102 tests.
- Command: python -m pytest tests/test_support_ticket_provider_landing_blog_execute.py -q
  - Passed, 11 tests.
- Command: bash extracted/_shared/scripts/sync_extracted.sh extracted_content_pipeline
  - Passed.
- Command: bash scripts/validate_extracted_content_pipeline.sh
  - Passed.
- Command: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
  - Passed.
- Command: python scripts/audit_extracted_standalone.py --fail-on-debt
  - Passed.
- Command: bash scripts/check_ascii_python.sh
  - Passed.
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/support-ticket-blog-descriptive-contract-pr-body.md
  - Passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~127 |
| Contract implementation | ~85 |
| Prompt updates | ~4 |
| Tests | ~94 |
| HARDENING cleanup | ~9 |
| **Total** | **~324** |

This stays under the soft cap by avoiding a live generated fixture and focusing
on structural contract wiring.
