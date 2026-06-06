# Support-Ticket Blog Observed Shell

## Why this slice exists

The latest 36-row SaaS demo blog retry still false-greened: generated-content
evaluation passed, but manual review rejected a softer unsupported benefit
claim about future customers recognizing their problem from copied customer
wording. Prior slices tightened prompts and evaluator deny patterns, but the
blog path still lets the model free-write too much connective benefit prose.

This slice moves the no-outcome, no-resolution support-ticket blog contract
toward deterministic slot filling. The model should receive an observed-data
section shell and review-needed FAQ shells derived from the ticket context, so
the article is built from counts, clusters, customer wording, and placeholders
instead of inferred outcomes.

The review-follow-up pushes the diff slightly over the 400 LOC soft cap because
the same source contract must also filter synthetic aggregate buckets and avoid
cross-cluster example fallback before the shell is truthful enough to ship.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-provider

Slice phase: Production hardening

1. Add deterministic observed-section and draft-FAQ-shell fields to the
   support-ticket descriptive blog contract.
2. Thread those fields through the existing seeded support-ticket blog smoke
   blueprint and the blog generation prompt.
3. Update the blog skill guidance to point at the shell fields instead of
   relying only on broad "do not claim" language.
4. Add focused tests proving the shell is generated from representative ticket
   context and reaches the LLM prompt.
5. Skip synthetic `remaining` and `uncategorized` cluster buckets when building
   shells.
6. Avoid assigning unrelated customer wording examples to clusters that have no
   matching sampled example.

### Files touched

- `extracted_content_pipeline/blog_generation.py` - observed shell contract and prompt guidance.
- `atlas_brain/skills/digest/blog_post_generation.md` - canonical blog skill guidance for the synced digest.
- `extracted_content_pipeline/skills/digest/blog_post_generation.md` - support-ticket blog rule update for shell fields.
- `scripts/smoke_content_ops_live_generation.py` - seeded support-ticket blog blueprint threads question and wording fields into the shell.
- `tests/test_atlas_content_ops_infrastructure.py` - host/extracted prompt contract expectation for the shell fields.
- `tests/test_extracted_blog_generation.py` - contract and prompt coverage.
- `tests/test_smoke_content_ops_live_generation.py` - seeded smoke blueprint coverage.
- `plans/PR-Support-Ticket-Blog-Observed-Shell.md` - this plan.

## Mechanism

The support-ticket descriptive blog contract already activates only when the
data context is support-ticket-backed and lacks measured outcomes and resolution
evidence. This slice extends that contract with two deterministic fields:

- `required_section_outline`: exact reader-facing sections and allowed source
  fields for the descriptive article.
- `draft_faq_shells`: cluster/question shells whose answer body is the existing
  review-needed placeholder.

The prompt addendum then tells the model to use those shells as the article
structure. No detector pattern is added; the fix is at the data contract layer.

## Intentional

- No live LLM retry is included in this slice. The behavior is demonstrated by
  deterministic contract and prompt tests; live acceptance belongs to the next
  validation slice after this source contract lands.
- Existing generated-content detectors stay in place as backstops.
- The shell uses observed cluster counts and customer wording only; it does not
  infer product capabilities, resolutions, search behavior, or business impact.
- Synthetic `remaining` and `uncategorized` buckets remain available in the
  source cluster summary, but are intentionally not draft FAQ topics.
- The local cross-layer caller hint for `_string_list` is a same-name helper in
  podcast modules, not a non-diff caller of this new blog helper.

## Deferred

- Live 36-row SaaS demo blog retry after this contract lands.
- Full deterministic renderer that bypasses free-form blog body generation.
- Parked hardening considered but left parked: `LLM usage storage schema
  mismatch hides per-run cost telemetry`; it affects cost visibility, not this
  support-ticket truthfulness shell.

## Verification

- Command: python -m py_compile extracted_content_pipeline/blog_generation.py scripts/smoke_content_ops_live_generation.py tests/test_atlas_content_ops_infrastructure.py tests/test_extracted_blog_generation.py tests/test_smoke_content_ops_live_generation.py
  - Passed.
- Command: python -m py_compile extracted_content_pipeline/blog_generation.py tests/test_extracted_blog_generation.py
  - Passed after review fixes.
- Command: python -m pytest tests/test_extracted_blog_generation.py tests/test_smoke_content_ops_live_generation.py -q
  - Passed, 104 tests.
- Command: python -m pytest tests/test_atlas_content_ops_infrastructure.py::test_blog_generation_prompt_trims_small_support_ticket_uploads tests/test_extracted_blog_generation.py tests/test_smoke_content_ops_live_generation.py -q
  - Passed, 106 tests after rebasing onto latest `origin/main`.
- Command: python -m pytest tests/test_extracted_blog_generation.py::test_support_ticket_draft_shells_skip_aggregate_buckets_and_unrelated_examples tests/test_extracted_blog_generation.py::test_support_ticket_descriptive_blog_contract_requires_no_outcome_or_resolution_evidence -q
  - Passed, 2 tests.
- Command: python -m pytest tests/test_extracted_blog_generation.py tests/test_smoke_content_ops_live_generation.py tests/test_atlas_content_ops_infrastructure.py::test_blog_generation_prompt_trims_small_support_ticket_uploads -q
  - Passed, 107 tests after review fixes.
- Command: bash scripts/validate_extracted_content_pipeline.sh
  - Passed after syncing the canonical blog skill file from `atlas_brain`.
- Command: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
  - Passed.
- Command: python scripts/audit_extracted_standalone.py --fail-on-debt
  - Passed.
- Command: bash scripts/check_ascii_python.sh
  - Passed.
- Command: bash extracted/_shared/scripts/sync_extracted.sh extracted_content_pipeline
  - Passed.
- Command: bash scripts/run_extracted_pipeline_checks.sh
  - First run failed on the stale host/extracted prompt contract test; after updating the test for `required_section_outline` and `draft_faq_shells`, rerun passed with 2,611 passed, 8 skipped. Post-rebase rerun passed with 2,612 passed, 8 skipped. Post-review-fix rerun passed with 2,613 passed, 8 skipped.
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/support-ticket-blog-observed-shell-pr-body.md
  - Passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Blog contract and prompt | ~120 |
| Smoke blueprint threading | ~5 |
| Skill guidance | ~20 |
| Review fixes | ~90 |
| Tests | ~155 |
| Plan doc | ~95 |
| Total | ~500 |
