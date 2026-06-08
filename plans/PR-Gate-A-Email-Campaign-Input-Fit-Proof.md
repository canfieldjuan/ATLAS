# PR-Gate-A-Email-Campaign-Input-Fit-Proof

## Why this slice exists

Issue #1376 asked for Gate A `email_campaign` coverage in the live-quality
harness. #1378 shipped the build plumbing: `email_campaign` review/export,
sequence-aware persistence checks, and the `--outputs` selector. #1383 and
#1392 then intentionally excluded `email_campaign` so the already-converged
`landing_page`, `blog_post`, and `sales_brief` paths could be judged without
the campaign input-fit question masking their signal.

The remaining issue #1376 follow-up is the live proof that the support-ticket
Gate A payload can produce, review, and export an email-campaign sequence. The
first live attempt exposed two harness input-fit failures:

1. The harness imported campaign opportunities before the support-ticket
   provider/blog setup finalized the request filters, so the campaign service
   could read with a different filter than the import used.
2. The campaign prompt had no real selling URL, so the model naturally used the
   fixture's `example.com` source URLs and the existing placeholder URL guard
   correctly skipped every draft.

Review on #1394 then exposed a correctness failure in the first committed
sample: the model fabricated source-volume/count claims such as "40+ queues"
and unsupported percentages. This update keeps the slice in the same lane but
closes that proof gap inline because a fabricated proof artifact cannot satisfy
Gate A. The fix adds an upstream sparse-support-ticket grounding path for the
exact one-ticket fixture and expands fail-closed validation for unsupported
numeric, scan, aggregate, timing, recurrence, and queue/support-traffic claims.

This PR keeps the slice narrow: fix harness input fit, make sparse one-ticket
campaign output grounded by construction, commit the live proof artifacts, and
report the generated samples for reviewer judgment.

This PR exceeds the 400 LOC soft cap because the raw JSON validation artifacts
are part of the deliverable and the review fix needs focused negative fixtures
for the fail-closed grounding detector branches.

## Scope (this PR)

Ownership lane: content-ops/gate-a-output-quality
Slice phase: Functional validation

1. Run `scripts/smoke_content_ops_gate_a_live_quality.py` against the real
   local database/model route with `--outputs email_campaign`.
2. Prepare output dependencies in execution order: support-ticket provider
   merge, optional blog seed/alignment only when `blog_post` is selected, then
   `email_campaign` opportunity import using the final payload filters.
3. Add a real `selling.affiliate_url` and sender context to the Gate A payload
   so the campaign prompt has a valid CTA URL while the placeholder URL guard
   remains strict.
4. Commit the exact run artifacts under
   `docs/extraction/validation/fixtures/` plus a markdown report with the
   command, resolved model, structural result, sequence shape, and sample
   pointers.
5. Keep this as validation. Do not self-certify product acceptance; the
   reviewer owns the GOOD-bar judgment against the exported campaign drafts.
6. Prevent sparse one-ticket `email_campaign` fabrication by using a
   deterministic scaffold for exactly-one `support_ticket` evidence rows and
   by failing closed on unsupported LLM proof/volume claims when the normal
   campaign path is used.

### Review Contract

- Acceptance criteria:
  - [ ] The committed command includes `--outputs email_campaign`.
  - [ ] The artifact set includes `execution-result.json`,
        `opportunity-import.json`, `review-results.json`, `summary.json`, and
        `export-email_campaign.json`.
  - [ ] The report records the resolved model route and that local Ollama
        fallback was disabled.
  - [ ] The report surfaces real email-campaign sample ids, subjects, CTAs, and
        sequence counts without claiming product pass/fail beyond the harness
        result.
  - [ ] Empty, missing, duplicate, collapsed, or quality-skipped sequence
        behavior is reported explicitly rather than hidden behind `ok=true`.
  - [ ] The run excludes `landing_page`, `blog_post`, and `sales_brief` so
        campaign input fit is not diluted by the already-proven generators.
- Affected surfaces: Gate A live-quality harness, focused harness tests,
  validation artifacts, and support-ticket live proof documentation.
- Risk areas: accidental local model fallback, over-claiming campaign quality,
  mistaking a non-empty sequence for useful copy, weakening placeholder guards,
  and importing campaign opportunities under filters different from execution.
- Reviewer rules triggered: R1, R10.

### Files touched

- `scripts/smoke_content_ops_gate_a_live_quality.py`
- `atlas_brain/skills/digest/b2b_campaign_generation.md`
- `extracted_content_pipeline/campaign_generation.py`
- `extracted_content_pipeline/services/campaign_quality.py`
- `extracted_content_pipeline/skills/digest/b2b_campaign_generation.md`
- `tests/test_extracted_campaign_generation.py`
- `tests/test_extracted_campaign_generation_seams.py`
- `tests/test_extracted_campaign_skill_registry.py`
- `tests/test_smoke_content_ops_gate_a_live_quality.py`
- `docs/extraction/validation/content_ops_gate_a_email_campaign_input_fit_2026-06-08.md`
- `docs/extraction/validation/fixtures/content_ops_gate_a_email_campaign_input_fit_20260608/execution-result.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_email_campaign_input_fit_20260608/export-email_campaign.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_email_campaign_input_fit_20260608/opportunity-import.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_email_campaign_input_fit_20260608/review-results.json`
- `docs/extraction/validation/fixtures/content_ops_gate_a_email_campaign_input_fit_20260608/summary.json`
- `plans/PR-Gate-A-Email-Campaign-Input-Fit-Proof.md`

## Mechanism

The harness now prepares dependencies with a small testable helper:

```python
payload = await _payload_with_support_ticket_provider(payload, scope=scope)
if "blog_post" in selected_outputs:
    seeded_blog_blueprint = await _seed_default_blog_blueprint(args, scope)
    _align_blog_payload_to_seed(payload, seeded_blog_blueprint)
if "email_campaign" in selected_outputs:
    opportunity_import = await seed_email_campaign_opportunities(
        filters=_payload_filters(payload),
        ...
    )
```

That sequence guarantees the `campaign_opportunities` import uses the same
filters the executor will pass into `CampaignGenerationService.generate(...)`.
For email-only runs, the support-ticket campaign filter remains
`support_ticket_faq_gap_live_gate_a`. For combined blog/email runs, the import
would follow the blog-aligned filters instead of importing stale rows.

The Gate A payload also includes a real selling context:
`selling.affiliate_url = https://finetunelab.ai/systems/ai-content-ops/intake`.
This gives the campaign prompt a valid CTA URL while preserving the existing
`example.com` placeholder URL rejection.

For sparse support-ticket evidence, `CampaignGenerationService` now detects the
exactly-one `support_ticket` shape before calling the LLM. That path renders a
small deterministic email sequence from only:

1. the exact uploaded ticket question,
2. the allowed inference that the question points to a possible FAQ gap, and
3. the claim that FineTune Lab can map this question against the help center to
   identify whether an answer is missing or hard to find.

The deterministic drafts still pass through the same
`campaign_quality_revalidation(...)` gate, metadata builder, review, and export
path as model-generated drafts. Multi-row or non-support-ticket campaign
opportunities still use the normal LLM path, now with the same sales-brief
truthfulness language plus stricter sparse-ticket prompt guidance.

The quality seam now fails closed on unsupported numeric/count/percentage
claims, scan/review claims, recurrence claims, unsupported turnaround/timing
claims, and product-general queue/support-traffic claims. The harness also
requires an `email_campaign` sequence to export both `email_cold` and
`email_followup`; one saved row no longer yields `ok=true`.

The proof reuses the #1378 harness path: execute the real Content Ops service
builder, persist generated campaign drafts, review saved ids, export the
reviewed sequence through `list_campaign_drafts(...)`, filter to the run's
saved ids, and write the same artifact envelope used by #1383/#1392.

## Intentional

- Use the clean support-ticket SaaS demo fixture first. The three-generator
  messy rerun is already complete, and email-campaign input-fit had not yet
  been proven on the baseline payload.
- Do not include `landing_page`, `blog_post`, or `sales_brief`; this slice
  isolates the issue #1376 email-campaign follow-up.
- Do not weaken campaign quality gates or placeholder URL detection. The fix
  supplies real campaign context instead.
- Do not add a quality-repair LLM pass for this slice. The sparse one-ticket
  path is deterministic to avoid billing a second generation and to prevent
  fabricated claims from entering the proof artifact.
- Keep the deterministic scaffold scoped to exactly-one `support_ticket`
  evidence rows. Broader email campaigns still use the existing LLM generator
  and fail closed if unsupported claims appear.
- Keep the proof account and generated JSON artifacts committed so reviewer
  and operator can inspect the exact live output.
- Do not close issue #1376 in this PR body. The reviewer/operator can decide
  whether the committed proof fully resolves the issue.

## Deferred

- Messy-input and multi-ticket `email_campaign` validation remain separate
  until the baseline support-ticket input-fit proof is reviewed.
- Cheaper-model readiness remains separate after Sonnet behavior is reviewed.
- Richer sparse-ticket copy is separate. This slice chooses a conservative
  grounded scaffold over marketing polish.

Parked hardening:

- None. The input-fit failures, fabricated proof claims, and collapsed-sequence
  false-green all blocked this slice's stated real flow and are fixed inline.

## Verification

- `python -m pytest tests/test_extracted_campaign_generation_seams.py tests/test_extracted_campaign_generation.py tests/test_extracted_campaign_skill_registry.py tests/test_smoke_content_ops_gate_a_live_quality.py -q`
  - Result: `115 passed, 1 warning in 1.19s`.
- `EXTRACTED_CAMPAIGN_LLM_AUTO_ACTIVATE_OLLAMA=false python scripts/smoke_content_ops_gate_a_live_quality.py --account-id 5b2f2a9c-6d1e-4f2c-9a87-31e64d42a901 --user-id 11111111-1111-4111-8111-111111111111 --support-ticket-csv extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv --env-file /home/juan-canfield/Desktop/Atlas/.env --env-file /home/juan-canfield/Desktop/Atlas/.env.local --output-dir tmp/content_ops_gate_a_email_campaign_input_fit_20260608 --outputs email_campaign --variant-count 3 --quality-repair-attempts 1 --max-cost-usd 20.00 --json`
  - Result: `status=passed`, `inserted=36`, `generated=2`, `saved_ids=2`,
    `export_counts.email_campaign=2`, `llm_model=deterministic/single-support-ticket`.
- Artifact JSON validation over
  `docs/extraction/validation/fixtures/content_ops_gate_a_email_campaign_input_fit_20260608/*.json`
  - Result: all committed artifact JSON files parsed successfully.
- Generation-route/generated-row check over `export-email_campaign.json`
  - Result: `rows=2`, `models=['deterministic/single-support-ticket']`,
    `channels=['email_cold', 'email_followup']`, `parse_attempts=[0]`.
- `bash scripts/run_extracted_pipeline_checks.sh`
  - Result: `3433 passed, 10 skipped, 1 warning in 51.31s`.
- `scripts/push_pr.sh` will run the repo's required local PR checks once
  before push.

## Estimated diff size

| Area | Diff |
|---|---:|
| Code/tests | ~1,150 LOC |
| Validation report/artifacts | ~780 LOC |
| **Total** | **1,957 LOC (16 files, +1923 / -34)** |
