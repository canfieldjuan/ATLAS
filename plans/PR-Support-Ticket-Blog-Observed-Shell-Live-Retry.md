# Support-Ticket Blog Observed Shell Live Retry

## Why this slice exists

PR #1086 added a deterministic observed-data shell for no-outcome,
no-resolution support-ticket blog generation, but intentionally deferred the
live 36-row SaaS demo retry. The product lane still needs evidence that the
real hosted generation path can now produce a source-truthful SaaS demo blog
instead of only passing deterministic prompt/unit tests.

This slice runs that real flow and records the outcome. If the draft passes the
support-ticket generated-content evaluator and manual truthfulness scan, the
slice promotes the saved draft to the current SaaS demo blog fixture. If it
fails, the slice records the concrete blocker without accepting a fixture.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-provider

Slice phase: Functional validation

1. Run live Haiku-routed blog generation through
   `scripts/smoke_content_ops_live_generation.py` using the 36-row SaaS support
   ticket CSV.
2. Evaluate the saved draft with the deterministic support-ticket
   generated-content evaluator.
3. Manually inspect the generated body for unsupported outcomes, invented
   dates/windows, and concrete answer steps without resolution evidence.
4. Update the SaaS demo blog acceptance note with the observed result.
5. Promote a minimal current SaaS demo blog fixture only if the generated draft
   passes both deterministic and manual acceptance.

### Files touched

- `docs/extraction/validation/support_ticket_saas_demo_blog_acceptance_2026-05-28.md` - append the observed-shell retry result.
- `HARDENING.md` - park the LLM usage telemetry issue referenced by the plan.
- `plans/PR-Support-Ticket-Blog-Observed-Shell-Live-Retry.md` - this plan.

## Mechanism

The validation command exercises the existing live smoke harness:

```bash
python scripts/smoke_content_ops_live_generation.py \
  --output blog_post \
  --account-id acct_support_ticket_observed_shell_live_retry_20260528 \
  --support-ticket-csv extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv \
  --env-file /home/juan-canfield/Desktop/Atlas/.env \
  --env-file /home/juan-canfield/Desktop/Atlas/.env.local \
  --env-file /home/juan-canfield/Desktop/Atlas-support-ticket-provider/tmp/support_ticket_live_haiku_eval_20260525/haiku.env \
  --export-saved-draft tmp/support_ticket_blog_observed_shell_live_retry_20260528/blog-post-draft.json \
  --output-result tmp/support_ticket_blog_observed_shell_live_retry_20260528/blog-post-result.json \
  --evaluate-generated-content \
  --json
```

The smoke writes both the host execution result and the exact saved draft export.
The evaluator runs on the saved draft export, so the recorded acceptance is tied
to persisted generated content rather than a prompt-only fixture.

## Intentional

- No prompt or evaluator changes are planned in this slice. If the live retry
  exposes another source-truthfulness blocker, the blocker is recorded and a
  follow-up source-fix slice should handle it.
- The root `HARDENING.md` cost-telemetry schema issue remains parked. It can
  hide per-run cost logging, but it does not affect generated-copy acceptance.
- The older `ATLAS-HARDENING.md` deep-dive blog items are unrelated to this
  support-ticket provider flow and remain parked.
- The fixture is promoted only on a passing generated draft. A failed live retry
  should not create an accepted fixture.

## Deferred

- The live retry surfaced `content_too_short:1111_words_need_1500` after the
  observed-data shell produced a compact failed candidate. That quality-policy
  source fix is deferred to the next product slice because this slice's job is
  to record the acceptance result honestly.
- The failed candidate also used "30, 60, and 90 days" in measurement guidance
  despite the undated upload. The next source-fix slice should decide whether
  future tracking intervals are acceptable or must be structurally forbidden for
  `has_dated_window=false` support-ticket inputs.
- Full deterministic rendering that bypasses free-form blog body generation
  remains deferred from PR #1086.
- Parked hardening considered but left parked: `LLM usage storage schema
  mismatch hides per-run cost telemetry`; it affects observability, not the
  acceptance decision in this slice. Review follow-up added the missing
  `HARDENING.md` entry for this item.

## Verification

- Command: python scripts/smoke_content_ops_live_generation.py --output blog_post --account-id acct_support_ticket_observed_shell_live_retry_20260528 --support-ticket-csv extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv --env-file /home/juan-canfield/Desktop/Atlas/.env --env-file /home/juan-canfield/Desktop/Atlas/.env.local --env-file /home/juan-canfield/Desktop/Atlas-support-ticket-provider/tmp/support_ticket_live_haiku_eval_20260525/haiku.env --export-saved-draft tmp/support_ticket_blog_observed_shell_live_retry_20260528/blog-post-draft.json --output-result tmp/support_ticket_blog_observed_shell_live_retry_20260528/blog-post-result.json --evaluate-generated-content --json
  - Failed before save on `content_too_short:1111_words_need_1500`.
  - No saved draft export was produced, so the deterministic generated-content
    evaluator did not run.
  - Result artifact:
    `tmp/support_ticket_blog_observed_shell_live_retry_20260528/blog-post-result.json`
    with `ok=false`, `saved_draft_export=null`, and failed candidate
    `word_count=1111`.
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/support-ticket-blog-observed-shell-live-retry-pr-body.md
  - Passed.
- Review follow-up: added the missing `HARDENING.md` telemetry entry referenced
  by this plan.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Validation note | ~40 |
| Hardening entry | ~10 |
| Accepted fixture | 0 |
| Plan doc | ~95 |
| Total | ~155 |
