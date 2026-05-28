# PR: Support-Ticket Blog Observed-Shell Live Telemetry

## Why this slice exists

PR-Support-Ticket-Blog-Observed-Shell changed the support-ticket blog contract
so the no-outcome path receives deterministic observed sections and review-needed
FAQ shells. PR-LLM-Usage-Schema-Cache-Telemetry fixed the local `llm_usage`
schema mismatch that previously hid Content Ops cache telemetry.

The next useful validation slice is to rerun the representative 36-row SaaS demo
blog path with Haiku, then prove the generated draft result and the persisted
usage telemetry line up. This keeps the work source-first: verify the real
support-ticket provider, generation, save/export, generated-content evaluator,
and usage-summary path together before adding more product surface.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-provider
Slice phase: Functional validation

1. Run a live Claude Haiku blog-post smoke with the 36-row SaaS demo
   support-ticket CSV after the observed-shell contract landed.
2. Export the saved draft if generation saves one.
3. Run the deterministic support-ticket generated-content evaluator against the
   saved export.
4. Query the Content Ops `llm_usage` summary for the run account/output to prove
   persisted cache/cost telemetry survives the schema fallback.
5. Record commands, artifact paths, generated-content result, cache-token
   metrics, and any follow-up in a committed validation doc.

### Files touched

- `plans/PR-Support-Ticket-Blog-Observed-Shell-Live-Telemetry.md` - plan doc for this validation slice.
- `docs/extraction/validation/support_ticket_blog_observed_shell_live_telemetry_2026-05-28.md` - live validation and telemetry record.

## Mechanism

Use the existing live smoke harness with the Haiku override and generated-content
evaluation enabled:

```bash
python scripts/smoke_content_ops_live_generation.py \
  --output blog_post \
  --account-id acct_support_ticket_blog_observed_shell_live_telemetry_20260528 \
  --support-ticket-csv extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv \
  --env-file /home/juan-canfield/Desktop/Atlas/.env \
  --env-file /home/juan-canfield/Desktop/Atlas/.env.local \
  --env-file tmp/support_ticket_live_haiku_eval_20260525/haiku.env \
  --export-saved-draft tmp/support_ticket_blog_observed_shell_live_telemetry_20260528/blog-post-draft.json \
  --output-result tmp/support_ticket_blog_observed_shell_live_telemetry_20260528/blog-post-result.json \
  --evaluate-generated-content \
  --json
```

The saved JSON artifacts stay under `tmp/`. The committed doc records the
artifact paths and summarizes the result. Usage proof reads the same database
through `summarize_content_ops_llm_usage`, filtered by account and `blog_post`,
so the doc can confirm whether persisted `cached_tokens`,
`cache_write_tokens`, `billable_input_tokens`, and call counts match the
generation metadata.

## Intentional

- This is validation, not a new generator rewrite, unless the live output still
  exposes a truthfulness blocker.
- The run uses Haiku because it is the cheaper test model and has been the
  stricter support-ticket prompt stress case.
- This does not add FAQ Article output or customer-language keyword promotion;
  those remain future product slices owned outside this validation PR.
- Raw live artifacts are not committed because they contain run-specific ids and
  full generated content. The validation doc commits the reproducible commands
  and the relevant metrics.

## Deferred

- Future PR: broader live acceptance across more representative customer CSV
  shapes after this 36-row SaaS demo path is accepted.
- Future PR: product UI for showing per-run Content Ops cost/cache telemetry.
- Future PR: deterministic renderer if free-form blog generation keeps leaking
  unsupported no-outcome support-ticket claims.
- Parked hardening: none.

## Verification

- Command: python scripts/smoke_content_ops_live_generation.py --output blog_post --account-id acct_support_ticket_blog_observed_shell_live_telemetry_20260528 --support-ticket-csv extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv --env-file /home/juan-canfield/Desktop/Atlas/.env --env-file /home/juan-canfield/Desktop/Atlas/.env.local --env-file tmp/support_ticket_live_haiku_eval_20260525/haiku.env --export-saved-draft tmp/support_ticket_blog_observed_shell_live_telemetry_20260528/blog-post-draft.json --output-result tmp/support_ticket_blog_observed_shell_live_telemetry_20260528/blog-post-result.json --evaluate-generated-content --json
  - Passed; saved blog draft `4792bdf3-5520-40f9-bfb3-79e2112d5624`.
- Command: python scripts/evaluate_support_ticket_generated_content.py --output blog_post tmp/support_ticket_blog_observed_shell_live_telemetry_20260528/blog-post-draft.json --pretty
  - Passed; generated-content evaluator returned `ok: true`.
- Command: Content Ops usage summary query filtered to account
  `acct_support_ticket_blog_observed_shell_live_telemetry_20260528` and
  `asset_type=blog_post`.
  - Passed; summary returned 3 calls, 0 failures, 29,013 input tokens, 9
    billable input tokens, 9,788 cached tokens, 19,216 cache-write tokens, 8,056
    output tokens, and $0.016131 cost.
- Command: `bash scripts/local_pr_review.sh --current-pr-body-file <PR body file>`
  - Passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~95 |
| Validation doc | ~120 |
| **Total** | **~215** |
