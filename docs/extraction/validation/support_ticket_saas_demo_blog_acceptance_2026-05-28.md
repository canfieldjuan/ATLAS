# Support-Ticket SaaS Demo Blog Acceptance Retry - 2026-05-28

## Scope

This validation retried the 36-row SaaS demo support-ticket blog path after the
structural `descriptive_no_outcome` contract landed, then retried again after
the deterministic observed-data shell landed.

Source CSV:

`extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv`

The run used Haiku test routing and the existing live generation smoke harness.

## Result

Status: not accepted.

No blog draft was saved, so there is no accepted fixture in this slice.

| Attempt | Status | Blocking result |
|---|---|---|
| 1 | Failed before save | `support_ticket_generated_content` blocked the unsupported outcome sentence: "The next step is to turn those clusters into published, verified, and measurable FAQ entries that help customers find answers." |
| 2 | Failed before save | `geo_entity_clarity_missing` after quality repair budget was exhausted. |
| 3 | Failed before save | Observed-data shell reached the draft body, but the quality gate blocked the repaired candidate on `content_too_short:1111_words_need_1500`. |

## What Changed During Validation

Attempt 1 exposed a small mismatch between the new descriptive contract and the
existing deterministic evaluator. The contract already forbade ticket reduction,
deflection, retention, and faster-resolution claims, but it did not explicitly
forbid the broader unsupported outcome phrasing "help customers find answers."
The evaluator blocks that phrasing for no-outcome support-ticket data, so this
slice added that wording to the structural `forbidden_claims` contract and
covered it with a focused test.

Attempt 2 got past the support-ticket outcome detector but failed the blog GEO
entity clarity gate. The generated title contained "Support Ticket FAQ Gaps,"
but the quality gate still returned `geo_entity_clarity_missing` after two
repair attempts. That is now the next source blocker before this SaaS demo blog
path can be accepted.

Attempt 3 ran after the observed-data shell contract landed. The failed
candidate used the 36 uploaded rows, the 35 question-like rows, and the 9
four-ticket clusters in the body. The title was "Support-ticket questions
customers keep asking" and the SEO title was "Support ticket FAQ gaps: 36
tickets analyzed", so the earlier GEO entity issue did not recur. The remaining
blocking issue is now length-policy alignment: the observed shell produced a
compact 1,111-word candidate, while the blog quality gate still requires 1,500
words.

Manual review of the failed candidate also found the measurement section using
"30, 60, and 90 days" as a future comparison cadence. Because the upload is
undated, the next source-fix slice should decide whether support-ticket
measurement guidance may use future tracking intervals or must stay interval
free for `has_dated_window=false` inputs.

## Acceptance Notes

The observed-data shell reduced the failure from unsupported outcome claims and
GEO repair failure to a specific support-ticket blog shape mismatch: the
truthful observed shell is compact, but the generic blog quality gate still
requires 1,500 words. The 36-row SaaS demo blog path is still not accepted. The
next source-fix slice should either tune the support-ticket blog contract to
safely clear the 1,500-word gate or give this no-outcome support-ticket shape a
more appropriate compact-blog policy.

The repeated `_store_local failed for span=content_ops.llm.complete: column
"account_id" of relation "llm_usage" does not exist` warning also appeared
during both live attempts. That is already parked in `HARDENING.md` as a cost
telemetry schema issue and did not block generation.

Attempt 3 logged `_store_local failed for span=content_ops.llm.complete: pool is
closing`, which is another symptom of the same live LLM usage telemetry path.
It did not block generation or change the acceptance decision.

## Verification

- Command: python scripts/smoke_content_ops_live_generation.py --output blog_post --account-id acct_support_ticket_saas_demo_blog_acceptance_20260528 --support-ticket-csv extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv --env-file /home/juan-canfield/Desktop/Atlas/.env --env-file /home/juan-canfield/Desktop/Atlas/.env.local --env-file tmp/support_ticket_live_haiku_eval_20260525/haiku.env --export-saved-draft tmp/support_ticket_saas_demo_blog_acceptance_20260528/blog-post-draft.json --output-result tmp/support_ticket_saas_demo_blog_acceptance_20260528/blog-post-result.json --evaluate-generated-content --json
  - Failed before save on unsupported "help customers find answers" outcome language.
- Command: python -m pytest tests/test_extracted_blog_generation.py tests/test_smoke_content_ops_live_generation.py -q
  - Passed, 102 tests, after adding the contract forbidden-claim fixture.
- Command: python scripts/smoke_content_ops_live_generation.py --output blog_post --account-id acct_support_ticket_saas_demo_blog_acceptance_20260528_retry2 --support-ticket-csv extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv --env-file /home/juan-canfield/Desktop/Atlas/.env --env-file /home/juan-canfield/Desktop/Atlas/.env.local --env-file tmp/support_ticket_live_haiku_eval_20260525/haiku.env --export-saved-draft tmp/support_ticket_saas_demo_blog_acceptance_20260528_retry2/blog-post-draft.json --output-result tmp/support_ticket_saas_demo_blog_acceptance_20260528_retry2/blog-post-result.json --evaluate-generated-content --json
  - Failed before save on `geo_entity_clarity_missing`.
- Command: python scripts/smoke_content_ops_live_generation.py --output blog_post --account-id acct_support_ticket_observed_shell_live_retry_20260528 --support-ticket-csv extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv --env-file /home/juan-canfield/Desktop/Atlas/.env --env-file /home/juan-canfield/Desktop/Atlas/.env.local --env-file /home/juan-canfield/Desktop/Atlas-support-ticket-provider/tmp/support_ticket_live_haiku_eval_20260525/haiku.env --export-saved-draft tmp/support_ticket_blog_observed_shell_live_retry_20260528/blog-post-draft.json --output-result tmp/support_ticket_blog_observed_shell_live_retry_20260528/blog-post-result.json --evaluate-generated-content --json
  - Failed before save on `content_too_short:1111_words_need_1500`; no saved draft export was produced, so the generated-content evaluator did not run.
- Result artifact: `tmp/support_ticket_blog_observed_shell_live_retry_20260528/blog-post-result.json`
  - `ok=false`, `saved_draft_export=null`, failed candidate `word_count=1111`.
