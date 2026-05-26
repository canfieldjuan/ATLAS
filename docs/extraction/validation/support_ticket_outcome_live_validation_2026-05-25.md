# Support-Ticket Outcome Live Validation - 2026-05-25

## Scope

This validation reran Content Ops support-ticket landing-page and blog-post
generation with the Claude Haiku OpenRouter override, then ran a targeted
Claude Sonnet comparison after review. The goal was to prove the
`support_ticket_outcome_claims_grounded` evaluator check from #970 against fresh
live output, not only saved historical artifacts, and to determine whether the
remaining failures were model-specific or contract-shaped.

Environment:

- Repo: `/home/juan-canfield/Desktop/Atlas-support-ticket-provider`
- Env files:
  - `/home/juan-canfield/Desktop/Atlas/.env`
  - `/home/juan-canfield/Desktop/Atlas/.env.local`
  - `tmp/support_ticket_live_haiku_eval_20260525/haiku.env`
- Artifact directory:
  - `tmp/support_ticket_outcome_live_validation_20260525`

The `haiku.env` override pins the live generation path to the Claude Haiku
family so this proof does not use Sonnet.

## Commands

Landing page smoke:

```bash
python scripts/smoke_content_ops_live_generation.py \
  --account-id acct_support_ticket_outcome_live_validation_20260525_landing \
  --support-ticket-csv \
  --env-file /home/juan-canfield/Desktop/Atlas/.env \
  --env-file /home/juan-canfield/Desktop/Atlas/.env.local \
  --env-file tmp/support_ticket_live_haiku_eval_20260525/haiku.env \
  --export-saved-draft tmp/support_ticket_outcome_live_validation_20260525/landing-page-draft.json \
  --output-result tmp/support_ticket_outcome_live_validation_20260525/landing-page-result.json \
  --evaluate-generated-content \
  --json
```

Blog smoke iterations used the same env files with `--output blog_post` and
account ids `acct_support_ticket_outcome_live_validation_20260525_blog` through
`acct_support_ticket_outcome_live_validation_20260525_blog10`.

Sonnet comparison used the same command shape with:

- account id: `acct_support_ticket_outcome_sonnet_eval_20260525_blog1`
- override model: `anthropic/claude-sonnet-4-5`
- output artifact:
  `tmp/support_ticket_outcome_sonnet_eval_20260525/blog-post-result-1.json`

Regression suite:

```bash
python -m pytest tests/test_evaluate_support_ticket_generated_content.py -q
```

## Results

Landing page:

- Result: passed.
- Saved draft id: `030c08da-3f01-41f6-afa3-474d600bfa6d`.
- `seo_aeo_readiness`: ready.
- `geo_readiness`: ready.
- Generated-content evaluation: passed, including
  `support_ticket_outcome_claims_grounded`.

Blog post:

- Initial live blog drafts produced false greens under the #970 detector.
- Manual review found unsupported variants including:
  - reduced support volume / customers find answers without opening tickets
  - customers are more likely to stay
  - fewer tickets on that topic
  - support-team capacity gains
  - resolve/find answers faster
  - direct churn/retention improvement guarantees
  - instant/immediate-result language
  - fabricated time/capacity claims such as "2-3 minutes" and "thousands of hours"
- Each false green was converted into evaluator coverage and prompt guidance.
- Stale saved drafts now fail when rerun through the current evaluator. For
  example, draft `66fc02f9-82cf-400c-ad31-b5491ee3647f` now fails for
  unsupported instant-result, support-capacity, support-volume, and "without
  opening a support ticket" claims while neutral churn/retention context and
  disclaimers are allowed.

The final state is intentionally conservative: the gate is stricter than the
first live run, and the latest known saved false green is no longer accepted by
the evaluator.

Sonnet comparison:

- Result: no saved draft.
- Blockers:
  - `content_too_short:1426_words_need_1500`
  - `geo_entity_clarity_missing`
  - `support_ticket_generated_content` flagged quoted negative examples from
    prompt guidance, such as "Avoid claims like..."
- Interpretation: Sonnet produced more explicit cautionary language than the
  Haiku runs, but the run still failed under the same structural pressure: the
  task asks for a persuasive FAQ/blog asset from question/count/cluster data
  that lacks outcomes, dates, and resolutions. The model filled the missing
  space with either conventional benefit language or copied prompt guidance.
  This supports treating model choice as mitigation, not the root fix.

## Residual Finding

Live generation also showed a broader source-truth issue: when uploaded tickets
only include customer questions/descriptions and no support-resolution fields,
the model can invent concrete product steps for FAQ answers.

That issue is parked in `ATLAS-HARDENING.md` as:

- `Support-ticket FAQ drafts can invent procedural answer steps when tickets lack resolutions`
- Owner/session: `content-ops/support-ticket-outcome-live-validation`

It is not fixed in this slice because the current load-bearing slice is the
outcome/time/capacity/support-volume claim gate. The follow-up should add a first-class
"resolution evidence present" contract and make the generator emit review-needed
answer placeholders when resolution evidence is absent.

## Verification

- `python -m pytest tests/test_evaluate_support_ticket_generated_content.py -q`
  - Expected: all tests pass.
- Rerun evaluator on known stale false-green drafts:
  - Expected: `support_ticket_outcome_claims_grounded` fails with the unsupported
    claim sentences listed in the JSON output.
