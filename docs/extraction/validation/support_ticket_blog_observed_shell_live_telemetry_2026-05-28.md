# Support-Ticket Blog Observed-Shell Live Telemetry - 2026-05-28

## Scope

This validation reran the 36-row SaaS demo support-ticket blog path after the
observed-section and draft-FAQ-shell contract landed.

Source CSV:

`extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv`

The run used the Claude Haiku OpenRouter override and the existing live
generation smoke harness. It also verifies that the `llm_usage` schema fallback
from `PR-LLM-Usage-Schema-Cache-Telemetry` lets Content Ops cost/cache telemetry
persist for the same account.

## Result

Status: accepted.

| Check | Result |
|---|---|
| Live generation | Passed |
| Saved draft id | `4792bdf3-5520-40f9-bfb3-79e2112d5624` |
| Generated-content evaluator | Passed |
| SEO/AEO readiness | `ready` |
| GEO readiness | `ready` |
| Model | `anthropic/claude-haiku-4-5` through OpenRouter |
| Generated content shape | 1,526 words, 5 H2 sections, 3 H3 sections |

The generated draft stayed inside the no-outcome support-ticket contract:

- source context showed 36 uploaded rows, 36 included rows, and 35 question-like
  rows
- all 9 tied ticket clusters were visible
- FAQ shells stayed review-needed where no resolution evidence existed
- generated-content evaluation found no unsupported outcome claims
- generated-content evaluation found no concrete answer steps without
  resolution evidence

## Telemetry Result

Generation metadata in the saved draft:

| Metric | Value |
|---|---:|
| `input_tokens` | 29,013 |
| `billable_input_tokens` | 9 |
| `cached_tokens` | 9,788 |
| `cache_write_tokens` | 19,216 |
| `output_tokens` | 8,056 |

Persisted `llm_usage` summary for account
`acct_support_ticket_blog_observed_shell_live_telemetry_20260528` and asset type
`blog_post`:

| Metric | Value |
|---|---:|
| `total_calls` | 3 |
| `failed_calls` | 0 |
| `total_cost_usd` | 0.016131 |
| `input_tokens` | 29,013 |
| `billable_input_tokens` | 9 |
| `cached_tokens` | 9,788 |
| `cache_write_tokens` | 19,216 |
| `output_tokens` | 8,056 |
| `cache_hit_calls` | 2 |

The persisted totals match the saved draft `generation_usage`, which proves the
cache-token telemetry survived the local `llm_usage` storage path. The cache
status breakdown still reports `cache_mode=no_store` and
`cache_reason=exact_cache_disabled` because that field describes the Content Ops
exact-cache policy, not provider prompt-caching tokens.

Raw `llm_usage` rows were 3 completed Haiku calls. The first wrote 7,646 cache
tokens and had 0 cached tokens; the next two reused 4,894 cached tokens each.
Each row stored the account id and `blog_post` asset type in metadata.

## Commands

```bash
mkdir -p tmp/support_ticket_blog_observed_shell_live_telemetry_20260528
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

Result: passed; saved one draft.

```bash
python scripts/evaluate_support_ticket_generated_content.py \
  --output blog_post \
  tmp/support_ticket_blog_observed_shell_live_telemetry_20260528/blog-post-draft.json \
  --pretty
```

Result: passed.

```bash
python - <<'PY'
from pathlib import Path
from dotenv import load_dotenv
import asyncio

ROOT = Path.cwd()
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / ".env.local", override=True)
load_dotenv(Path("/home/juan-canfield/Desktop/Atlas/.env"), override=True)
load_dotenv(Path("/home/juan-canfield/Desktop/Atlas/.env.local"), override=True)

ACCOUNT = "acct_support_ticket_blog_observed_shell_live_telemetry_20260528"

async def main():
    from atlas_brain.storage.database import close_database, get_db_pool, init_database
    from extracted_content_pipeline.content_ops_usage_summary import summarize_content_ops_llm_usage

    await init_database()
    try:
        summary = await summarize_content_ops_llm_usage(
            get_db_pool(),
            days=1,
            account_id=ACCOUNT,
            asset_type="blog_post",
        )
        print(summary["summary"])
    finally:
        await close_database()

asyncio.run(main())
PY
```

Result: returned the persisted totals shown above.

## Follow-Up

No blocking product follow-up from this validation. The 36-row SaaS demo blog is
accepted for the current no-outcome support-ticket contract.

Next useful slices:

- broaden live validation across more customer CSV shapes
- decide whether to promote this accepted artifact into a minimized regression
  fixture
- per-run Content Ops usage/cost in the product UI is addressed by
  `plans/PR-Content-Ops-Run-Usage-Summary.md`
