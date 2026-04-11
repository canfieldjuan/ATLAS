# Atlas LLM Consumer Map

> Last updated: 2026-04-11
>
> **WARNING**: GPU is not detected (NVRM: No NVIDIA GPU found). All local
> inference (vLLM, Ollama) is non-functional. Tasks designed for free local
> inference are falling back to paid APIs. See docs/GPU_STATUS.md.
> Full power cycle of the Brain PC is required to recover.

## Active API Providers

| Provider | Models | Billing | Status |
|---|---|---|---|
| **Anthropic Direct** | claude-3-5-haiku-latest | Anthropic API key | Active |
| **Anthropic Batch** | claude-haiku-4-5, claude-sonnet-4-5 (50% discount) | Anthropic API key | Active |
| **OpenRouter** | anthropic/claude-haiku-4-5, anthropic/claude-sonnet-4-5 | OpenRouter API key | Active |
| **vLLM (local)** | Qwen3-30B-A3B-AWQ | Free | **DOWN - GPU not detected** |
| **Ollama (local)** | qwen3:14b, qwen3:32b, phi3:mini | Free | **DOWN - GPU not detected** |

## .env Model Overrides

```
ATLAS_LLM_OPENROUTER_REASONING_MODEL=anthropic/claude-sonnet-4-5
ATLAS_LLM_OPENROUTER_REASONING_STRICT=true
ATLAS_B2B_CHURN_ENRICHMENT_OPENROUTER_MODEL=anthropic/claude-haiku-4-5
ATLAS_B2B_CHURN_ENRICHMENT_REPAIR_MODEL=anthropic/claude-haiku-4-5
ATLAS_B2B_CHURN_BRIEFING_ANALYST_MODEL=anthropic/claude-sonnet-4-5
ATLAS_B2B_CHURN_BLOG_POST_OPENROUTER_MODEL=anthropic/claude-sonnet-4-5
ATLAS_B2B_CHURN_PRODUCT_PROFILE_OPENROUTER_MODEL=anthropic/claude-sonnet-4-5
```

## B2B Churn Pipeline Tasks

| Task | Enabled | Interval | Model | API | Batch? | Cost |
|---|---|---|---|---|---|---|
| b2b_enrichment | YES | 5min | claude-haiku-4-5 | OpenRouter + Anthropic Batch | YES | High |
| b2b_enrichment_repair | YES | 15min | claude-haiku-4-5 | OpenRouter + Anthropic Batch | YES | High |
| b2b_churn_reports | YES | daily 21:30 | claude-sonnet-4-5 | OpenRouter + Anthropic Batch | YES | High |
| b2b_battle_cards | YES | daily 21:30 | claude-sonnet-4-5 (inherits reasoning) | OpenRouter + Anthropic Batch | YES | High |
| b2b_product_profiles | YES | daily 21:30 | claude-sonnet-4-5 | OpenRouter + Anthropic Batch | YES | High |
| b2b_tenant_report | YES | weekly | claude-sonnet-4-5 | OpenRouter + Anthropic Batch | YES | High |
| b2b_reasoning_synthesis | YES | daily 21:15 | claude-sonnet-4-5 | OpenRouter + Anthropic Batch | YES | High |
| b2b_challenger_brief | YES | daily 21:40 | None (SQL aggregation only) | N/A | NO | None |
| b2b_accounts_in_motion | YES | daily 21:35 | None (SQL aggregation only) | N/A | NO | None |
| b2b_churn_intelligence | YES | daily 21:00 | None (deterministic pools) | N/A | NO | None |
| b2b_churn_alert | YES | hourly | None | N/A | NO | None |
| b2b_scrape_intake | YES | hourly | None (web scraping) | N/A | NO | None |
| b2b_account_resolution | YES | 10min | None | N/A | NO | None |
| b2b_watchlist_alert_delivery | YES | hourly | None | N/A | NO | None |

## Disabled Tasks (cost reduction 2026-04-11)

| Task | Was | Reason Disabled |
|---|---|---|
| b2b_blog_post_generation | 35 runs/wk, 3-6min each | Sonnet-4-5 heavy, not needed for MVP |
| b2b_campaign_generation | daily | Haiku batch, not needed for MVP |
| email_auto_approve | 4,594 runs/wk | Haiku triage every 2min, major cost drain |
| email_intake | 849 runs/wk | Haiku triage every 10min |
| news_intake | 553 runs/wk | Every 15min |
| market_intake | 1,766 runs/wk | Every 5min |

## Win/Loss Predictor (on-demand)

| Endpoint | Model | API | Cache |
|---|---|---|---|
| POST /win-loss | claude-haiku-4-5 | OpenRouter | Exact cache (win_loss.strategy) |
| POST /win-loss/compare | claude-haiku-4-5 (x2 parallel) | OpenRouter | Exact cache |

## Reasoning Agent

| Component | Model | API | Notes |
|---|---|---|---|
| reasoning/agent.py | claude-haiku-4-5 | OpenRouter | .env override |
| reasoning/reflection.py | claude-3-5-haiku-latest | Anthropic Direct (triage) | |
| reasoning/falsification.py | Qwen3-30B-A3B-AWQ / claude-sonnet-4-5 | vLLM / OpenRouter | Inactive (vLLM down) |

## LLM Router Singletons

| Singleton | Model | API | Purpose |
|---|---|---|---|
| get_triage_llm() | claude-3-5-haiku-latest | Anthropic Direct | Cheap classification |
| get_draft_llm() | claude-sonnet-4-5 | Anthropic Direct | Email drafts |
| get_reasoning_llm() | claude-3-5-haiku-latest | Anthropic Direct | Deep reasoning fallback |

## Local Inference (currently inactive)

| Runtime | Model | Used By |
|---|---|---|
| vLLM | Qwen3-30B-A3B-AWQ | Atlas Agent, Home Agent, b2b_churn_intelligence |
| Ollama (day) | qwen3:14b | Email graph sync, intent router fallback |
| Ollama (night) | qwen3:32b | Graphiti nightly sync |
| llama.cpp (Pi) | Phi-3-mini-4k Q4 | Edge node offline fallback |

## Cost Tiers

- **High**: Tasks using claude-sonnet-4-5 (reasoning synthesis, reports, battle cards, blog gen)
- **Medium**: Tasks using claude-haiku-4-5 (enrichment, repair, win/loss)
- **Low**: Tasks using triage haiku for classification only (email, intent routing)
- **None**: SQL aggregation, scraping, alerting tasks with no LLM calls
