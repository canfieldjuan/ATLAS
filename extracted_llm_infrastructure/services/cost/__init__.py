"""Cost-closure surface for the extracted LLM-infrastructure package.

Owned modules in this package (not synced from atlas_brain):

- ``cache_savings`` -- persists one row per cache hit with the would-have-
  been input/output tokens and cost, then rolls them up for the
  "saved by cache" hero metric on the cost dashboard.
- ``openai_billing`` -- OpenAI Costs API fetcher. Sibling of the
  lifted ``services/provider_cost_sync.py`` (which covers OpenRouter
  + Anthropic). Persists daily costs to the same
  ``llm_provider_daily_costs`` table.

Future siblings (queued in ``docs/extraction/cost_closure_audit_2026-05-03.md``):

- ``drift`` (PR-A4): local-vs-invoiced reconciliation per (provider,
  model, day, attribution).
- ``budget`` (PR-A4): runtime budget gate with daily and per-attribution
  caps.
"""
