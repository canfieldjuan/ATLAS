"""Cost-closure surface for the extracted LLM-infrastructure package.

Owned modules in this package (not synced from atlas_brain):

- ``cache_savings`` -- persists one row per cache hit with the would-have-
  been input/output tokens and cost, then rolls them up for the
  "saved by cache" hero metric on the cost dashboard.
- ``openai_billing`` -- OpenAI Costs API fetcher. Sibling of the
  lifted ``services/provider_cost_sync.py`` (which covers OpenRouter
  + Anthropic). Persists daily costs to the same
  ``llm_provider_daily_costs`` table.
- ``drift`` -- per-day reconciliation between locally-tracked spend
  (``llm_usage``) and invoiced spend (``llm_provider_daily_costs``).
  Returns ``DriftRow`` objects with explanatory chips
  (``stale_pricing``, ``missing_local_rows``, ``high_drift``, etc.).
  The differentiated cost-closure wedge.
- ``budget`` -- runtime budget gate. ``BudgetGate.check_before_call``
  returns a ``BudgetDecision`` so the call site can deny LLM calls
  that would breach a configured daily or per-attribution cap.
"""
