"""Cost-closure surface for the extracted LLM-infrastructure package.

Owned modules in this package (not synced from atlas_brain):

- ``cache_savings`` -- persists one row per cache hit with the would-have-
  been input/output tokens and cost, then rolls them up for the
  "saved by cache" hero metric on the cost dashboard.
- ``budget`` -- runtime budget gate. ``BudgetGate.check_before_call``
  returns a ``BudgetDecision`` so the call site can deny LLM calls
  that would breach a configured daily or per-attribution cap.
"""
