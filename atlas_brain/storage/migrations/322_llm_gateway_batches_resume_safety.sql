-- LLM Gateway resume + retry safety (PR-D4e).
--
-- Closes two PR-D4c audit items:
--
-- 1. Duplicate-pay window: previously the resume claim treated
--    any stale ``queued`` row (no provider_batch_id, >60s old) as
--    "crashed pre-submit, safe to resubmit". That conflated two
--    failure modes:
--      a) atlas crashed before reaching the Anthropic call (safe
--         to re-submit -- batch never existed).
--      b) Anthropic accepted the create call BUT the local
--         UPDATE writing provider_batch_id failed (asyncpg
--         transient, container restart, network partition).
--         Re-submitting in this case creates a duplicate paid
--         batch on the customer's account.
--
--    ``anthropic_call_initiated_at`` is set BEFORE the
--    AsyncAnthropic.batches.create call. The resume claim
--    narrows to ``WHERE anthropic_call_initiated_at IS NULL``
--    so we only auto-resume rows that demonstrably never
--    reached Anthropic. Ambiguous rows (initiated_at set, no
--    provider_batch_id) require manual recovery and are logged
--    loudly so ops can find them.
--
-- 2. Retry-on-terminal storm: refresh_customer_batch_status
--    fires _persist_batch_usage on every poll when a terminal
--    row has usage_tracked=FALSE. Under transient SDK failures
--    a 1Hz poll loop hammers Anthropic's results endpoint.
--    ``last_usage_retry_at`` records the timestamp of each
--    retry attempt so the refresh path can enforce a cooldown
--    (currently 30s -- shorter than the 60s resume threshold so
--    customers still see retry happen on a typical poll cadence).

ALTER TABLE llm_gateway_batches
    ADD COLUMN IF NOT EXISTS anthropic_call_initiated_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS last_usage_retry_at TIMESTAMPTZ;
