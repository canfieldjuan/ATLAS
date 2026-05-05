-- Per-account scoping on llm_usage (PR-D3, LLM Gateway MVP).
--
-- Atlas's existing pipeline writes to llm_usage 100k+ times/month
-- without knowing about accounts. The sentinel UUID
-- '00000000-0000-0000-0000-000000000000' marks those rows as
-- "atlas internal" so the pipeline keeps working unmodified --
-- new INSERTs from atlas-side code that omit account_id pick up
-- the DEFAULT and write SENTINEL.
--
-- Customer-facing API endpoints (PR-D4) explicitly thread
-- account_id from require_api_key()/require_auth(), producing
-- per-tenant usage rows. Per-account analytics (e.g., monthly
-- token spend for billing) filter on account_id != SENTINEL or
-- account_id = $customer_id.

ALTER TABLE llm_usage
    ADD COLUMN IF NOT EXISTS account_id UUID NOT NULL
    DEFAULT '00000000-0000-0000-0000-000000000000';

-- Per-account analytics: cost-by-day per account.
CREATE INDEX IF NOT EXISTS idx_llm_usage_account_created
    ON llm_usage (account_id, created_at DESC);
