-- Storage for provider billing reconciliation.
--
-- OpenRouter exposes account-level cumulative usage via /api/v1/credits, so we
-- store snapshots and derive daily deltas at query time.
-- Anthropic exposes daily cost reports directly via the Admin API, so we store
-- normalized daily rows separately.

CREATE TABLE IF NOT EXISTS llm_provider_usage_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    provider TEXT NOT NULL,
    snapshot_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    total_usage_usd NUMERIC(12, 6) NOT NULL,
    total_credits_usd NUMERIC(12, 6),
    raw_payload JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_llm_provider_usage_snapshots_provider_time
    ON llm_provider_usage_snapshots (provider, snapshot_at DESC);

CREATE TABLE IF NOT EXISTS llm_provider_daily_costs (
    provider TEXT NOT NULL,
    cost_date DATE NOT NULL,
    provider_cost_usd NUMERIC(12, 6) NOT NULL,
    currency TEXT NOT NULL DEFAULT 'USD',
    source_kind TEXT NOT NULL,
    raw_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    imported_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (provider, cost_date, source_kind)
);

CREATE INDEX IF NOT EXISTS idx_llm_provider_daily_costs_provider_date
    ON llm_provider_daily_costs (provider, cost_date DESC);
