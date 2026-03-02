-- Stripe billing event audit log
CREATE TABLE IF NOT EXISTS billing_events (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id      UUID REFERENCES saas_accounts(id) ON DELETE SET NULL,
    stripe_event_id VARCHAR(128) UNIQUE NOT NULL,
    event_type      VARCHAR(128) NOT NULL,
    payload         JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_billing_events_account ON billing_events(account_id);
CREATE INDEX IF NOT EXISTS idx_billing_events_type ON billing_events(event_type);
