-- Migration 107: Webhook outbound delivery infrastructure
--
-- Adds per-tenant webhook subscriptions and delivery logging.
-- Tenants subscribe to event types (change_event, churn_alert, report_generated,
-- signal_update) and Atlas POSTs signed payloads to their URLs.

-- Subscription table: one row per (account, url) pair
CREATE TABLE IF NOT EXISTS b2b_webhook_subscriptions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id  UUID NOT NULL REFERENCES saas_accounts(id) ON DELETE CASCADE,
    url         TEXT NOT NULL,
    secret      TEXT NOT NULL,          -- HMAC-SHA256 signing key
    event_types TEXT[] NOT NULL DEFAULT '{}',
    -- Allowed values: 'change_event', 'churn_alert', 'report_generated', 'signal_update'
    enabled     BOOLEAN NOT NULL DEFAULT true,
    description TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_webhook_account_url UNIQUE (account_id, url)
);

-- Delivery log: append-only, one row per delivery attempt
CREATE TABLE IF NOT EXISTS b2b_webhook_delivery_log (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subscription_id UUID NOT NULL REFERENCES b2b_webhook_subscriptions(id) ON DELETE CASCADE,
    event_type      TEXT NOT NULL,
    payload         JSONB NOT NULL,
    status_code     INT,
    response_body   TEXT,
    duration_ms     INT,
    attempt         INT NOT NULL DEFAULT 1,
    success         BOOLEAN NOT NULL DEFAULT false,
    error           TEXT,
    delivered_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_webhook_subs_account
    ON b2b_webhook_subscriptions (account_id) WHERE enabled = true;

CREATE INDEX IF NOT EXISTS idx_webhook_delivery_sub
    ON b2b_webhook_delivery_log (subscription_id, delivered_at DESC);

CREATE INDEX IF NOT EXISTS idx_webhook_delivery_event
    ON b2b_webhook_delivery_log (event_type, delivered_at DESC);

-- Constraint on event_types array values (enforced at application level,
-- not via CHECK because Postgres CHECK can't iterate array elements easily).
-- Valid values: 'change_event', 'churn_alert', 'report_generated', 'signal_update'
