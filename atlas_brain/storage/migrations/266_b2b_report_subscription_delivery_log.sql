CREATE TABLE IF NOT EXISTS b2b_report_subscription_delivery_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subscription_id UUID NOT NULL REFERENCES b2b_report_subscriptions(id) ON DELETE CASCADE,
    account_id UUID NOT NULL REFERENCES saas_accounts(id) ON DELETE CASCADE,
    scheduled_for TIMESTAMPTZ NOT NULL,
    scope_type TEXT NOT NULL
        CHECK (scope_type IN ('library', 'report')),
    scope_key TEXT NOT NULL,
    delivered_report_ids UUID[] NOT NULL DEFAULT '{}',
    recipient_emails TEXT[] NOT NULL DEFAULT '{}',
    delivery_frequency TEXT NOT NULL
        CHECK (delivery_frequency IN ('weekly', 'monthly', 'quarterly')),
    deliverable_focus TEXT NOT NULL
        CHECK (deliverable_focus IN ('all', 'battle_cards', 'executive_reports', 'comparison_packs')),
    freshness_policy TEXT NOT NULL
        CHECK (freshness_policy IN ('fresh_only', 'fresh_or_monitor', 'any')),
    freshness_state TEXT NOT NULL DEFAULT 'none'
        CHECK (freshness_state IN ('fresh', 'monitor', 'stale', 'unknown', 'mixed', 'none')),
    status TEXT NOT NULL DEFAULT 'processing'
        CHECK (status IN ('processing', 'sent', 'partial', 'skipped', 'failed')),
    message_ids TEXT[] NOT NULL DEFAULT '{}',
    summary TEXT NOT NULL DEFAULT '',
    error TEXT,
    delivered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (subscription_id, scheduled_for)
);

CREATE INDEX IF NOT EXISTS idx_b2b_report_subscription_delivery_log_subscription
    ON b2b_report_subscription_delivery_log (subscription_id, delivered_at DESC);

CREATE INDEX IF NOT EXISTS idx_b2b_report_subscription_delivery_log_account
    ON b2b_report_subscription_delivery_log (account_id, delivered_at DESC);

CREATE INDEX IF NOT EXISTS idx_b2b_report_subscription_delivery_log_status
    ON b2b_report_subscription_delivery_log (status, delivered_at DESC);
