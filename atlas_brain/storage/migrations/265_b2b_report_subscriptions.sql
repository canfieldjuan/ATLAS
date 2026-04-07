CREATE TABLE IF NOT EXISTS b2b_report_subscriptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id UUID NOT NULL REFERENCES saas_accounts(id) ON DELETE CASCADE,
    report_id UUID REFERENCES b2b_intelligence(id) ON DELETE CASCADE,
    scope_type TEXT NOT NULL
        CHECK (scope_type IN ('library', 'report')),
    scope_key TEXT NOT NULL,
    scope_label TEXT NOT NULL,
    delivery_frequency TEXT NOT NULL DEFAULT 'weekly'
        CHECK (delivery_frequency IN ('weekly', 'monthly', 'quarterly')),
    deliverable_focus TEXT NOT NULL DEFAULT 'all'
        CHECK (deliverable_focus IN ('all', 'battle_cards', 'executive_reports', 'comparison_packs')),
    freshness_policy TEXT NOT NULL DEFAULT 'fresh_or_monitor'
        CHECK (freshness_policy IN ('fresh_only', 'fresh_or_monitor', 'any')),
    recipient_emails TEXT[] NOT NULL DEFAULT '{}',
    delivery_note TEXT NOT NULL DEFAULT '',
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    next_delivery_at TIMESTAMPTZ,
    created_by UUID REFERENCES saas_users(id) ON DELETE SET NULL,
    updated_by UUID REFERENCES saas_users(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (account_id, scope_type, scope_key)
);

ALTER TABLE b2b_report_subscriptions
    DROP CONSTRAINT IF EXISTS b2b_report_subscriptions_scope_match;

ALTER TABLE b2b_report_subscriptions
    ADD CONSTRAINT b2b_report_subscriptions_scope_match
    CHECK (
        (scope_type = 'library' AND report_id IS NULL AND scope_key = 'library')
        OR (scope_type = 'report' AND report_id IS NOT NULL)
    );

CREATE INDEX IF NOT EXISTS idx_b2b_report_subscriptions_account
    ON b2b_report_subscriptions (account_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_b2b_report_subscriptions_next_delivery
    ON b2b_report_subscriptions (account_id, enabled, next_delivery_at)
    WHERE enabled = TRUE;

CREATE INDEX IF NOT EXISTS idx_b2b_report_subscriptions_report
    ON b2b_report_subscriptions (report_id)
    WHERE report_id IS NOT NULL;
