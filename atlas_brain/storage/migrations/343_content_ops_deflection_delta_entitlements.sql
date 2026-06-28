-- Stripe Billing-backed monthly delta entitlement records.
-- The delta automation resolver grants only active/trialing subscription rows.

CREATE TABLE IF NOT EXISTS content_ops_deflection_delta_entitlements (
    account_id TEXT NOT NULL,
    stripe_subscription_id TEXT NOT NULL,
    stripe_customer_id TEXT,
    stripe_price_id TEXT,
    stripe_subscription_status TEXT NOT NULL,
    entitlement_source TEXT NOT NULL DEFAULT 'stripe_billing',
    current_period_end TIMESTAMPTZ,
    granted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    revoked_at TIMESTAMPTZ,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (account_id, stripe_subscription_id),
    UNIQUE (stripe_subscription_id),
    CHECK (account_id <> ''),
    CHECK (stripe_subscription_id <> ''),
    CHECK (
        stripe_subscription_status IN (
            'active',
            'trialing',
            'past_due',
            'unpaid',
            'canceled',
            'incomplete',
            'incomplete_expired',
            'paused'
        )
    )
);

CREATE INDEX IF NOT EXISTS idx_content_ops_deflection_delta_entitlements_active
    ON content_ops_deflection_delta_entitlements (account_id, updated_at DESC)
    WHERE stripe_subscription_status IN ('active', 'trialing')
      AND revoked_at IS NULL;
