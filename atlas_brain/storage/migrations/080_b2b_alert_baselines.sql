-- 080: B2B alert baselines, tenant-scoped reports, onboarding sequences

-- Churn signal spike alert baselines per tracked vendor per account
CREATE TABLE IF NOT EXISTS b2b_alert_baselines (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id      UUID NOT NULL REFERENCES saas_accounts(id) ON DELETE CASCADE,
    vendor_name     TEXT NOT NULL,
    metric          TEXT NOT NULL,  -- signal_count, avg_urgency, displacement_count
    baseline_value  NUMERIC NOT NULL DEFAULT 0,
    last_alerted_at TIMESTAMPTZ,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(account_id, vendor_name, metric)
);

-- Tenant-scoped intelligence reports
ALTER TABLE b2b_intelligence ADD COLUMN IF NOT EXISTS account_id UUID REFERENCES saas_accounts(id);
CREATE INDEX IF NOT EXISTS idx_b2b_intelligence_account ON b2b_intelligence(account_id);

-- Sequence type discriminator for onboarding vs outreach sequences
ALTER TABLE campaign_sequences ADD COLUMN IF NOT EXISTS sequence_type VARCHAR(32) NOT NULL DEFAULT 'outreach';
