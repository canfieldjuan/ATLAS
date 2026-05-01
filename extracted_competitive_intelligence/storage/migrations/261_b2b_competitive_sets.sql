-- Migration 261: Competitive-set scoped synthesis control plane

CREATE TABLE IF NOT EXISTS b2b_competitive_sets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id UUID NOT NULL REFERENCES saas_accounts(id) ON DELETE CASCADE,
    name VARCHAR(200) NOT NULL,
    focal_vendor_name TEXT NOT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    refresh_mode VARCHAR(16) NOT NULL DEFAULT 'manual',
    refresh_interval_hours INT,
    vendor_synthesis_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    pairwise_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    category_council_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    asymmetry_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    last_run_at TIMESTAMPTZ,
    last_success_at TIMESTAMPTZ,
    last_run_status VARCHAR(16),
    last_run_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_b2b_competitive_sets_refresh_mode
        CHECK (refresh_mode IN ('manual', 'scheduled')),
    CONSTRAINT chk_b2b_competitive_sets_last_run_status
        CHECK (last_run_status IS NULL OR last_run_status IN ('running', 'succeeded', 'partial', 'failed')),
    CONSTRAINT chk_b2b_competitive_sets_interval
        CHECK (
            (refresh_mode = 'manual' AND refresh_interval_hours IS NULL)
            OR (refresh_mode = 'scheduled' AND refresh_interval_hours IS NOT NULL AND refresh_interval_hours BETWEEN 1 AND 720)
        )
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_b2b_competitive_sets_account_name
    ON b2b_competitive_sets (account_id, LOWER(name));
CREATE INDEX IF NOT EXISTS idx_b2b_competitive_sets_account_active
    ON b2b_competitive_sets (account_id, active, refresh_mode);
CREATE INDEX IF NOT EXISTS idx_b2b_competitive_sets_due
    ON b2b_competitive_sets (active, refresh_mode, last_success_at, created_at);

CREATE TABLE IF NOT EXISTS b2b_competitive_set_vendors (
    competitive_set_id UUID NOT NULL REFERENCES b2b_competitive_sets(id) ON DELETE CASCADE,
    vendor_name TEXT NOT NULL,
    sort_order SMALLINT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (competitive_set_id, vendor_name)
);

CREATE INDEX IF NOT EXISTS idx_b2b_competitive_set_vendors_set
    ON b2b_competitive_set_vendors (competitive_set_id, sort_order, vendor_name);
