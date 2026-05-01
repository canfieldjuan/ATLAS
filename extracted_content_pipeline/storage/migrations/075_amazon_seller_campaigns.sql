-- Migration 075: Amazon Seller campaign outreach support
--
-- Extends the campaign pipeline to target Amazon sellers with consumer
-- review intelligence (feature gaps, competitive flows, safety signals).
-- Reuses b2b_campaigns + campaign_sequences for send/sequence/audit infra.

-- 1. Allow 'amazon_seller' as a target_mode on b2b_campaigns
--    (column added in 074, default 'churning_company', no CHECK constraint)

-- 2. Seller targets: Amazon seller contacts to outreach with category intelligence
CREATE TABLE IF NOT EXISTS seller_targets (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    seller_name     VARCHAR(255),
    company_name    VARCHAR(255),
    email           VARCHAR(255),
    seller_type     VARCHAR(30) NOT NULL DEFAULT 'private_label'
        CHECK (seller_type IN ('private_label', 'manufacturer', 'agency', 'wholesale_reseller')),
    categories      TEXT[] NOT NULL DEFAULT '{}',
    storefront_url  TEXT,
    notes           TEXT,
    status          VARCHAR(20) NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'paused', 'unsubscribed', 'bounced')),
    source          VARCHAR(50) DEFAULT 'manual',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_seller_targets_status ON seller_targets(status);
CREATE INDEX IF NOT EXISTS idx_seller_targets_type ON seller_targets(seller_type);
CREATE INDEX IF NOT EXISTS idx_seller_targets_email ON seller_targets(LOWER(email));
CREATE INDEX IF NOT EXISTS idx_seller_targets_categories ON seller_targets USING GIN(categories);

-- 3. Category intelligence snapshots: cached aggregation for campaign generation
--    Prevents re-computing expensive queries on every campaign run.
CREATE TABLE IF NOT EXISTS category_intelligence_snapshots (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    category            TEXT NOT NULL,
    snapshot_date       DATE NOT NULL DEFAULT CURRENT_DATE,

    -- Aggregated intelligence
    total_reviews       INT NOT NULL DEFAULT 0,
    total_brands        INT NOT NULL DEFAULT 0,
    total_products      INT NOT NULL DEFAULT 0,
    top_pain_points     JSONB NOT NULL DEFAULT '[]',
    feature_gaps        JSONB NOT NULL DEFAULT '[]',
    competitive_flows   JSONB NOT NULL DEFAULT '[]',
    brand_health        JSONB NOT NULL DEFAULT '[]',
    safety_signals      JSONB NOT NULL DEFAULT '[]',
    manufacturing_insights JSONB NOT NULL DEFAULT '[]',
    top_root_causes     JSONB NOT NULL DEFAULT '[]',

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_cat_intel_snapshot_unique
    ON category_intelligence_snapshots (category, snapshot_date);
CREATE INDEX IF NOT EXISTS idx_cat_intel_snapshot_date
    ON category_intelligence_snapshots (snapshot_date DESC);
