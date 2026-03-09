-- Migration 100: Promote vendor pain points, use cases, and integrations
-- to first-class UPSERT tables with provenance tracking.
--
-- These entities were previously only stored as JSONB arrays inside
-- b2b_churn_signals and b2b_product_profiles.  New tables enable direct
-- SQL queries like "which vendors have the worst pricing pain?" without
-- scanning JSONB blobs.

BEGIN;

-- -----------------------------------------------------------------------
-- 1. b2b_vendor_pain_points
-- -----------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS b2b_vendor_pain_points (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vendor_name      TEXT NOT NULL,
    pain_category    TEXT NOT NULL CHECK (pain_category IN (
        'pricing', 'support', 'features', 'ux',
        'reliability', 'performance', 'integration', 'security',
        'onboarding', 'other'
    )),
    mention_count    INT NOT NULL DEFAULT 0,
    primary_count    INT NOT NULL DEFAULT 0,
    secondary_count  INT NOT NULL DEFAULT 0,
    minor_count      INT NOT NULL DEFAULT 0,
    avg_urgency      NUMERIC(3,1),
    avg_rating       NUMERIC(3,2),
    source_distribution  JSONB DEFAULT '{}',
    sample_review_ids    UUID[],
    confidence_score     NUMERIC(3,2) DEFAULT 0,
    first_seen_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (vendor_name, pain_category)
);

CREATE INDEX IF NOT EXISTS idx_bvpp_vendor
    ON b2b_vendor_pain_points (vendor_name);
CREATE INDEX IF NOT EXISTS idx_bvpp_category
    ON b2b_vendor_pain_points (pain_category);
CREATE INDEX IF NOT EXISTS idx_bvpp_mentions
    ON b2b_vendor_pain_points (mention_count DESC);
CREATE INDEX IF NOT EXISTS idx_bvpp_confidence
    ON b2b_vendor_pain_points (confidence_score DESC);

-- -----------------------------------------------------------------------
-- 2. b2b_vendor_use_cases
-- -----------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS b2b_vendor_use_cases (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vendor_name      TEXT NOT NULL,
    use_case_name    TEXT NOT NULL,
    mention_count    INT NOT NULL DEFAULT 0,
    avg_urgency      NUMERIC(3,1),
    lock_in_distribution JSONB DEFAULT '{}',
    source_distribution  JSONB DEFAULT '{}',
    sample_review_ids    UUID[],
    first_seen_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (vendor_name, use_case_name)
);

CREATE INDEX IF NOT EXISTS idx_bvuc_vendor
    ON b2b_vendor_use_cases (vendor_name);
CREATE INDEX IF NOT EXISTS idx_bvuc_use_case
    ON b2b_vendor_use_cases (use_case_name);
CREATE INDEX IF NOT EXISTS idx_bvuc_mentions
    ON b2b_vendor_use_cases (mention_count DESC);

-- -----------------------------------------------------------------------
-- 3. b2b_vendor_integrations
-- -----------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS b2b_vendor_integrations (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vendor_name      TEXT NOT NULL,
    integration_name TEXT NOT NULL,
    mention_count    INT NOT NULL DEFAULT 0,
    source_distribution  JSONB DEFAULT '{}',
    sample_review_ids    UUID[],
    first_seen_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (vendor_name, integration_name)
);

CREATE INDEX IF NOT EXISTS idx_bvi_vendor
    ON b2b_vendor_integrations (vendor_name);
CREATE INDEX IF NOT EXISTS idx_bvi_integration
    ON b2b_vendor_integrations (integration_name);
CREATE INDEX IF NOT EXISTS idx_bvi_mentions
    ON b2b_vendor_integrations (mention_count DESC);

COMMIT;
