-- Migration 101: Promote buyer authority profiles to a first-class UPSERT
-- table with per-vendor aggregation.
--
-- Buyer authority data was previously only stored as JSONB inside
-- b2b_churn_signals.  This table enables direct queries like "which
-- vendors have the most economic buyers leaving?" without scanning JSONB.

BEGIN;

CREATE TABLE IF NOT EXISTS b2b_vendor_buyer_profiles (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vendor_name      TEXT NOT NULL,
    role_type        TEXT NOT NULL CHECK (role_type IN (
        'economic_buyer', 'champion', 'evaluator', 'end_user', 'unknown'
    )),
    buying_stage     TEXT NOT NULL CHECK (buying_stage IN (
        'active_purchase', 'evaluation', 'renewal_decision', 'post_purchase', 'unknown'
    )),
    review_count     INT NOT NULL DEFAULT 0,
    dm_count         INT NOT NULL DEFAULT 0,
    avg_urgency      NUMERIC(3,1),
    source_distribution  JSONB DEFAULT '{}',
    sample_review_ids    UUID[],
    first_seen_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (vendor_name, role_type, buying_stage)
);

CREATE INDEX IF NOT EXISTS idx_bvbp_vendor
    ON b2b_vendor_buyer_profiles (vendor_name);
CREATE INDEX IF NOT EXISTS idx_bvbp_role_type
    ON b2b_vendor_buyer_profiles (role_type);
CREATE INDEX IF NOT EXISTS idx_bvbp_review_count
    ON b2b_vendor_buyer_profiles (review_count DESC);

COMMIT;
