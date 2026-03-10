-- Migration 115: Product profile snapshots (Phase 3 Sprint 5)
--
-- Append-only daily snapshots of product profiles, mirroring the
-- b2b_vendor_snapshots pattern for vendor health data.

CREATE TABLE IF NOT EXISTS b2b_product_profile_snapshots (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vendor_name             TEXT NOT NULL,
    snapshot_date           DATE NOT NULL DEFAULT CURRENT_DATE,
    total_reviews_analyzed  INTEGER NOT NULL DEFAULT 0,
    avg_rating              NUMERIC(3,2),
    recommend_rate          NUMERIC(3,2),
    avg_urgency             NUMERIC(3,1),
    strength_count          INTEGER NOT NULL DEFAULT 0,
    weakness_count          INTEGER NOT NULL DEFAULT 0,
    top_strength            TEXT,
    top_weakness            TEXT,
    top_use_case            TEXT,
    top_integration         TEXT,
    compared_to_count       INTEGER NOT NULL DEFAULT 0,
    switched_from_count     INTEGER NOT NULL DEFAULT 0,
    pain_categories_covered INTEGER NOT NULL DEFAULT 0,
    profile_summary_len     INTEGER NOT NULL DEFAULT 0,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (vendor_name, snapshot_date)
);

CREATE INDEX IF NOT EXISTS idx_bpps_vendor_date
    ON b2b_product_profile_snapshots (vendor_name, snapshot_date DESC);
CREATE INDEX IF NOT EXISTS idx_bpps_date
    ON b2b_product_profile_snapshots (snapshot_date DESC);
