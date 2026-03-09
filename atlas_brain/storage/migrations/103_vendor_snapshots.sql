-- Migration 103: Vendor health snapshots + change events (Phase 3)
--
-- Append-only tables for historical vendor health tracking and
-- structural change event detection.

-- Append-only daily vendor health snapshot
CREATE TABLE IF NOT EXISTS b2b_vendor_snapshots (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vendor_name     TEXT NOT NULL,
    snapshot_date   DATE NOT NULL DEFAULT CURRENT_DATE,
    total_reviews   INTEGER NOT NULL DEFAULT 0,
    churn_intent    INTEGER NOT NULL DEFAULT 0,
    churn_density   NUMERIC(5,1) NOT NULL DEFAULT 0,
    avg_urgency     NUMERIC(4,1) NOT NULL DEFAULT 0,
    positive_review_pct NUMERIC(5,1),
    recommend_ratio NUMERIC(5,1),
    top_pain        TEXT,
    top_competitor  TEXT,
    pain_count      INTEGER NOT NULL DEFAULT 0,
    competitor_count INTEGER NOT NULL DEFAULT 0,
    displacement_edge_count INTEGER NOT NULL DEFAULT 0,
    high_intent_company_count INTEGER NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (vendor_name, snapshot_date)
);

CREATE INDEX IF NOT EXISTS idx_bvs_vendor_date
    ON b2b_vendor_snapshots (vendor_name, snapshot_date DESC);
CREATE INDEX IF NOT EXISTS idx_bvs_date
    ON b2b_vendor_snapshots (snapshot_date DESC);

-- Structural change event log
CREATE TABLE IF NOT EXISTS b2b_change_events (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vendor_name     TEXT NOT NULL,
    event_date      DATE NOT NULL DEFAULT CURRENT_DATE,
    event_type      TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    old_value       NUMERIC,
    new_value       NUMERIC,
    delta           NUMERIC,
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_bce_vendor_date
    ON b2b_change_events (vendor_name, event_date DESC);
CREATE INDEX IF NOT EXISTS idx_bce_type_date
    ON b2b_change_events (event_type, event_date DESC);
CREATE INDEX IF NOT EXISTS idx_bce_date
    ON b2b_change_events (event_date DESC);
