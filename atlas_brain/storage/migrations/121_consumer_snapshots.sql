-- Migration 121: Consumer intelligence snapshots + change events
--
-- Mirrors B2B vendor_snapshots / change_events pattern for consumer
-- product review pipeline. Enables historical brand health tracking
-- and anomaly detection (pain score spikes, safety emergences, etc.).

-- Append-only daily brand health snapshot
CREATE TABLE IF NOT EXISTS brand_intelligence_snapshots (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    brand           TEXT NOT NULL,
    snapshot_date   DATE NOT NULL DEFAULT CURRENT_DATE,
    total_reviews   INTEGER NOT NULL DEFAULT 0,
    avg_rating      NUMERIC(3,2),
    avg_pain_score  NUMERIC(3,1),
    health_score    NUMERIC(5,2),
    repurchase_yes  INTEGER NOT NULL DEFAULT 0,
    repurchase_no   INTEGER NOT NULL DEFAULT 0,
    complaint_count INTEGER NOT NULL DEFAULT 0,
    safety_count    INTEGER NOT NULL DEFAULT 0,
    top_complaint   TEXT,
    top_feature_request TEXT,
    competitive_flow_count INTEGER NOT NULL DEFAULT 0,
    trajectory_positive INTEGER NOT NULL DEFAULT 0,
    trajectory_negative INTEGER NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (brand, snapshot_date)
);

CREATE INDEX IF NOT EXISTS idx_bis_brand_date
    ON brand_intelligence_snapshots (brand, snapshot_date DESC);
CREATE INDEX IF NOT EXISTS idx_bis_date
    ON brand_intelligence_snapshots (snapshot_date DESC);

-- Consumer product change event log
CREATE TABLE IF NOT EXISTS product_change_events (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    brand           TEXT NOT NULL,
    asin            TEXT,
    event_date      DATE NOT NULL DEFAULT CURRENT_DATE,
    event_type      TEXT NOT NULL,
    description     TEXT NOT NULL DEFAULT '',
    old_value       NUMERIC,
    new_value       NUMERIC,
    delta           NUMERIC,
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pce_brand_date
    ON product_change_events (brand, event_date DESC);
CREATE INDEX IF NOT EXISTS idx_pce_type_date
    ON product_change_events (event_type, event_date DESC);
CREATE INDEX IF NOT EXISTS idx_pce_date
    ON product_change_events (event_date DESC);
