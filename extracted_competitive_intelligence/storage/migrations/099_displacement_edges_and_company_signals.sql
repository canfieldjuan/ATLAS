-- Migration 099: First-class displacement edges + company signals
--
-- Promotes two high-value entities from JSONB blobs to queryable tables:
--   b2b_displacement_edges  -- append-only time-series of competitive flows
--   b2b_company_signals     -- UPSERT per company-vendor pair

BEGIN;

-- -----------------------------------------------------------------------
-- Displacement edges (append-only, one row per vendor-pair per day)
-- -----------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS b2b_displacement_edges (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    from_vendor         TEXT NOT NULL,
    to_vendor           TEXT NOT NULL,
    mention_count       INT NOT NULL DEFAULT 0,
    primary_driver      TEXT,
    signal_strength     TEXT CHECK (signal_strength IN ('strong', 'moderate', 'emerging')),
    key_quote           TEXT,
    source_distribution JSONB DEFAULT '{}'::jsonb,
    sample_review_ids   UUID[] DEFAULT '{}',
    confidence_score    NUMERIC(3,2) DEFAULT 0.00,
    computed_date       DATE NOT NULL DEFAULT CURRENT_DATE,
    report_id           UUID REFERENCES b2b_intelligence(id) ON DELETE SET NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (from_vendor, to_vendor, computed_date)
);

CREATE INDEX IF NOT EXISTS idx_displacement_edges_from_vendor
    ON b2b_displacement_edges (from_vendor);
CREATE INDEX IF NOT EXISTS idx_displacement_edges_to_vendor
    ON b2b_displacement_edges (to_vendor);
CREATE INDEX IF NOT EXISTS idx_displacement_edges_computed_date
    ON b2b_displacement_edges (computed_date DESC);
CREATE INDEX IF NOT EXISTS idx_displacement_edges_strong
    ON b2b_displacement_edges (signal_strength)
    WHERE signal_strength = 'strong';

-- -----------------------------------------------------------------------
-- Company signals (UPSERT on company + vendor)
-- -----------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS b2b_company_signals (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_name    TEXT NOT NULL,
    vendor_name     TEXT NOT NULL,
    urgency_score   NUMERIC(3,1),
    pain_category   TEXT,
    buyer_role      TEXT,
    decision_maker  BOOLEAN,
    seat_count      INT,
    contract_end    TEXT,
    buying_stage    TEXT,
    review_id       UUID REFERENCES b2b_reviews(id) ON DELETE SET NULL,
    source          TEXT,
    first_seen_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at    TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (company_name, vendor_name)
);

CREATE INDEX IF NOT EXISTS idx_company_signals_vendor
    ON b2b_company_signals (vendor_name);
CREATE INDEX IF NOT EXISTS idx_company_signals_urgency
    ON b2b_company_signals (urgency_score DESC);
CREATE INDEX IF NOT EXISTS idx_company_signals_last_seen
    ON b2b_company_signals (last_seen_at DESC);

COMMIT;
