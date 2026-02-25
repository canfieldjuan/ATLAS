-- Complaint mining: Amazon review analysis pipeline
-- product_reviews: imported reviews with inline enrichment columns
-- product_pain_points: per-ASIN aggregated state
-- complaint_reports: analysis history

CREATE TABLE IF NOT EXISTS product_reviews (
    id                          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dedup_key                   TEXT NOT NULL UNIQUE,
    asin                        TEXT NOT NULL,
    rating                      NUMERIC(2,1) NOT NULL,
    summary                     TEXT,
    review_text                 TEXT,
    reviewer_id                 TEXT,
    -- Source metadata
    source                      TEXT NOT NULL DEFAULT 'amazon',
    source_category             TEXT,
    matched_keywords            TEXT[] NOT NULL DEFAULT '{}',
    hardware_category           TEXT[] NOT NULL DEFAULT '{}',
    issue_types                 TEXT[] NOT NULL DEFAULT '{}',
    -- Helpfulness
    helpful_votes               INT NOT NULL DEFAULT 0,
    total_votes                 INT NOT NULL DEFAULT 0,
    reviewed_at                 TIMESTAMPTZ,
    imported_at                 TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- Enrichment (LLM-populated)
    root_cause                  TEXT,
    specific_complaint          TEXT,
    severity                    TEXT,
    pain_score                  NUMERIC(3,1),
    time_to_failure             TEXT,
    workaround_found            BOOLEAN,
    workaround_text             TEXT,
    alternative_mentioned       BOOLEAN,
    alternative_asin            TEXT,
    alternative_name            TEXT,
    actionable_for_manufacturing BOOLEAN,
    manufacturing_suggestion    TEXT,
    -- Status
    enrichment_status           TEXT NOT NULL DEFAULT 'pending',
    enrichment_attempts         INT NOT NULL DEFAULT 0,
    enriched_at                 TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_product_reviews_dedup ON product_reviews(dedup_key);
CREATE INDEX IF NOT EXISTS idx_product_reviews_asin ON product_reviews(asin);
CREATE INDEX IF NOT EXISTS idx_product_reviews_enrichment
    ON product_reviews(enrichment_status) WHERE enrichment_status = 'pending';

-- Per-ASIN aggregated state (like entity_pressure_baselines)
CREATE TABLE IF NOT EXISTS product_pain_points (
    id                          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asin                        TEXT NOT NULL UNIQUE,
    product_name                TEXT,
    category                    TEXT,
    total_reviews               INT NOT NULL DEFAULT 0,
    complaint_reviews           INT NOT NULL DEFAULT 0,
    complaint_rate              NUMERIC(5,4) NOT NULL DEFAULT 0.0,
    top_complaints              JSONB NOT NULL DEFAULT '[]',
    root_cause_distribution     JSONB NOT NULL DEFAULT '{}',
    severity_distribution       JSONB NOT NULL DEFAULT '{}',
    differentiation_opportunities JSONB NOT NULL DEFAULT '[]',
    alternative_products        JSONB NOT NULL DEFAULT '[]',
    pain_score                  NUMERIC(3,1) NOT NULL DEFAULT 0.0,
    last_computed_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_product_pain_points_asin ON product_pain_points(asin);
CREATE INDEX IF NOT EXISTS idx_product_pain_points_pain ON product_pain_points(pain_score DESC);

-- Analysis history (like reasoning_journal)
CREATE TABLE IF NOT EXISTS complaint_reports (
    id                          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_date                 DATE NOT NULL DEFAULT CURRENT_DATE,
    report_type                 TEXT NOT NULL DEFAULT 'daily',
    category_filter             TEXT,
    analysis_output             TEXT,
    top_pain_points             JSONB NOT NULL DEFAULT '[]',
    opportunities               JSONB NOT NULL DEFAULT '[]',
    recommendations             JSONB NOT NULL DEFAULT '[]',
    product_highlights          JSONB NOT NULL DEFAULT '[]',
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
