-- B2B software review intelligence tables for churn prediction pipeline.
-- Source-agnostic: works with G2, Capterra, TrustRadius, Reddit, manual, etc.
-- Category-agnostic: any B2B software vertical (CRM, cloud, marketing, etc.)

-- Core review table with single-pass LLM enrichment state machine
CREATE TABLE IF NOT EXISTS b2b_reviews (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dedup_key               TEXT NOT NULL UNIQUE,

    -- Source
    source                  TEXT NOT NULL,
    source_url              TEXT,
    source_review_id        TEXT,

    -- Product/Vendor
    vendor_name             TEXT NOT NULL,
    product_name            TEXT,
    product_category        TEXT,

    -- Review content
    rating                  NUMERIC(3,1),
    rating_max              NUMERIC(2,0) NOT NULL DEFAULT 5 CHECK (rating_max > 0),
    summary                 TEXT,
    review_text             TEXT NOT NULL,
    pros                    TEXT,
    cons                    TEXT,

    -- Reviewer context (from source, pre-enrichment)
    reviewer_name           TEXT,
    reviewer_title          TEXT,
    reviewer_company        TEXT,
    company_size_raw        TEXT,
    reviewer_industry       TEXT,
    reviewed_at             TIMESTAMPTZ,

    -- Import metadata
    imported_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    import_batch_id         TEXT,
    raw_metadata            JSONB NOT NULL DEFAULT '{}',

    -- Enrichment (single-pass LLM)
    enrichment              JSONB,
    enrichment_status       TEXT NOT NULL DEFAULT 'pending',
    enrichment_attempts     INT NOT NULL DEFAULT 0,
    enriched_at             TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_b2b_reviews_dedup ON b2b_reviews(dedup_key);
CREATE INDEX IF NOT EXISTS idx_b2b_reviews_vendor ON b2b_reviews(vendor_name);
CREATE INDEX IF NOT EXISTS idx_b2b_reviews_category ON b2b_reviews(product_category);
CREATE INDEX IF NOT EXISTS idx_b2b_reviews_enrichment_pending
    ON b2b_reviews(enrichment_status) WHERE enrichment_status = 'pending';
CREATE INDEX IF NOT EXISTS idx_b2b_reviews_enrichment_data
    ON b2b_reviews USING GIN (enrichment) WHERE enrichment IS NOT NULL;


-- Per-vendor aggregated churn metrics (upserted by intelligence task)
CREATE TABLE IF NOT EXISTS b2b_churn_signals (
    id                          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vendor_name                 TEXT NOT NULL,
    product_category            TEXT,

    -- Aggregate metrics
    total_reviews               INT NOT NULL DEFAULT 0,
    negative_reviews            INT NOT NULL DEFAULT 0,
    churn_intent_count          INT NOT NULL DEFAULT 0,
    avg_urgency_score           NUMERIC(3,1) NOT NULL DEFAULT 0,
    avg_rating_normalized       NUMERIC(3,2),
    nps_proxy                   NUMERIC(5,1),

    -- Aggregated intelligence (JSONB)
    top_pain_categories         JSONB NOT NULL DEFAULT '[]',
    top_competitors             JSONB NOT NULL DEFAULT '[]',
    top_feature_gaps            JSONB NOT NULL DEFAULT '[]',
    price_complaint_rate        NUMERIC(5,4),
    decision_maker_churn_rate   NUMERIC(5,4),
    company_churn_list          JSONB NOT NULL DEFAULT '[]',
    quotable_evidence           JSONB NOT NULL DEFAULT '[]',

    last_computed_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- NULL-safe unique: COALESCE ensures upserts work when product_category is NULL
CREATE UNIQUE INDEX IF NOT EXISTS idx_b2b_churn_signals_vendor_category
    ON b2b_churn_signals (vendor_name, COALESCE(product_category, ''));


-- Structured intelligence products (weekly feeds, scorecards, reports)
CREATE TABLE IF NOT EXISTS b2b_intelligence (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_date             DATE NOT NULL DEFAULT CURRENT_DATE,
    report_type             TEXT NOT NULL,
    vendor_filter           TEXT,
    category_filter         TEXT,
    intelligence_data       JSONB NOT NULL DEFAULT '{}',
    executive_summary       TEXT,
    data_density            JSONB NOT NULL DEFAULT '{}',
    status                  TEXT NOT NULL DEFAULT 'draft',
    llm_model               TEXT,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_b2b_intelligence_type ON b2b_intelligence(report_type);
CREATE INDEX IF NOT EXISTS idx_b2b_intelligence_date ON b2b_intelligence(report_date DESC);
