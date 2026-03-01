-- Market intelligence aggregation layer: per-brand scorecards and daily reports
-- Aggregates over deep_extraction JSONB (source-agnostic)

-- Per-brand materialized scorecard, upserted daily
CREATE TABLE IF NOT EXISTS brand_intelligence (
    brand               TEXT NOT NULL,
    source              TEXT NOT NULL DEFAULT 'all',
    total_reviews       INT NOT NULL DEFAULT 0,
    avg_rating          NUMERIC(3,2),
    avg_pain_score      NUMERIC(3,1),
    repurchase_yes      INT NOT NULL DEFAULT 0,
    repurchase_no       INT NOT NULL DEFAULT 0,
    sentiment_breakdown JSONB DEFAULT '{}'::jsonb,
    top_feature_requests JSONB DEFAULT '[]'::jsonb,
    top_complaints      JSONB DEFAULT '[]'::jsonb,
    competitive_flows   JSONB DEFAULT '[]'::jsonb,
    buyer_profile       JSONB DEFAULT '{}'::jsonb,
    positive_aspects    JSONB DEFAULT '[]'::jsonb,
    health_score        NUMERIC(5,2),
    last_computed_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (brand, source)
);

CREATE INDEX IF NOT EXISTS idx_brand_intelligence_health
    ON brand_intelligence (health_score DESC NULLS LAST);

-- Daily LLM analysis reports
CREATE TABLE IF NOT EXISTS market_intelligence_reports (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_date         DATE NOT NULL,
    report_type         TEXT NOT NULL DEFAULT 'daily_competitive',
    analysis_text       TEXT,
    competitive_flows   JSONB DEFAULT '[]'::jsonb,
    feature_gaps        JSONB DEFAULT '[]'::jsonb,
    buyer_personas      JSONB DEFAULT '[]'::jsonb,
    brand_scorecards    JSONB DEFAULT '[]'::jsonb,
    insights            JSONB DEFAULT '[]'::jsonb,
    recommendations     JSONB DEFAULT '[]'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (report_date, report_type)
);

CREATE INDEX IF NOT EXISTS idx_market_intelligence_reports_date
    ON market_intelligence_reports (report_date DESC);

-- Performance indexes for competitive intelligence CROSS JOIN queries
-- (product_reviews queries filter on deep_enrichment_status = 'enriched')
CREATE INDEX IF NOT EXISTS idx_product_reviews_deep_enriched
    ON product_reviews (deep_enrichment_status)
    WHERE deep_enrichment_status = 'enriched';
