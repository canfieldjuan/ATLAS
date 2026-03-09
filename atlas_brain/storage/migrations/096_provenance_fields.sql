-- Provenance columns: trace every intelligence row back to raw evidence.
-- Phase 0, Deliverable 3.

-- b2b_reviews: which model enriched this review
ALTER TABLE b2b_reviews ADD COLUMN IF NOT EXISTS enrichment_model TEXT;
CREATE INDEX IF NOT EXISTS idx_b2b_reviews_enrich_model
    ON b2b_reviews (enrichment_model) WHERE enrichment_model IS NOT NULL;

-- b2b_churn_signals: provenance
ALTER TABLE b2b_churn_signals
    ADD COLUMN IF NOT EXISTS source_distribution  JSONB NOT NULL DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS sample_review_ids    UUID[] NOT NULL DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS review_window_start  TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS review_window_end    TIMESTAMPTZ;

-- b2b_product_profiles: provenance
ALTER TABLE b2b_product_profiles
    ADD COLUMN IF NOT EXISTS source_distribution  JSONB NOT NULL DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS sample_review_ids    UUID[] NOT NULL DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS review_window_start  TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS review_window_end    TIMESTAMPTZ;

-- b2b_intelligence: provenance
ALTER TABLE b2b_intelligence
    ADD COLUMN IF NOT EXISTS source_review_count  INT,
    ADD COLUMN IF NOT EXISTS source_distribution  JSONB NOT NULL DEFAULT '{}';
