-- Promote 4 heavily-queried deep_extraction enum fields to indexed columns.
-- Migration 153
--
-- These are extracted by the LLM deep enrichment pass and queried 155+
-- times across dashboards, materialized views, and intelligence tasks.
-- GIN index on the JSONB blob does NOT optimize ->> text extraction.

ALTER TABLE product_reviews
    ADD COLUMN IF NOT EXISTS would_repurchase      BOOLEAN,
    ADD COLUMN IF NOT EXISTS replacement_behavior   TEXT,
    ADD COLUMN IF NOT EXISTS sentiment_trajectory   TEXT,
    ADD COLUMN IF NOT EXISTS consequence_severity   TEXT;

CREATE INDEX IF NOT EXISTS idx_product_reviews_sentiment_trajectory
    ON product_reviews (sentiment_trajectory)
    WHERE sentiment_trajectory IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_product_reviews_would_repurchase
    ON product_reviews (would_repurchase)
    WHERE would_repurchase IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_product_reviews_replacement_behavior
    ON product_reviews (replacement_behavior)
    WHERE replacement_behavior IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_product_reviews_consequence_severity
    ON product_reviews (consequence_severity)
    WHERE consequence_severity IS NOT NULL;
