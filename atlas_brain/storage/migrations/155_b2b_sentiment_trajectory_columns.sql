-- Promote B2B sentiment_trajectory subfields from enrichment JSONB to columns.
-- Migration 155
--
-- enrichment->'sentiment_trajectory' is a nested dict with direction, tenure,
-- and turning_point. These 3 subfields are queried heavily in aggregations
-- across _b2b_shared.py, b2b_dashboard.py, and campaign generation.

ALTER TABLE b2b_reviews
    ADD COLUMN IF NOT EXISTS sentiment_direction      TEXT,
    ADD COLUMN IF NOT EXISTS sentiment_tenure          TEXT,
    ADD COLUMN IF NOT EXISTS sentiment_turning_point   TEXT;

CREATE INDEX IF NOT EXISTS idx_b2b_reviews_sentiment_direction
    ON b2b_reviews (sentiment_direction)
    WHERE sentiment_direction IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_b2b_reviews_sentiment_tenure
    ON b2b_reviews (sentiment_tenure)
    WHERE sentiment_tenure IS NOT NULL;
