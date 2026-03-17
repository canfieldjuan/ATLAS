-- Extract source_weight from raw_metadata JSONB into indexed column.
-- Migration 152
--
-- Every parser sets source_weight (0.0-1.0) in raw_metadata to indicate
-- signal quality: G2/Gartner=1.0, Capterra=0.9, Reddit=0.5-0.6, etc.
-- Heavily read in aggregations, ordering, and filters.

ALTER TABLE b2b_reviews
    ADD COLUMN IF NOT EXISTS source_weight REAL;

CREATE INDEX IF NOT EXISTS idx_b2b_reviews_source_weight
    ON b2b_reviews (source_weight DESC)
    WHERE source_weight IS NOT NULL;
