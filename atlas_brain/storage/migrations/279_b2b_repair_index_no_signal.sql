-- Widen repair status index to cover no_signal rows (the main repair query
-- fetches enrichment_status IN (enriched, no_signal) but the original index
-- only covered enriched).

DROP INDEX IF EXISTS idx_b2b_reviews_repair_status;

CREATE INDEX IF NOT EXISTS idx_b2b_reviews_repair_status
    ON b2b_reviews (enrichment_repair_status, enriched_at DESC)
    WHERE enrichment_status IN ('enriched', 'no_signal');
