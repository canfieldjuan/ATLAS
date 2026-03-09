-- Composite index for source-health dashboard aggregation.
-- The health query groups by source and filters on started_at.
CREATE INDEX IF NOT EXISTS idx_b2b_scrape_log_source_started
    ON b2b_scrape_log (source, started_at DESC);
