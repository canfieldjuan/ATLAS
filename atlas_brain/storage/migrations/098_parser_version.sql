-- Parser versioning: track which parser version extracted each review
-- Enables selective re-extraction when parser logic changes

ALTER TABLE b2b_reviews ADD COLUMN IF NOT EXISTS parser_version TEXT;
CREATE INDEX IF NOT EXISTS idx_b2b_reviews_parser_version
    ON b2b_reviews (parser_version) WHERE parser_version IS NOT NULL;

ALTER TABLE b2b_scrape_log ADD COLUMN IF NOT EXISTS parser_version TEXT;
CREATE INDEX IF NOT EXISTS idx_b2b_scrape_log_parser_version
    ON b2b_scrape_log (parser_version) WHERE parser_version IS NOT NULL;
