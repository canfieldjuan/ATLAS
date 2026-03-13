-- Reddit scraper monitoring indexes.
--
-- The /admin/costs/scraping/reddit/* endpoints filter exclusively on
-- source='reddit' with an imported_at range window.  Without these indexes
-- every query does a full b2b_reviews table scan.
--
-- idx_b2b_reviews_source_imported  — covers all four Reddit endpoints
-- idx_b2b_reviews_reddit_subreddit — accelerates the by-subreddit GROUP BY

CREATE INDEX IF NOT EXISTS idx_b2b_reviews_source_imported
    ON b2b_reviews(source, imported_at DESC);

CREATE INDEX IF NOT EXISTS idx_b2b_reviews_reddit_subreddit
    ON b2b_reviews((raw_metadata->>'subreddit'))
    WHERE source = 'reddit';
