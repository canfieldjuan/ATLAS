-- Promote Reddit-specific analytics fields from raw_metadata to columns.
-- Migration 156
--
-- These 6 fields are extracted by the Reddit parser into raw_metadata JSONB
-- and queried ~45 times across admin analytics endpoints with repeated
-- ->> extraction + casting. Reddit-only; other sources get NULL.

ALTER TABLE b2b_reviews
    ADD COLUMN IF NOT EXISTS reddit_subreddit     TEXT,
    ADD COLUMN IF NOT EXISTS reddit_trending      TEXT,
    ADD COLUMN IF NOT EXISTS reddit_flair         TEXT,
    ADD COLUMN IF NOT EXISTS reddit_is_edited     BOOLEAN,
    ADD COLUMN IF NOT EXISTS reddit_is_crosspost  BOOLEAN,
    ADD COLUMN IF NOT EXISTS reddit_num_comments  INT;

CREATE INDEX IF NOT EXISTS idx_b2b_reviews_subreddit
    ON b2b_reviews (reddit_subreddit)
    WHERE reddit_subreddit IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_b2b_reviews_trending
    ON b2b_reviews (reddit_trending)
    WHERE reddit_trending IS NOT NULL;
