-- Extract author_churn_score from raw_metadata JSONB into indexed column.
-- Migration 148
--
-- Reddit parser computes per-author churn scores (0-10) based on post
-- history: migration keywords, upvotes, churn qualifiers. Only Reddit
-- reviews have scores; other sources get NULL.

ALTER TABLE b2b_reviews
    ADD COLUMN IF NOT EXISTS author_churn_score REAL;

CREATE INDEX IF NOT EXISTS idx_b2b_reviews_author_churn
    ON b2b_reviews (author_churn_score DESC)
    WHERE author_churn_score IS NOT NULL AND author_churn_score > 0;
