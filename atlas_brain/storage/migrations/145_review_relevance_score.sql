-- Persist relevance scores on b2b_reviews for threshold tuning and analytics.
-- Migration 145
--
-- Structured sources (G2, Capterra, etc.) bypass the filter and get NULL.
-- Social sources (Reddit, HN, GitHub, RSS) get 0.0-1.0 from the rule scorer.

ALTER TABLE b2b_reviews
    ADD COLUMN IF NOT EXISTS relevance_score REAL;

-- Partial index: only social-source reviews have scores, structured are NULL
CREATE INDEX IF NOT EXISTS idx_b2b_reviews_relevance
    ON b2b_reviews (relevance_score)
    WHERE relevance_score IS NOT NULL;
