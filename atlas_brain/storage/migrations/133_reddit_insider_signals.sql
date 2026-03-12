-- Reddit comment harvesting + insider signal support.
-- Adds content_type, threading columns to b2b_reviews, and insider aggregate
-- columns to b2b_churn_signals.

-- --------------------------------------------------------------------------
-- b2b_reviews: content type + thread metadata
-- --------------------------------------------------------------------------

ALTER TABLE b2b_reviews ADD COLUMN IF NOT EXISTS content_type   TEXT NOT NULL DEFAULT 'review';
ALTER TABLE b2b_reviews ADD COLUMN IF NOT EXISTS parent_review_id UUID REFERENCES b2b_reviews(id) ON DELETE SET NULL;
ALTER TABLE b2b_reviews ADD COLUMN IF NOT EXISTS thread_id      TEXT;
ALTER TABLE b2b_reviews ADD COLUMN IF NOT EXISTS comment_depth  SMALLINT NOT NULL DEFAULT 0;

-- Also add parser_version if not already present (was added in 055 but guard anyway)
-- (no-op if column exists)
ALTER TABLE b2b_reviews ADD COLUMN IF NOT EXISTS parser_version TEXT;

CREATE INDEX IF NOT EXISTS idx_b2b_reviews_thread
    ON b2b_reviews(thread_id) WHERE thread_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_b2b_reviews_content_type
    ON b2b_reviews(content_type) WHERE content_type != 'review';

CREATE INDEX IF NOT EXISTS idx_b2b_reviews_parent
    ON b2b_reviews(parent_review_id) WHERE parent_review_id IS NOT NULL;

-- --------------------------------------------------------------------------
-- b2b_churn_signals: insider aggregate columns
-- --------------------------------------------------------------------------

ALTER TABLE b2b_churn_signals ADD COLUMN IF NOT EXISTS insider_signal_count        INT NOT NULL DEFAULT 0;
ALTER TABLE b2b_churn_signals ADD COLUMN IF NOT EXISTS insider_org_health_summary  JSONB NOT NULL DEFAULT '{}';
ALTER TABLE b2b_churn_signals ADD COLUMN IF NOT EXISTS insider_talent_drain_rate   NUMERIC(5,4);
ALTER TABLE b2b_churn_signals ADD COLUMN IF NOT EXISTS insider_quotable_evidence   JSONB NOT NULL DEFAULT '[]';
