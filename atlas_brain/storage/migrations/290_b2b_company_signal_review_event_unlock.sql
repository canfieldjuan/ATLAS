ALTER TABLE b2b_company_signal_review_events
    ADD COLUMN IF NOT EXISTS candidate_source TEXT;

ALTER TABLE b2b_company_signal_review_events
    ADD COLUMN IF NOT EXISTS canonical_gap_reason TEXT;

ALTER TABLE b2b_company_signal_review_events
    ADD COLUMN IF NOT EXISTS review_unlock_path TEXT;

ALTER TABLE b2b_company_signal_review_events
    ADD COLUMN IF NOT EXISTS review_unlock_reason TEXT;

CREATE INDEX IF NOT EXISTS idx_b2b_company_signal_review_events_unlock_path
    ON b2b_company_signal_review_events (review_unlock_path, created_at DESC);
