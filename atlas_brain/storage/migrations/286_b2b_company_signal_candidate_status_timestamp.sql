ALTER TABLE b2b_company_signal_candidates
    ADD COLUMN IF NOT EXISTS review_status_updated_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_b2b_company_signal_candidates_status_updated
    ON b2b_company_signal_candidates (review_status, review_status_updated_at DESC, last_seen_at DESC);
