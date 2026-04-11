ALTER TABLE b2b_company_signal_candidates
    ADD COLUMN IF NOT EXISTS review_status TEXT NOT NULL DEFAULT 'pending'
        CHECK (review_status IN ('pending', 'approved', 'suppressed'));

ALTER TABLE b2b_company_signal_candidates
    ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMPTZ;

ALTER TABLE b2b_company_signal_candidates
    ADD COLUMN IF NOT EXISTS reviewed_by TEXT;

ALTER TABLE b2b_company_signal_candidates
    ADD COLUMN IF NOT EXISTS review_notes TEXT;

CREATE INDEX IF NOT EXISTS idx_b2b_company_signal_candidates_review_status
    ON b2b_company_signal_candidates (review_status, candidate_bucket, last_seen_at DESC);
