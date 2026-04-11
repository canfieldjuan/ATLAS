ALTER TABLE b2b_company_signal_review_events
    ADD COLUMN IF NOT EXISTS review_priority_band TEXT;

ALTER TABLE b2b_company_signal_review_events
    ADD COLUMN IF NOT EXISTS review_priority_reason TEXT;
