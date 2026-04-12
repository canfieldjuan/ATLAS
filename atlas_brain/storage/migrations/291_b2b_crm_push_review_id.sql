-- Preserve exact review drillthrough context for CRM push activity rows.
ALTER TABLE b2b_crm_push_log
    ADD COLUMN IF NOT EXISTS review_id UUID REFERENCES b2b_reviews(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_crm_push_log_review
    ON b2b_crm_push_log (review_id)
    WHERE review_id IS NOT NULL;
