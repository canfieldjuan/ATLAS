-- Migration 041: Add business intent classification column
-- Supports Stage 2 intent classification (estimate_request, reschedule, complaint, info_admin)

ALTER TABLE processed_emails
    ADD COLUMN IF NOT EXISTS intent VARCHAR(32);

ALTER TABLE contact_interactions
    ADD COLUMN IF NOT EXISTS intent VARCHAR(32);

CREATE INDEX IF NOT EXISTS idx_processed_emails_intent
    ON processed_emails(intent) WHERE intent IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_contact_interactions_intent
    ON contact_interactions(intent) WHERE intent IS NOT NULL;
