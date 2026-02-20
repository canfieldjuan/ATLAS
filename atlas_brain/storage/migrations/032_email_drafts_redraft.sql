-- 032_email_drafts_redraft.sql
-- Add redraft support columns to email_drafts table.
-- Enables draft chaining: rejected drafts can be redrafted with context.

ALTER TABLE email_drafts
    ADD COLUMN IF NOT EXISTS attempt_number INTEGER NOT NULL DEFAULT 1;

ALTER TABLE email_drafts
    ADD COLUMN IF NOT EXISTS parent_draft_id UUID REFERENCES email_drafts(id);

ALTER TABLE email_drafts
    ADD COLUMN IF NOT EXISTS rejection_reason TEXT;

CREATE INDEX IF NOT EXISTS idx_email_drafts_chain
    ON email_drafts (gmail_message_id, attempt_number DESC);
