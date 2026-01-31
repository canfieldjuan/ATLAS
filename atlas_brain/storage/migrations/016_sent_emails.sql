-- Sent emails table for email history tracking
-- Migration: 016_sent_emails.sql

CREATE TABLE IF NOT EXISTS sent_emails (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    to_addresses TEXT[] NOT NULL,
    cc_addresses TEXT[] DEFAULT '{}',
    subject TEXT NOT NULL,
    body TEXT NOT NULL,
    template_type VARCHAR(32),  -- "generic", "estimate", "proposal"
    session_id UUID REFERENCES sessions(id) ON DELETE SET NULL,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    attachments TEXT[] DEFAULT '{}',
    resend_message_id VARCHAR(128),
    sent_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Index for querying emails by date
CREATE INDEX IF NOT EXISTS idx_sent_emails_sent_at
    ON sent_emails(sent_at DESC);

-- Index for querying emails by recipient
CREATE INDEX IF NOT EXISTS idx_sent_emails_to
    ON sent_emails USING GIN(to_addresses);

-- Index for querying emails by template type
CREATE INDEX IF NOT EXISTS idx_sent_emails_template
    ON sent_emails(template_type, sent_at DESC)
    WHERE template_type IS NOT NULL;

-- Index for user's email history
CREATE INDEX IF NOT EXISTS idx_sent_emails_user
    ON sent_emails(user_id, sent_at DESC)
    WHERE user_id IS NOT NULL;

-- Index for session's emails
CREATE INDEX IF NOT EXISTS idx_sent_emails_session
    ON sent_emails(session_id, sent_at DESC)
    WHERE session_id IS NOT NULL;

COMMENT ON TABLE sent_emails IS 'History of sent emails for tracking and queries';
COMMENT ON COLUMN sent_emails.to_addresses IS 'List of recipient email addresses';
COMMENT ON COLUMN sent_emails.template_type IS 'Type of email template used: generic, estimate, proposal';
COMMENT ON COLUMN sent_emails.resend_message_id IS 'Message ID from Resend API for tracking';
