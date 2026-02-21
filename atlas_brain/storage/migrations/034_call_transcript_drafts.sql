-- Add drafts JSONB column to call_transcripts
-- Stores generated draft content keyed by type: {"email": "...", "sms": "..."}

ALTER TABLE call_transcripts
    ADD COLUMN IF NOT EXISTS drafts JSONB DEFAULT '{}'::jsonb;
