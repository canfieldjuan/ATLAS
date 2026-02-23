-- Migration 043: Add stale_check_metadata to processed_emails for stale email re-engagement tracking
ALTER TABLE processed_emails
    ADD COLUMN IF NOT EXISTS stale_check_metadata JSONB DEFAULT '{}'::jsonb;
