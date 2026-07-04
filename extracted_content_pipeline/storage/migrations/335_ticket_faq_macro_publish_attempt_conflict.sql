-- Persist the terminal-state conflict signal on FAQ macro publish attempts.
-- Set when the external macro was published but the FAQ row was concurrently
-- moved to a review-decided state (reject/archive) so the approved->published
-- mark could not land: the immediate response reports this via status_conflict,
-- and the append-only attempt history must carry the same durable signal.

ALTER TABLE ticket_faq_macro_publish_attempts
    ADD COLUMN IF NOT EXISTS status_conflict BOOLEAN NOT NULL DEFAULT FALSE;
