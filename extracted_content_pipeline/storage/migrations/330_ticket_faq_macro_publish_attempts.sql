-- Append-only audit trail for FAQ macro publish attempts.

CREATE TABLE IF NOT EXISTS ticket_faq_macro_publish_attempts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id TEXT NOT NULL,
    faq_draft_id UUID NOT NULL REFERENCES ticket_faq_markdown(id) ON DELETE CASCADE,
    draft_status TEXT NOT NULL DEFAULT '',
    ok BOOLEAN NOT NULL DEFAULT FALSE,
    publishable_count INTEGER NOT NULL DEFAULT 0,
    skipped_count INTEGER NOT NULL DEFAULT 0,
    published_count INTEGER NOT NULL DEFAULT 0,
    updated_count INTEGER NOT NULL DEFAULT 0,
    failed_count INTEGER NOT NULL DEFAULT 0,
    pending_reconcile_count INTEGER NOT NULL DEFAULT 0,
    draft_status_updated BOOLEAN NOT NULL DEFAULT FALSE,
    skipped JSONB NOT NULL DEFAULT '[]'::jsonb,
    results JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_ticket_faq_macro_publish_attempts_account_id
        CHECK (btrim(account_id) <> ''),
    CONSTRAINT chk_ticket_faq_macro_publish_attempts_counts
        CHECK (
            publishable_count >= 0
            AND skipped_count >= 0
            AND published_count >= 0
            AND updated_count >= 0
            AND failed_count >= 0
            AND pending_reconcile_count >= 0
        )
);

CREATE INDEX IF NOT EXISTS idx_ticket_faq_macro_publish_attempts_faq
    ON ticket_faq_macro_publish_attempts (account_id, faq_draft_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_ticket_faq_macro_publish_attempts_created
    ON ticket_faq_macro_publish_attempts (account_id, created_at DESC);
