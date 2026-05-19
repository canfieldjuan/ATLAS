-- Ticket FAQ Markdown: deterministic FAQ documents generated from support
-- tickets, cases, conversations, and complaint source rows. One row is one
-- reviewable Markdown document; individual FAQ entries live in items JSONB.

CREATE TABLE IF NOT EXISTS ticket_faq_markdown (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id TEXT NOT NULL DEFAULT '',
    target_id TEXT NOT NULL DEFAULT '',
    target_mode TEXT NOT NULL,
    title TEXT NOT NULL,
    markdown TEXT NOT NULL,
    items JSONB NOT NULL DEFAULT '[]'::jsonb,
    source_count INTEGER NOT NULL DEFAULT 0,
    ticket_source_count INTEGER NOT NULL DEFAULT 0,
    output_checks JSONB NOT NULL DEFAULT '{}'::jsonb,
    warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    status TEXT NOT NULL DEFAULT 'draft',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ticket_faq_markdown_target
    ON ticket_faq_markdown (account_id, target_mode, target_id);

CREATE INDEX IF NOT EXISTS idx_ticket_faq_markdown_status
    ON ticket_faq_markdown (account_id, status, created_at DESC);
