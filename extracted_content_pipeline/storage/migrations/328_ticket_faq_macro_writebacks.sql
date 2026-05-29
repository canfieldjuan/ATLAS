-- FAQ macro writeback idempotency: maps one approved FAQ item to the external
-- macro / saved reply / canned response created in a customer's support tool.

CREATE TABLE IF NOT EXISTS ticket_faq_macro_writebacks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id TEXT NOT NULL DEFAULT '',
    platform TEXT NOT NULL,
    faq_draft_id UUID NOT NULL REFERENCES ticket_faq_markdown(id) ON DELETE CASCADE,
    faq_item_id TEXT NOT NULL,
    external_id TEXT NOT NULL,
    external_url TEXT NOT NULL DEFAULT '',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_ticket_faq_macro_writebacks_account_id CHECK (btrim(account_id) <> ''),
    CONSTRAINT chk_ticket_faq_macro_writebacks_platform CHECK (btrim(platform) <> ''),
    CONSTRAINT chk_ticket_faq_macro_writebacks_faq_item_id CHECK (btrim(faq_item_id) <> ''),
    CONSTRAINT chk_ticket_faq_macro_writebacks_external_id CHECK (btrim(external_id) <> ''),
    UNIQUE (account_id, platform, faq_draft_id, faq_item_id),
    UNIQUE (account_id, platform, external_id)
);

CREATE INDEX IF NOT EXISTS idx_ticket_faq_macro_writebacks_faq
    ON ticket_faq_macro_writebacks (faq_draft_id);

CREATE INDEX IF NOT EXISTS idx_ticket_faq_macro_writebacks_platform
    ON ticket_faq_macro_writebacks (account_id, platform, updated_at DESC);
