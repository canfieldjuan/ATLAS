-- Quote-card drafts: deterministic, evidence-backed customer-proof cards
-- generated from Content Ops source material. One row is one reviewable
-- quote-card draft.
--
-- The status lifecycle mirrors the other generated assets: drafts start at
-- 'draft' and host workflows can move them through 'approved', 'rejected', or
-- custom intermediate states without a CHECK constraint.

CREATE TABLE IF NOT EXISTS quote_card_drafts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id TEXT NOT NULL DEFAULT '',
    target_id TEXT NOT NULL DEFAULT '',
    target_mode TEXT NOT NULL,
    theme TEXT NOT NULL DEFAULT 'customer_proof',
    quote TEXT NOT NULL,
    attribution TEXT NOT NULL DEFAULT '',
    headline TEXT NOT NULL DEFAULT '',
    supporting_text TEXT NOT NULL DEFAULT '',
    source_id TEXT NOT NULL DEFAULT '',
    source_type TEXT NOT NULL DEFAULT '',
    company_name TEXT NOT NULL DEFAULT '',
    vendor_name TEXT NOT NULL DEFAULT '',
    pain_points JSONB NOT NULL DEFAULT '[]'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    status TEXT NOT NULL DEFAULT 'draft',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_quote_card_drafts_target
    ON quote_card_drafts (account_id, target_mode, target_id);

CREATE INDEX IF NOT EXISTS idx_quote_card_drafts_status
    ON quote_card_drafts (account_id, status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_quote_card_drafts_theme
    ON quote_card_drafts (account_id, theme, created_at DESC);
