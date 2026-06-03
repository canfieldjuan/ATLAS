-- Ad-copy drafts: deterministic, evidence-backed paid-social copy generated
-- from Content Ops source material. One row is one reviewable ad-copy draft.
--
-- The status lifecycle mirrors the other generated assets: drafts start at
-- 'draft' and host workflows can move them through 'approved', 'rejected', or
-- custom intermediate states without a CHECK constraint.

CREATE TABLE IF NOT EXISTS ad_copy_drafts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id TEXT NOT NULL DEFAULT '',
    target_id TEXT NOT NULL DEFAULT '',
    target_mode TEXT NOT NULL,
    channel TEXT NOT NULL DEFAULT 'paid_social',
    format TEXT NOT NULL DEFAULT 'single_image',
    headline TEXT NOT NULL,
    primary_text TEXT NOT NULL,
    cta TEXT NOT NULL DEFAULT '',
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

CREATE INDEX IF NOT EXISTS idx_ad_copy_drafts_target
    ON ad_copy_drafts (account_id, target_mode, target_id);

CREATE INDEX IF NOT EXISTS idx_ad_copy_drafts_status
    ON ad_copy_drafts (account_id, status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_ad_copy_drafts_channel
    ON ad_copy_drafts (account_id, channel, created_at DESC);
