-- Landing pages: per-campaign rendered marketing landing-page drafts produced
-- by the AI Content Ops landing-page generator. Sibling to b2b_campaigns
-- (transactional emails) and reports (vendor-pressure reports), but with a
-- richer marketing-shaped schema: hero section, CTA, SEO meta, ordered body
-- sections, and the marketing-campaign context (name / persona / value prop)
-- the page was generated for.
--
-- Trigger shape is per-campaign (not per-opportunity), so target_id columns
-- from the campaigns/reports schema are intentionally absent. Hosts identify
-- a landing page by (account_id, campaign_name, slug).
--
-- Status lifecycle mirrors b2b_campaigns / reports: draft / queued /
-- approved / rejected / expired. No CHECK constraint on status so hosts may
-- extend with intermediate states (e.g. in_review) without a migration.

CREATE TABLE IF NOT EXISTS landing_pages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id TEXT NOT NULL DEFAULT '',
    campaign_name TEXT NOT NULL,
    persona TEXT NOT NULL DEFAULT '',
    value_prop TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL,
    slug TEXT NOT NULL,
    hero JSONB NOT NULL DEFAULT '{}'::jsonb,
    sections JSONB NOT NULL DEFAULT '[]'::jsonb,
    cta JSONB NOT NULL DEFAULT '{}'::jsonb,
    meta JSONB NOT NULL DEFAULT '{}'::jsonb,
    reference_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    status TEXT NOT NULL DEFAULT 'draft',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_landing_pages_campaign
    ON landing_pages (account_id, campaign_name, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_landing_pages_status
    ON landing_pages (account_id, status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_landing_pages_slug
    ON landing_pages (account_id, slug);
