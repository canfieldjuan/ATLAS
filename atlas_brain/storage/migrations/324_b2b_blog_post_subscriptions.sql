-- B2B blog post subscriptions: per-tenant opt-in for scheduled blueprint
-- fanout. The autonomous blog-post task (b2b_blog_post_generation) runs as a
-- single-host cron; without per-tenant fanout, the blueprints it generates
-- are visible to no authenticated tenant via /api/v1/content-ops/execute.
--
-- This table mirrors b2b_report_subscriptions (migration 0265) -- same FK
-- shape, same enabled flag, same per-account uniqueness -- minus the
-- delivery-mechanic fields (recipient_emails, frequency, freshness_policy)
-- since blueprint fan-out is a write-side push (autonomous task ->
-- blog_blueprints), not a read-side delivery.
--
-- topic_types / target_modes are filter arrays; an empty array means "accept
-- all values". A subscription with empty filters receives every blueprint
-- the autonomous task generates.
--
-- PR-Subscriptions-2 wires the autonomous task to read this list and fan
-- out blueprints per matching subscription via
-- PostgresBlogBlueprintRepository.save_blueprints.

CREATE TABLE IF NOT EXISTS b2b_blog_post_subscriptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id UUID NOT NULL REFERENCES saas_accounts(id) ON DELETE CASCADE,
    topic_types TEXT[] NOT NULL DEFAULT '{}',
    target_modes TEXT[] NOT NULL DEFAULT '{}',
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (account_id)
);

CREATE INDEX IF NOT EXISTS idx_b2b_blog_post_subscriptions_enabled
    ON b2b_blog_post_subscriptions (enabled, account_id);
