-- Global email/domain suppression list for B2B campaigns.
-- Prevents sending to hard-bounced, complained, or manually blocked addresses.

CREATE TABLE IF NOT EXISTS campaign_suppressions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email       TEXT,            -- exact email (NULL if domain-level)
    domain      TEXT,            -- entire domain blocked (NULL if email-level)
    reason      TEXT NOT NULL,   -- bounce_hard, bounce_soft, complaint, manual, unsubscribe
    source      TEXT NOT NULL DEFAULT 'system',  -- webhook, api, manual, import
    campaign_id UUID REFERENCES b2b_campaigns(id) ON DELETE SET NULL,
    notes       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at  TIMESTAMPTZ,     -- NULL = permanent. Soft bounces expire after 30 days
    CONSTRAINT chk_email_or_domain CHECK (email IS NOT NULL OR domain IS NOT NULL)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_suppressions_email
    ON campaign_suppressions (LOWER(email)) WHERE email IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_suppressions_domain
    ON campaign_suppressions (LOWER(domain)) WHERE domain IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_suppressions_reason
    ON campaign_suppressions (reason, created_at DESC);
