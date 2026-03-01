-- Campaign sequences: stateful B2B email sequence tracking
-- with engagement signals (opens, clicks, bounces, replies) and audit log.

-- =========================================================================
-- 1. campaign_sequences — one row per company+batch outreach sequence
-- =========================================================================

CREATE TABLE IF NOT EXISTS campaign_sequences (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_name        TEXT NOT NULL,
    batch_id            TEXT NOT NULL,
    partner_id          UUID,

    -- Context snapshot (frozen at creation)
    company_context     JSONB NOT NULL DEFAULT '{}',
    selling_context     JSONB NOT NULL DEFAULT '{}',

    -- Sequence state
    current_step        INT NOT NULL DEFAULT 1,
    max_steps           INT NOT NULL DEFAULT 4,
    status              TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'paused', 'completed', 'replied', 'bounced', 'unsubscribed')),

    -- Recipient (must be set before sending)
    recipient_email     TEXT,

    -- Engagement (updated by webhooks + email_intake)
    last_campaign_id    UUID,
    last_sent_at        TIMESTAMPTZ,
    last_opened_at      TIMESTAMPTZ,
    last_clicked_at     TIMESTAMPTZ,
    open_count          INT NOT NULL DEFAULT 0,
    click_count         INT NOT NULL DEFAULT 0,
    reply_received_at   TIMESTAMPTZ,
    reply_intent        TEXT,
    reply_summary       TEXT,
    bounce_type         TEXT,
    bounced_at          TIMESTAMPTZ,

    -- Scheduling
    next_step_after     TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_campaign_seq_company_batch
    ON campaign_sequences (LOWER(company_name), batch_id);
CREATE INDEX IF NOT EXISTS idx_campaign_seq_active_next
    ON campaign_sequences (next_step_after)
    WHERE status = 'active' AND next_step_after IS NOT NULL;


-- =========================================================================
-- 2. b2b_campaigns extensions — link campaigns to sequences + ESP tracking
-- =========================================================================

-- Create b2b_campaigns table if it does not exist yet (first-time setup)
CREATE TABLE IF NOT EXISTS b2b_campaigns (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_name    TEXT NOT NULL,
    batch_id        TEXT NOT NULL,
    partner_id      UUID,
    channel         TEXT NOT NULL DEFAULT 'email_cold',
    subject         TEXT,
    body            TEXT,
    status          TEXT NOT NULL DEFAULT 'draft'
        CHECK (status IN ('draft', 'approved', 'queued', 'sent', 'cancelled', 'expired')),
    approved_at     TIMESTAMPTZ,
    sent_at         TIMESTAMPTZ,
    opened_at       TIMESTAMPTZ,
    clicked_at      TIMESTAMPTZ,
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Add new columns for sequence tracking + ESP integration
ALTER TABLE b2b_campaigns
    ADD COLUMN IF NOT EXISTS sequence_id UUID REFERENCES campaign_sequences(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS step_number INT,
    ADD COLUMN IF NOT EXISTS recipient_email TEXT,
    ADD COLUMN IF NOT EXISTS sent_message_id TEXT,
    ADD COLUMN IF NOT EXISTS esp_message_id TEXT,
    ADD COLUMN IF NOT EXISTS from_email TEXT;

-- Extend status CHECK to include queued + cancelled
ALTER TABLE b2b_campaigns DROP CONSTRAINT IF EXISTS b2b_campaigns_status_check;
ALTER TABLE b2b_campaigns ADD CONSTRAINT b2b_campaigns_status_check
    CHECK (status IN ('draft', 'approved', 'queued', 'sent', 'cancelled', 'expired'));

CREATE INDEX IF NOT EXISTS idx_b2b_campaigns_sequence
    ON b2b_campaigns (sequence_id) WHERE sequence_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_b2b_campaigns_esp_msg
    ON b2b_campaigns (esp_message_id) WHERE esp_message_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_b2b_campaigns_sent_msg
    ON b2b_campaigns (sent_message_id) WHERE sent_message_id IS NOT NULL;


-- =========================================================================
-- 3. campaign_audit_log — immutable log of every state change (debugging)
-- =========================================================================

CREATE TABLE IF NOT EXISTS campaign_audit_log (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    campaign_id     UUID REFERENCES b2b_campaigns(id) ON DELETE SET NULL,
    sequence_id     UUID REFERENCES campaign_sequences(id) ON DELETE SET NULL,
    event_type      TEXT NOT NULL,
    step_number     INT,
    subject         TEXT,
    body            TEXT,
    recipient_email TEXT,
    esp_message_id  TEXT,
    error_detail    TEXT,
    source          TEXT NOT NULL DEFAULT 'system',
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_campaign_audit_sequence
    ON campaign_audit_log (sequence_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_campaign_audit_campaign
    ON campaign_audit_log (campaign_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_campaign_audit_type
    ON campaign_audit_log (event_type, created_at DESC);
