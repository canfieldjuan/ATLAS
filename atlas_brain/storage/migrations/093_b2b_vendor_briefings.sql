-- Vendor intelligence briefing delivery tracking.
-- Stores sent briefings for history, analytics, and deduplication.

CREATE TABLE IF NOT EXISTS b2b_vendor_briefings (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vendor_name     TEXT NOT NULL,
    recipient_email TEXT NOT NULL,
    subject         TEXT NOT NULL,
    briefing_data   JSONB NOT NULL DEFAULT '{}',
    resend_id       TEXT,
    status          TEXT NOT NULL DEFAULT 'sent'
        CHECK (status IN ('sent', 'opened', 'clicked', 'bounced', 'failed')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_b2b_vendor_briefings_vendor
    ON b2b_vendor_briefings (vendor_name);

CREATE INDEX IF NOT EXISTS idx_b2b_vendor_briefings_created
    ON b2b_vendor_briefings (created_at DESC);
