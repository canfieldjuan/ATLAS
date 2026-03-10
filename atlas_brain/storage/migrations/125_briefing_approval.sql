-- Add pending_approval status and approval tracking to vendor briefings

ALTER TABLE b2b_vendor_briefings
    DROP CONSTRAINT IF EXISTS b2b_vendor_briefings_status_check;

ALTER TABLE b2b_vendor_briefings
    ADD CONSTRAINT b2b_vendor_briefings_status_check
    CHECK (status IN ('pending_approval', 'sent', 'opened', 'clicked', 'bounced', 'failed', 'suppressed', 'rejected'));

ALTER TABLE b2b_vendor_briefings
    ADD COLUMN IF NOT EXISTS briefing_html TEXT,
    ADD COLUMN IF NOT EXISTS approved_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS rejected_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS reject_reason TEXT,
    ADD COLUMN IF NOT EXISTS target_mode TEXT DEFAULT 'vendor_retention';

CREATE INDEX IF NOT EXISTS idx_b2b_vendor_briefings_status
    ON b2b_vendor_briefings (status);
