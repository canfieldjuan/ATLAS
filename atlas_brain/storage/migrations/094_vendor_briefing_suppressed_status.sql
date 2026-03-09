-- Add 'suppressed' status to vendor briefings.
-- Required for suppression-aware batch sending.

ALTER TABLE b2b_vendor_briefings
    DROP CONSTRAINT IF EXISTS b2b_vendor_briefings_status_check;

ALTER TABLE b2b_vendor_briefings
    ADD CONSTRAINT b2b_vendor_briefings_status_check
    CHECK (status IN ('sent', 'opened', 'clicked', 'bounced', 'failed', 'suppressed'));
