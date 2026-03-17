-- Route hard unmatched prospect-org candidates into a manual/domain-assisted queue.
-- Migration 160

ALTER TABLE prospect_org_cache
    DROP CONSTRAINT IF EXISTS prospect_org_cache_status_check;

ALTER TABLE prospect_org_cache
    ADD CONSTRAINT prospect_org_cache_status_check
    CHECK (status IN ('pending', 'enriched', 'not_found', 'error', 'manual_review'));
