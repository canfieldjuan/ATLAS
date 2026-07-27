-- EOM lead review queue keyset index.
--
-- The office review route reads active Effingham lead/new contacts newest
-- first and continues by (created_at, id). Keep this as a standalone
-- concurrent index migration so production request handling is not blocked by
-- a regular table-locking index build. The drop is intentionally concurrent and
-- first: if a prior canceled startup left an invalid same-named catalog entry,
-- CREATE INDEX IF NOT EXISTS would skip rebuilding it and the migration ledger
-- would record a broken index as applied.

DROP INDEX CONCURRENTLY IF EXISTS idx_contacts_eom_lead_review_queue;

CREATE INDEX CONCURRENTLY idx_contacts_eom_lead_review_queue
    ON contacts (created_at DESC, id DESC)
    WHERE business_context_id = 'effingham_maids'
      AND status = 'active'
      AND contact_type = 'lead'
      AND lead_stage = 'new';
