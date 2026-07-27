-- EOM lead review queue keyset index.
--
-- The office review route reads active Effingham lead/new contacts newest
-- first and continues by (created_at, id). Keep this as a standalone
-- concurrent index migration so production request handling is not blocked by
-- a regular table-locking index build.

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_contacts_eom_lead_review_queue
    ON contacts (created_at DESC, id DESC)
    WHERE business_context_id = 'effingham_maids'
      AND status = 'active'
      AND contact_type = 'lead'
      AND lead_stage = 'new';
