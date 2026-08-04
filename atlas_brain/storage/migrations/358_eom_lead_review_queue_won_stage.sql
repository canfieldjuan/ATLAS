-- Keep the EOM lead review queue indexed after the first cleaning is booked.
--
-- A won lead (first cleaning booked) is still a lead until the office
-- explicitly approves the customer/site handoff, so the review queue now
-- reads lead/new, lead/estimate_booked, and lead/won records.
--
-- The drop is intentionally concurrent and first: if a prior canceled
-- startup left an INVALID same-named catalog entry, CREATE INDEX IF NOT
-- EXISTS would skip rebuilding it and the migration ledger would record a
-- broken index as applied. The drop-then-recreate pair keeps replays safe
-- (same pattern as migrations 355/356/357).
--
-- Rollback evidence:
--   DROP INDEX CONCURRENTLY IF EXISTS idx_contacts_eom_lead_review_queue;
--   CREATE INDEX CONCURRENTLY idx_contacts_eom_lead_review_queue
--       ON contacts (created_at DESC, id DESC)
--       WHERE business_context_id = 'effingham_maids'
--         AND status = 'active'
--         AND contact_type = 'lead'
--         AND lead_stage IN ('new', 'estimate_booked');
-- Roll-forward safety: if application code is reverted while this widened
-- predicate remains, old code still filters lead_stage at query time; the
-- wider partial index may be less selective, but it does not widen the old
-- review queue result set.

DROP INDEX CONCURRENTLY IF EXISTS idx_contacts_eom_lead_review_queue;

CREATE INDEX CONCURRENTLY idx_contacts_eom_lead_review_queue
    ON contacts (created_at DESC, id DESC)
    WHERE business_context_id = 'effingham_maids'
      AND status = 'active'
      AND contact_type = 'lead'
      AND lead_stage IN ('new', 'estimate_booked', 'won');
