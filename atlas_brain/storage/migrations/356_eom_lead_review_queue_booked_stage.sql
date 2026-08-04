-- Keep the EOM lead review queue indexed after estimate booking.
--
-- Booked estimates remain leads until the office explicitly approves the
-- customer/site handoff, so the review queue now reads both lead/new and
-- lead/estimate_booked records.
--
-- Rollback evidence:
--   DROP INDEX CONCURRENTLY IF EXISTS idx_contacts_eom_lead_review_queue;
--   CREATE INDEX CONCURRENTLY idx_contacts_eom_lead_review_queue
--       ON contacts (created_at DESC, id DESC)
--       WHERE business_context_id = 'effingham_maids'
--         AND status = 'active'
--         AND contact_type = 'lead'
--         AND lead_stage = 'new';
-- Roll-forward safety: if application code is reverted while this widened
-- predicate remains, old code still filters lead_stage = 'new' at query time;
-- the wider partial index may be less selective, but it does not widen the
-- old review queue result set.

DROP INDEX CONCURRENTLY IF EXISTS idx_contacts_eom_lead_review_queue;

CREATE INDEX CONCURRENTLY idx_contacts_eom_lead_review_queue
    ON contacts (created_at DESC, id DESC)
    WHERE business_context_id = 'effingham_maids'
      AND status = 'active'
      AND contact_type = 'lead'
      AND lead_stage IN ('new', 'estimate_booked');
