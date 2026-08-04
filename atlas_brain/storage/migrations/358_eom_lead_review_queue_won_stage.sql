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
--
-- Application rollback (data step, run only if reverting to pre-won code):
-- old application code admits only 'new'/'estimate_booked' to the review
-- queue and customer handoff, so persisted won leads would be invisible and
-- unfinalizable until roll-forward. To keep them operable under old code:
--   UPDATE contacts SET lead_stage = 'estimate_booked', updated_at = NOW()
--    WHERE business_context_id = 'effingham_maids'
--      AND contact_type = 'lead'
--      AND lead_stage = 'won';
-- This touches only mutable stage state; the append-only lifecycle ledger
-- keeps the first_clean_booked evidence, prepare replays for the completed
-- first-clean key still return the booked outcome, and the booked-operation
-- guard still refuses a second first-clean booking for those leads. On
-- roll-forward no reverse step is needed: handoff admits 'estimate_booked'
-- directly, or the office re-runs the first-clean completion replay.

DROP INDEX CONCURRENTLY IF EXISTS idx_contacts_eom_lead_review_queue;

CREATE INDEX CONCURRENTLY idx_contacts_eom_lead_review_queue
    ON contacts (created_at DESC, id DESC)
    WHERE business_context_id = 'effingham_maids'
      AND status = 'active'
      AND contact_type = 'lead'
      AND lead_stage IN ('new', 'estimate_booked', 'won');
