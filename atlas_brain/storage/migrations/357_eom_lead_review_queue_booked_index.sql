-- EOM lead review queue keyset index for all approval-reachable lead stages.
--
-- The review route now keeps active lead/estimate_booked rows visible so Juan
-- can approve a lead after the estimate booking command completes. Replace the
-- earlier lead/new-only partial index concurrently so production request
-- handling is not blocked by a regular index build.

DROP INDEX CONCURRENTLY IF EXISTS idx_contacts_eom_lead_review_queue;

CREATE INDEX CONCURRENTLY idx_contacts_eom_lead_review_queue
    ON contacts (created_at DESC, id DESC)
    WHERE business_context_id = 'effingham_maids'
      AND status = 'active'
      AND contact_type = 'lead'
      AND lead_stage IN ('new', 'estimate_booked');
