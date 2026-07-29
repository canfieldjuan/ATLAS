-- EOM lead review queue keyset index for all approval-reachable lead stages.
--
-- The review route now keeps active lead/estimate_booked rows visible so Juan
-- can approve a lead after the estimate booking command completes. Build the
-- widened replacement concurrently under a temporary name first, then retire
-- and rename over the earlier lead/new-only partial index, so old replicas do
-- not lose their usable live index during the replacement build.

-- A failed CREATE INDEX CONCURRENTLY can leave an invalid relation behind; drop
-- only the temporary replacement name before retrying, while the live old index
-- remains available to existing replicas.
DROP INDEX CONCURRENTLY IF EXISTS idx_contacts_eom_lead_review_queue_booked;

CREATE INDEX CONCURRENTLY idx_contacts_eom_lead_review_queue_booked
    ON contacts (created_at DESC, id DESC)
    WHERE business_context_id = 'effingham_maids'
      AND status = 'active'
      AND contact_type = 'lead'
      AND lead_stage IN ('new', 'estimate_booked');

DROP INDEX CONCURRENTLY IF EXISTS idx_contacts_eom_lead_review_queue;

ALTER INDEX IF EXISTS idx_contacts_eom_lead_review_queue_booked
    RENAME TO idx_contacts_eom_lead_review_queue;
