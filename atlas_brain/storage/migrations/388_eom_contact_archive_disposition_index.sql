-- Extend the disposition operation-key index to contact archive/restore.
--
-- archive_eom_contact / restore_eom_contact reject Idempotency-Key reuse
-- across contacts by probing eom_lead_lifecycle_events for a matching
-- operation_key on a *different* contact (event_type 'contact_archived' /
-- 'contact_restored'), exactly like the lost/reopen disposition pair. The
-- existing disposition index (migration 362) is partial to the two lead
-- disposition event types, so the new probes would scan the whole ledger
-- once the table has production history. Re-creating the same index with the
-- two new event types keeps one operation_key-leading index serving all four
-- disposition probes.
--
-- The drop is intentionally concurrent and first: if a prior canceled startup
-- left an INVALID same-named catalog entry, CREATE INDEX IF NOT EXISTS would
-- skip rebuilding it and the migration ledger would record a broken index as
-- applied. The drop-then-recreate pair keeps replays safe (same pattern as
-- migrations 355/356/357/359/362).
--
-- Rollback evidence:
--   DROP INDEX CONCURRENTLY IF EXISTS idx_eom_lead_lifecycle_disposition_operation_key;
--   CREATE INDEX CONCURRENTLY idx_eom_lead_lifecycle_disposition_operation_key
--       ON eom_lead_lifecycle_events (operation_key, contact_id, event_type)
--       WHERE operation_key IS NOT NULL
--         AND event_type IN ('lead_lost', 'lead_reopened');
--
-- Roll-forward safety:
--   This index is additive and partial. It does not change lifecycle event
--   uniqueness, append-only behavior, trigger behavior, or contact state. It
--   only widens the event-type set whose cross-contact key-ownership probe
--   the disposition endpoints accelerate. The lost/reopen probes keep their
--   coverage because the widened predicate is a superset of the old one.

DROP INDEX CONCURRENTLY IF EXISTS idx_eom_lead_lifecycle_disposition_operation_key;

CREATE INDEX CONCURRENTLY idx_eom_lead_lifecycle_disposition_operation_key
    ON eom_lead_lifecycle_events (operation_key, contact_id, event_type)
    WHERE operation_key IS NOT NULL
      AND event_type IN (
          'lead_lost',
          'lead_reopened',
          'contact_archived',
          'contact_restored'
      );
