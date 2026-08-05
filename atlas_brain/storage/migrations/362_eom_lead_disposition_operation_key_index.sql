-- Cover lost/reopen disposition events in an operation-key-leading index.
--
-- mark_eom_lead_lost / reopen_eom_lead reject Idempotency-Key reuse across
-- contacts by probing eom_lead_lifecycle_events for a matching operation_key
-- on a *different* contact (event_type 'lead_lost' / 'lead_reopened'). The
-- lifecycle unique index is contact-led (351:26-28) so it cannot serve an
-- operation_key-leading lookup, and the existing operation-key index is
-- partial to the eight booking event types (359), which excludes the two
-- disposition events. Without this index every lost/reopen request scans the
-- whole ledger just to reject key reuse once the table has production history.
--
-- The drop is intentionally concurrent and first: if a prior canceled startup
-- left an INVALID same-named catalog entry, CREATE INDEX IF NOT EXISTS would
-- skip rebuilding it and the migration ledger would record a broken index as
-- applied. The drop-then-recreate pair keeps replays safe (same pattern as
-- migrations 355/356/357/359).
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
--   uniqueness, append-only behavior, trigger behavior, or lead/customer
--   state. It only accelerates the cross-contact key-ownership probe the
--   disposition endpoints already perform.

DROP INDEX CONCURRENTLY IF EXISTS idx_eom_lead_lifecycle_disposition_operation_key;

CREATE INDEX CONCURRENTLY idx_eom_lead_lifecycle_disposition_operation_key
    ON eom_lead_lifecycle_events (operation_key, contact_id, event_type)
    WHERE operation_key IS NOT NULL
      AND event_type IN ('lead_lost', 'lead_reopened');
