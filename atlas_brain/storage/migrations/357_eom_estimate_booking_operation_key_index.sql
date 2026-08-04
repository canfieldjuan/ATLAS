-- Give estimate-booking operation-key ownership checks a leading-key access path.
--
-- The drop is intentionally concurrent and first: if a prior canceled startup
-- left an INVALID same-named catalog entry, CREATE INDEX IF NOT EXISTS would
-- skip rebuilding it and the migration ledger would record a broken index as
-- applied while the operation-key lookup stayed a full ledger scan. The
-- drop-then-recreate pair keeps replays safe (same pattern as migration 355).
--
-- Rollback evidence:
--   DROP INDEX CONCURRENTLY IF EXISTS idx_eom_lead_lifecycle_booking_operation_key;
--
-- Roll-forward safety:
--   This index is additive and partial. It does not change lifecycle event
--   uniqueness, append-only behavior, trigger behavior, or lead/customer state.

DROP INDEX CONCURRENTLY IF EXISTS idx_eom_lead_lifecycle_booking_operation_key;

CREATE INDEX CONCURRENTLY idx_eom_lead_lifecycle_booking_operation_key
    ON eom_lead_lifecycle_events (operation_key, contact_id, event_type)
    WHERE operation_key IS NOT NULL
      AND event_type IN (
          'estimate_booking_requested',
          'estimate_booking_calendar_failed',
          'estimate_booking_calendar_ambiguous',
          'estimate_booked'
      );
