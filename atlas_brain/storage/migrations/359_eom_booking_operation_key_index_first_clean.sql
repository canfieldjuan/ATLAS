-- Cover first-clean booking events in the operation-key ownership index.
--
-- Booking-key ownership checks now scan both booking families (estimate and
-- first clean) on every prepare, so the partial index predicate must include
-- the four first-clean event types or cross-family key lookups fall back to
-- a full ledger scan.
--
-- The drop is intentionally concurrent and first: if a prior canceled
-- startup left an INVALID same-named catalog entry, CREATE INDEX IF NOT
-- EXISTS would skip rebuilding it and the migration ledger would record a
-- broken index as applied. The drop-then-recreate pair keeps replays safe
-- (same pattern as migrations 355/356/357).
--
-- Rollback evidence:
--   DROP INDEX CONCURRENTLY IF EXISTS idx_eom_lead_lifecycle_booking_operation_key;
--   CREATE INDEX CONCURRENTLY idx_eom_lead_lifecycle_booking_operation_key
--       ON eom_lead_lifecycle_events (operation_key, contact_id, event_type)
--       WHERE operation_key IS NOT NULL
--         AND event_type IN (
--             'estimate_booking_requested',
--             'estimate_booking_calendar_failed',
--             'estimate_booking_calendar_ambiguous',
--             'estimate_booked'
--         );
--
-- Roll-forward safety:
--   This index is additive and partial. It does not change lifecycle event
--   uniqueness, append-only behavior, trigger behavior, or lead/customer
--   state. Old code querying only estimate event types still matches the
--   widened predicate.

DROP INDEX CONCURRENTLY IF EXISTS idx_eom_lead_lifecycle_booking_operation_key;

CREATE INDEX CONCURRENTLY idx_eom_lead_lifecycle_booking_operation_key
    ON eom_lead_lifecycle_events (operation_key, contact_id, event_type)
    WHERE operation_key IS NOT NULL
      AND event_type IN (
          'estimate_booking_requested',
          'estimate_booking_calendar_failed',
          'estimate_booking_calendar_ambiguous',
          'estimate_booked',
          'first_clean_booking_requested',
          'first_clean_booking_calendar_failed',
          'first_clean_booking_calendar_ambiguous',
          'first_clean_booked'
      );
