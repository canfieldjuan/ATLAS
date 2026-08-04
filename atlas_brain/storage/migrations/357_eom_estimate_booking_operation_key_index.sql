-- Give estimate-booking operation-key ownership checks a leading-key access path.
--
-- Rollback evidence:
--   DROP INDEX CONCURRENTLY IF EXISTS idx_eom_lead_lifecycle_booking_operation_key;
--
-- Roll-forward safety:
--   This index is additive and partial. It does not change lifecycle event
--   uniqueness, append-only behavior, trigger behavior, or lead/customer state.

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_eom_lead_lifecycle_booking_operation_key
    ON eom_lead_lifecycle_events (operation_key, contact_id, event_type)
    WHERE operation_key IS NOT NULL
      AND event_type IN (
          'estimate_booking_requested',
          'estimate_booking_calendar_failed',
          'estimate_booking_calendar_ambiguous',
          'estimate_booked'
      );
