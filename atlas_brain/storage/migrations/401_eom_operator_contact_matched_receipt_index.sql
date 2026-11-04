-- Keep the operator contact receipt lookup index-backed once it also reads
-- contact_matched receipts.
--
-- mutate_eom_operator_contact_atomic probes eom_lead_lifecycle_events by
-- operation_key for every receipt type in EOM_OPERATOR_CONTACT_EVENT_TYPES.
-- Migration 364 built that lookup's index as partial to contact_created and
-- contact_updated. A create-or-return sent with onMatch="return" records a
-- third receipt, contact_matched, when it matches an existing contact and
-- leaves it unchanged. A predicate that omits it cannot serve a lookup that
-- asks for it, so without this migration every operator mutation would scan
-- the append-only ledger.
--
-- Same name, same columns, widened predicate. The drop is first and
-- concurrent so a canceled earlier attempt that left an INVALID entry is
-- rebuilt rather than skipped (same pattern as 357/359/362/364).
--
-- Rollback evidence (restores migration 364's index exactly):
--   DROP INDEX CONCURRENTLY IF EXISTS idx_eom_lead_lifecycle_operator_contact_operation_key;
--   CREATE INDEX CONCURRENTLY idx_eom_lead_lifecycle_operator_contact_operation_key
--       ON eom_lead_lifecycle_events (operation_key, contact_id, event_type)
--       WHERE operation_key IS NOT NULL
--         AND event_type IN ('contact_created', 'contact_updated');
--
-- Roll-forward safety:
--   Index-only. It changes no table, no lifecycle event uniqueness, no
--   append-only or trigger behavior, and no lead or customer state.

DROP INDEX CONCURRENTLY IF EXISTS idx_eom_lead_lifecycle_operator_contact_operation_key;

CREATE INDEX CONCURRENTLY idx_eom_lead_lifecycle_operator_contact_operation_key
    ON eom_lead_lifecycle_events (operation_key, contact_id, event_type)
    WHERE operation_key IS NOT NULL
      AND event_type IN ('contact_created', 'contact_updated', 'contact_matched');
