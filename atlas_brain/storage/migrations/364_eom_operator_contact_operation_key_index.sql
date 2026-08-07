-- Cover operator contact mutation receipts in an operation-key-leading index.
--
-- mutate_eom_operator_contact_atomic probes eom_lead_lifecycle_events by
-- operation_key before creating or updating a contact so idempotent replays and
-- key conflicts resolve before a second mutation is attempted. The lifecycle
-- unique index is contact-led (351), and the booking/disposition access-path
-- indexes are partial to their own event families, so contact_created and
-- contact_updated would otherwise scan the append-only ledger as history grows.
--
-- The drop is intentionally concurrent and first: if a prior canceled startup
-- left an INVALID same-named catalog entry, CREATE INDEX IF NOT EXISTS would
-- skip rebuilding it and the migration ledger would record a broken index as
-- applied. The drop-then-recreate pair keeps replays safe (same pattern as
-- migrations 357/359/362).
--
-- Rollback evidence:
--   DROP INDEX CONCURRENTLY IF EXISTS idx_eom_lead_lifecycle_operator_contact_operation_key;
--
-- Roll-forward safety:
--   This index is additive and partial. It does not change lifecycle event
--   uniqueness, append-only behavior, trigger behavior, or lead/customer
--   state. It only accelerates the operator contact receipt lookup.

DROP INDEX CONCURRENTLY IF EXISTS idx_eom_lead_lifecycle_operator_contact_operation_key;

CREATE INDEX CONCURRENTLY idx_eom_lead_lifecycle_operator_contact_operation_key
    ON eom_lead_lifecycle_events (operation_key, contact_id, event_type)
    WHERE operation_key IS NOT NULL
      AND event_type IN ('contact_created', 'contact_updated');
