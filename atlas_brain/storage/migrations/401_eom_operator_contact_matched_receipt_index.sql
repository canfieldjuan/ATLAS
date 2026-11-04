-- atlas: atomic-bookkeeping
-- Keep the operator contact receipt lookup index-backed once it also reads
-- contact_matched receipts.
--
-- mutate_eom_operator_contact_atomic probes eom_lead_lifecycle_events by
-- operation_key for every receipt type in EOM_OPERATOR_CONTACT_EVENT_TYPES.
-- Migration 364 built that lookup's index as partial to contact_created and
-- contact_updated. A create-or-return sent with onMatch="return" records a
-- third receipt, contact_matched, when it matches an existing contact and
-- leaves it unchanged. A predicate that omits it cannot serve a lookup that
-- asks for it, so until this runs the lookup scans the ledger. That is correct
-- but unindexed.
--
-- Controlled DBA-only migration (CONTROLLED_DBA_MIGRATION_NAMES): migration 394
-- gave eom_lead_lifecycle_events to the no-login guard owner
-- atlas_eom_handoff_owner, so the runtime atlas role can neither drop nor
-- create an index on it, and startup must not attempt this file. Apply it with
--   ./ops db controlled eom-operator-contact-receipt-index apply
-- Not CONCURRENTLY: the controlled runner commits the DDL and its ledger row as
-- one transaction, and the ledger is small, so the rebuild's write lock is
-- brief.
--
-- Rollback evidence (restores migration 364's index exactly), as a DBA:
--   DROP INDEX IF EXISTS idx_eom_lead_lifecycle_operator_contact_operation_key;
--   CREATE INDEX idx_eom_lead_lifecycle_operator_contact_operation_key
--       ON eom_lead_lifecycle_events (operation_key, contact_id, event_type)
--       WHERE operation_key IS NOT NULL
--         AND event_type IN ('contact_created', 'contact_updated');

DO $$
BEGIN
    IF NOT COALESCE(
        (SELECT role.rolsuper FROM pg_roles AS role WHERE role.rolname = current_user),
        FALSE
    ) THEN
        RAISE EXCEPTION
            'database administrator must run 401_eom_operator_contact_matched_receipt_index';
    END IF;
END;
$$;

DROP INDEX IF EXISTS idx_eom_lead_lifecycle_operator_contact_operation_key;

CREATE INDEX idx_eom_lead_lifecycle_operator_contact_operation_key
    ON eom_lead_lifecycle_events (operation_key, contact_id, event_type)
    WHERE operation_key IS NOT NULL
      AND event_type IN ('contact_created', 'contact_updated', 'contact_matched');
