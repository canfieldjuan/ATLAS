-- atlas: atomic-bookkeeping
-- Explicit, append-only recovery evidence for a commercial Gmail draft that
-- reconciliation proved missing. The existing root row remains one-per-
-- approval/PDF so a prior provider revision can continue to read it after a
-- rollback; this table retains the prior generation before the root becomes
-- the next current delivery attempt.
--
-- This never sends mail or changes invoice lifecycle state. Sent-mail proof
-- remains the responsibility of the reconciliation service.
--
-- Rollback: stop the replacement route/service and retain the event rows and
-- current root generation. The prior application revision still sees one root
-- draft record per approval/PDF and fails closed rather than creating a
-- duplicate because it cannot derive a later generation's RFC Message-ID.

ALTER TABLE commercial_billing_invoice_gmail_drafts
    ADD COLUMN IF NOT EXISTS draft_generation INTEGER NOT NULL DEFAULT 1,
    ADD COLUMN IF NOT EXISTS last_replaced_by VARCHAR(128),
    ADD COLUMN IF NOT EXISTS last_replaced_at TIMESTAMPTZ;

ALTER TABLE commercial_billing_invoice_gmail_drafts
    ADD CONSTRAINT commercial_billing_invoice_gmail_drafts_generation_check
        CHECK (draft_generation > 0),
    ADD CONSTRAINT commercial_billing_invoice_gmail_drafts_last_replacement_check
        CHECK (
            (last_replaced_by IS NULL AND last_replaced_at IS NULL)
            OR (
                length(btrim(COALESCE(last_replaced_by, ''))) > 0
                AND last_replaced_at IS NOT NULL
            )
        );

-- Every ordinary create/recover/reuse key is also bound to the exact current
-- identity generation.  Before H-15 every draft operation necessarily refers
-- to generation 1, so the default backfills old rows without changing them.
ALTER TABLE commercial_billing_invoice_gmail_draft_operations
    ADD COLUMN IF NOT EXISTS draft_generation INTEGER NOT NULL DEFAULT 1,
    ADD CONSTRAINT commercial_billing_invoice_gmail_draft_operations_generation_check
        CHECK (draft_generation > 0);

-- A completed reconciliation is an observation of one exact current identity.
-- Before H-15 there was only generation 1, so this default safely backfills
-- deployed operations while subsequent inserts persist their observed generation.
ALTER TABLE commercial_billing_gmail_sent_reconciliation_operations
    ADD COLUMN IF NOT EXISTS draft_generation INTEGER NOT NULL DEFAULT 1,
    ADD CONSTRAINT commercial_billing_gmail_sent_reconciliation_operations_generation_check
        CHECK (draft_generation > 0);

-- A newer completed observation can safely retire an abandoned pending
-- observation of the same durable identity.  The original idempotency row is
-- retained with its authenticated supersession actor/timestamp; a replay of it
-- fails closed instead of blocking missing-draft recovery forever.
ALTER TABLE commercial_billing_gmail_sent_reconciliation_operations
    ADD COLUMN IF NOT EXISTS superseded_by VARCHAR(128),
    ADD COLUMN IF NOT EXISTS superseded_at TIMESTAMPTZ;

ALTER TABLE commercial_billing_gmail_sent_reconciliation_operations
    DROP CONSTRAINT IF EXISTS cb_gmail_sent_recon_ops_state_check,
    DROP CONSTRAINT IF EXISTS cb_gmail_sent_recon_ops_outcome_check,
    ADD CONSTRAINT cb_gmail_sent_recon_ops_state_check
        CHECK (state IN ('pending', 'completed', 'superseded')),
    ADD CONSTRAINT cb_gmail_sent_recon_ops_outcome_check
        CHECK (
            (state = 'pending'
                AND outcome_state IS NULL
                AND completed_at IS NULL
                AND superseded_by IS NULL
                AND superseded_at IS NULL)
            OR (state = 'completed'
                AND outcome_state IN ('draft_present', 'draft_missing', 'sent_confirmed')
                AND completed_at IS NOT NULL
                AND superseded_by IS NULL
                AND superseded_at IS NULL)
            OR (state = 'superseded'
                AND outcome_state IS NULL
                AND completed_at IS NULL
                AND length(btrim(COALESCE(superseded_by, ''))) > 0
                AND superseded_at IS NOT NULL)
        );

CREATE TABLE commercial_billing_invoice_gmail_draft_replacement_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    gmail_draft_record_id UUID NOT NULL
        REFERENCES commercial_billing_invoice_gmail_drafts(id) ON DELETE RESTRICT,
    operation_id UUID NOT NULL UNIQUE
        REFERENCES commercial_billing_invoice_gmail_draft_operations(id) ON DELETE RESTRICT,
    prior_generation INTEGER NOT NULL,
    replacement_generation INTEGER NOT NULL,
    prior_snapshot JSONB NOT NULL,
    replaced_by VARCHAR(128) NOT NULL,
    replaced_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT commercial_billing_gmail_draft_replacement_generation_check
        CHECK (
            prior_generation > 0
            AND replacement_generation = prior_generation + 1
        ),
    CONSTRAINT commercial_billing_gmail_draft_replacement_snapshot_check
        CHECK (jsonb_typeof(prior_snapshot) = 'object'),
    CONSTRAINT commercial_billing_gmail_draft_replacement_actor_check
        CHECK (length(btrim(replaced_by)) > 0),
    CONSTRAINT commercial_billing_gmail_draft_replacement_once_per_generation
        UNIQUE (gmail_draft_record_id, replacement_generation)
);

CREATE INDEX idx_commercial_billing_gmail_draft_replacement_events_record
    ON commercial_billing_invoice_gmail_draft_replacement_events (
        gmail_draft_record_id, replacement_generation DESC
    );

COMMENT ON TABLE commercial_billing_invoice_gmail_draft_replacement_events IS
    'Append-only snapshots of reconciliation-proven missing commercial Gmail draft generations replaced by an explicit operator action.';

COMMENT ON COLUMN commercial_billing_invoice_gmail_drafts.draft_generation IS
    'Current Gmail draft identity generation; prior identities are retained in replacement events.';

-- A replacement commits its new intent before the external Gmail Drafts call.
-- Every invoice writer therefore shares the replacement service's approval lock
-- and rejects while this particular replacement generation remains unresolved.
-- The trigger never changes invoice lifecycle state; it only prevents a legacy
-- writer from bypassing the committed no-send delivery intent with stale fields.
CREATE OR REPLACE FUNCTION commercial_billing_reject_invoice_mutation_while_gmail_replacement_pending()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    approval_row RECORD;
BEGIN
    FOR approval_row IN
        SELECT id
        FROM commercial_billing_candidate_approvals
        WHERE invoice_id = OLD.id
        ORDER BY id
    LOOP
        PERFORM pg_advisory_xact_lock(
            hashtextextended(
                'commercial-billing-invoice-gmail-draft:approval:' || approval_row.id::text,
                0
            )
        );
    END LOOP;

    IF EXISTS (
        SELECT 1
        FROM commercial_billing_candidate_approvals AS approval
        JOIN commercial_billing_invoice_gmail_drafts AS draft
          ON draft.approval_id = approval.id
        JOIN commercial_billing_invoice_gmail_draft_replacement_events AS replacement
          ON replacement.gmail_draft_record_id = draft.id
         AND replacement.replacement_generation = draft.draft_generation
        WHERE approval.invoice_id = OLD.id
          AND draft.state IN ('creating', 'retryable', 'recovery_required')
    ) THEN
        RAISE EXCEPTION
            'Commercial billing invoice mutation is blocked while a Gmail draft replacement is pending'
            USING ERRCODE = '23514';
    END IF;

    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS commercial_billing_reject_invoice_mutation_while_gmail_replacement_pending
    ON invoices;

CREATE TRIGGER commercial_billing_reject_invoice_mutation_while_gmail_replacement_pending
BEFORE UPDATE ON invoices
FOR EACH ROW
EXECUTE FUNCTION commercial_billing_reject_invoice_mutation_while_gmail_replacement_pending();
