-- Durable sent-mail reconciliation for a manually sent EOM commercial Gmail draft.
--
-- Gmail deletes a draft when an operator sends it and returns a different sent
-- message identity.  The durable draft record therefore retains the stable RFC
-- Message-ID and stores only a verified Sent-mail outcome.  A missing draft is
-- explicitly distinguishable from a confirmed sent message.  The service, not
-- this migration, performs the proof-gated invoice lifecycle update inside the
-- same PostgreSQL transaction that records the confirmation.
--
-- Rollback: stop the reconciliation route/service and retain these audit rows.
-- Removing sent-mail identities or reconciliation evidence is a separately
-- authorized destructive retention action, never a mixed-version rollback.

ALTER TABLE commercial_billing_invoice_gmail_drafts
    ADD COLUMN IF NOT EXISTS reconciliation_state VARCHAR(32)
        NOT NULL DEFAULT 'not_reconciled',
    ADD COLUMN IF NOT EXISTS gmail_sent_message_id VARCHAR(256),
    ADD COLUMN IF NOT EXISTS gmail_sent_thread_id VARCHAR(256),
    ADD COLUMN IF NOT EXISTS gmail_sent_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS sent_reconciled_by VARCHAR(128),
    ADD COLUMN IF NOT EXISTS sent_reconciled_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS last_reconciled_by VARCHAR(128),
    ADD COLUMN IF NOT EXISTS last_reconciled_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS draft_missing_by VARCHAR(128),
    ADD COLUMN IF NOT EXISTS draft_missing_at TIMESTAMPTZ;

ALTER TABLE commercial_billing_invoice_gmail_drafts
    ADD CONSTRAINT commercial_billing_invoice_gmail_drafts_reconciliation_state_check
        CHECK (
            reconciliation_state IN (
                'not_reconciled', 'draft_present', 'draft_missing', 'sent_confirmed'
            )
        ),
    ADD CONSTRAINT commercial_billing_invoice_gmail_drafts_sent_identity_check
        CHECK (
            (
                gmail_sent_message_id IS NULL
                AND gmail_sent_thread_id IS NULL
                AND gmail_sent_at IS NULL
                AND sent_reconciled_by IS NULL
                AND sent_reconciled_at IS NULL
            )
            OR (
                length(btrim(COALESCE(gmail_sent_message_id, ''))) > 0
                AND length(btrim(COALESCE(gmail_sent_thread_id, ''))) > 0
                AND gmail_sent_at IS NOT NULL
                AND length(btrim(COALESCE(sent_reconciled_by, ''))) > 0
                AND sent_reconciled_at IS NOT NULL
            )
        ),
    ADD CONSTRAINT commercial_billing_invoice_gmail_drafts_sent_state_check
        CHECK (
            reconciliation_state = 'sent_confirmed'
            OR (
                gmail_sent_message_id IS NULL
                AND gmail_sent_thread_id IS NULL
                AND gmail_sent_at IS NULL
                AND sent_reconciled_by IS NULL
                AND sent_reconciled_at IS NULL
            )
        ),
    ADD CONSTRAINT commercial_billing_invoice_gmail_drafts_reconciled_attempt_check
        CHECK (
            (last_reconciled_by IS NULL AND last_reconciled_at IS NULL)
            OR (
                length(btrim(COALESCE(last_reconciled_by, ''))) > 0
                AND last_reconciled_at IS NOT NULL
            )
        ),
    ADD CONSTRAINT commercial_billing_invoice_gmail_drafts_missing_draft_check
        CHECK (
            (draft_missing_by IS NULL AND draft_missing_at IS NULL)
            OR (
                length(btrim(COALESCE(draft_missing_by, ''))) > 0
                AND draft_missing_at IS NOT NULL
            )
        );

CREATE TABLE IF NOT EXISTS commercial_billing_gmail_sent_reconciliation_operations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    gmail_draft_record_id UUID NOT NULL
        REFERENCES commercial_billing_invoice_gmail_drafts(id) ON DELETE RESTRICT,
    source VARCHAR(32) NOT NULL DEFAULT 'eom_admin',
    idempotency_key VARCHAR(128) NOT NULL,
    request_fingerprint VARCHAR(64) NOT NULL,
    state VARCHAR(32) NOT NULL DEFAULT 'pending',
    outcome_state VARCHAR(32),
    requested_by VARCHAR(128) NOT NULL,
    requested_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT cb_gmail_sent_recon_ops_source_key
        UNIQUE (source, idempotency_key),
    CONSTRAINT cb_gmail_sent_recon_ops_fingerprint_check
        CHECK (request_fingerprint ~ '^[0-9a-f]{64}$'),
    CONSTRAINT cb_gmail_sent_recon_ops_actor_check
        CHECK (length(btrim(requested_by)) > 0),
    CONSTRAINT cb_gmail_sent_recon_ops_state_check
        CHECK (state IN ('pending', 'completed')),
    CONSTRAINT cb_gmail_sent_recon_ops_outcome_check
        CHECK (
            (state = 'pending' AND outcome_state IS NULL AND completed_at IS NULL)
            OR (
                state = 'completed'
                AND outcome_state IN ('draft_present', 'draft_missing', 'sent_confirmed')
                AND completed_at IS NOT NULL
            )
        )
);

CREATE INDEX IF NOT EXISTS idx_cb_gmail_sent_recon_ops_record
    ON commercial_billing_gmail_sent_reconciliation_operations (
        gmail_draft_record_id, requested_at DESC
    );

COMMENT ON COLUMN commercial_billing_invoice_gmail_drafts.reconciliation_state IS
    'Observed Gmail delivery proof state; draft_missing is not sent and sent_confirmed requires metadata proof.';

COMMENT ON TABLE commercial_billing_gmail_sent_reconciliation_operations IS
    'Idempotent authenticated Sent-mail reconciliation requests for one durable commercial Gmail draft.';
