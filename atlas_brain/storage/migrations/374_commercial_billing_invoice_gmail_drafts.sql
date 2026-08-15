-- Durable no-send Gmail draft identity for explicitly approved EOM invoices.
--
-- This is deliberately a delivery-audit state machine, not invoice financial
-- state.  A row means ATLAS has a persisted intent or Gmail draft identity for
-- one already-approved invoice PDF.  It does not mean Gmail sent anything and
-- it never updates invoices, receivables, payments, or service evidence.
--
-- The intent row is committed before the Gmail API call.  Its deterministic
-- RFC Message-ID lets a later retry query Gmail's Drafts mailbox when the API
-- outcome was uncertain, rather than issue a second create request.  A row in
-- ``recovery_required`` is deliberately retained evidence; it cannot be
-- interpreted as sent and it is not silently retried.
--
-- Rollback: stop the route/service and retain these delivery/audit records.
-- Dropping rows that describe customer recipient/draft identity is a separately
-- authorized destructive retention action, never a mixed-version rollback.

CREATE TABLE IF NOT EXISTS commercial_billing_invoice_gmail_drafts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    approval_id UUID NOT NULL UNIQUE
        REFERENCES commercial_billing_candidate_approvals(id) ON DELETE RESTRICT,
    artifact_id UUID NOT NULL UNIQUE
        REFERENCES commercial_billing_invoice_pdf_artifacts(id) ON DELETE RESTRICT,
    state VARCHAR(32) NOT NULL DEFAULT 'creating',
    recipient_email VARCHAR(256) NOT NULL,
    subject VARCHAR(512) NOT NULL,
    rfc_message_id VARCHAR(320) NOT NULL UNIQUE,
    gmail_draft_id VARCHAR(256),
    gmail_message_id VARCHAR(256),
    gmail_thread_id VARCHAR(256),
    created_by VARCHAR(128) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_attempt_by VARCHAR(128) NOT NULL,
    last_attempt_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    draft_created_at TIMESTAMPTZ,
    recovery_required_at TIMESTAMPTZ,
    CONSTRAINT commercial_billing_invoice_gmail_drafts_state_check
        CHECK (state IN ('creating', 'retryable', 'recovery_required', 'draft_created')),
    CONSTRAINT commercial_billing_invoice_gmail_drafts_recipient_check
        CHECK (length(btrim(recipient_email)) > 0),
    CONSTRAINT commercial_billing_invoice_gmail_drafts_subject_check
        CHECK (length(btrim(subject)) > 0),
    CONSTRAINT commercial_billing_invoice_gmail_drafts_rfc_message_id_check
        CHECK (
            rfc_message_id ~ '^<[^[:space:]<>]+@[^[:space:]<>]+>$'
            AND length(btrim(rfc_message_id)) = length(rfc_message_id)
        ),
    CONSTRAINT commercial_billing_invoice_gmail_drafts_actor_check
        CHECK (
            length(btrim(created_by)) > 0
            AND length(btrim(last_attempt_by)) > 0
        ),
    CONSTRAINT commercial_billing_invoice_gmail_drafts_external_identity_check
        CHECK (
            (
                gmail_draft_id IS NULL
                AND gmail_message_id IS NULL
                AND gmail_thread_id IS NULL
            )
            OR (
                length(btrim(COALESCE(gmail_draft_id, ''))) > 0
                AND length(btrim(COALESCE(gmail_message_id, ''))) > 0
                AND length(btrim(COALESCE(gmail_thread_id, ''))) > 0
            )
        ),
    CONSTRAINT commercial_billing_invoice_gmail_drafts_completed_state_check
        CHECK (
            state <> 'draft_created'
            OR (
                gmail_draft_id IS NOT NULL
                AND gmail_message_id IS NOT NULL
                AND gmail_thread_id IS NOT NULL
                AND draft_created_at IS NOT NULL
            )
        ),
    CONSTRAINT commercial_billing_invoice_gmail_drafts_recovery_timestamp_check
        CHECK (
            state = 'recovery_required'
            OR recovery_required_at IS NULL
        )
);

CREATE TABLE IF NOT EXISTS commercial_billing_invoice_gmail_draft_operations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    gmail_draft_record_id UUID NOT NULL
        REFERENCES commercial_billing_invoice_gmail_drafts(id) ON DELETE RESTRICT,
    source VARCHAR(32) NOT NULL DEFAULT 'eom_admin',
    idempotency_key VARCHAR(128) NOT NULL,
    request_fingerprint VARCHAR(64) NOT NULL,
    requested_by VARCHAR(128) NOT NULL,
    requested_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT commercial_billing_invoice_gmail_draft_operations_source_key
        UNIQUE (source, idempotency_key),
    CONSTRAINT commercial_billing_invoice_gmail_draft_operations_request_fingerprint_check
        CHECK (request_fingerprint ~ '^[0-9a-f]{64}$'),
    CONSTRAINT commercial_billing_invoice_gmail_draft_operations_actor_check
        CHECK (length(btrim(requested_by)) > 0)
);

CREATE INDEX IF NOT EXISTS idx_commercial_billing_invoice_gmail_draft_operations_record
    ON commercial_billing_invoice_gmail_draft_operations (
        gmail_draft_record_id, requested_at DESC
    );

COMMENT ON TABLE commercial_billing_invoice_gmail_drafts IS
    'One no-send Gmail draft intent/identity per approved EOM commercial invoice PDF; never invoice sent state.';

COMMENT ON TABLE commercial_billing_invoice_gmail_draft_operations IS
    'Idempotent authenticated requests that create, recover, or reuse one commercial Gmail draft identity.';
