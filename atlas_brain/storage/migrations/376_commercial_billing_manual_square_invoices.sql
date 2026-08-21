-- Durable manual-Square invoice reference and explicit sent-state evidence.
--
-- An approved EOM commercial invoice whose immutable invoice metadata declares
-- manual_square begins as a derived queue item; this migration deliberately
-- does not backfill a row or change an invoice. A durable row appears only
-- after an authenticated operator records the externally created Square invoice
-- reference. A later explicit provider action may then mark the linked ATLAS
-- invoice sent via Square. No Square API, Gmail, PDF, email, payment, or
-- service-marker operation is performed by this schema migration.
--
-- Rollback: stop the manual-Square route/service and retain these audit rows.
-- Dropping manual delivery references or operation evidence is a separately
-- authorized destructive retention action, never a mixed-version rollback.

CREATE TABLE IF NOT EXISTS commercial_billing_manual_square_invoices (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    approval_id UUID NOT NULL UNIQUE
        REFERENCES commercial_billing_candidate_approvals(id) ON DELETE RESTRICT,
    invoice_id UUID NOT NULL UNIQUE
        REFERENCES invoices(id) ON DELETE RESTRICT,
    state VARCHAR(32) NOT NULL DEFAULT 'reference_recorded',
    square_invoice_reference VARCHAR(256) NOT NULL,
    reference_recorded_by VARCHAR(128) NOT NULL,
    reference_recorded_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    sent_via_square_by VARCHAR(128),
    sent_via_square_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT commercial_billing_manual_square_invoices_state_check
        CHECK (state IN ('reference_recorded', 'sent_via_square')),
    CONSTRAINT commercial_billing_manual_square_invoices_reference_check
        CHECK (
            length(btrim(square_invoice_reference)) > 0
            AND position(E'\n' IN square_invoice_reference) = 0
            AND position(E'\r' IN square_invoice_reference) = 0
        ),
    CONSTRAINT commercial_billing_manual_square_invoices_reference_actor_check
        CHECK (length(btrim(reference_recorded_by)) > 0),
    CONSTRAINT commercial_billing_manual_square_invoices_sent_state_check
        CHECK (
            (
                state = 'reference_recorded'
                AND sent_via_square_by IS NULL
                AND sent_via_square_at IS NULL
            )
            OR (
                state = 'sent_via_square'
                AND length(btrim(COALESCE(sent_via_square_by, ''))) > 0
                AND sent_via_square_at IS NOT NULL
            )
        )
);

CREATE TABLE IF NOT EXISTS commercial_billing_manual_square_invoice_operations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    manual_square_invoice_id UUID NOT NULL
        REFERENCES commercial_billing_manual_square_invoices(id) ON DELETE RESTRICT,
    source VARCHAR(32) NOT NULL DEFAULT 'eom_admin',
    idempotency_key VARCHAR(128) NOT NULL,
    request_fingerprint VARCHAR(64) NOT NULL,
    operation_kind VARCHAR(32) NOT NULL,
    requested_by VARCHAR(128) NOT NULL,
    requested_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT commercial_billing_manual_square_invoice_operations_source_key
        UNIQUE (source, idempotency_key),
    CONSTRAINT commercial_billing_manual_square_invoice_operations_fingerprint_check
        CHECK (request_fingerprint ~ '^[0-9a-f]{64}$'),
    CONSTRAINT commercial_billing_manual_square_invoice_operations_kind_check
        CHECK (operation_kind IN ('record_reference', 'mark_sent')),
    CONSTRAINT commercial_billing_manual_square_invoice_operations_actor_check
        CHECK (length(btrim(requested_by)) > 0)
);

CREATE INDEX IF NOT EXISTS idx_commercial_billing_manual_square_operations_record
    ON commercial_billing_manual_square_invoice_operations (
        manual_square_invoice_id, requested_at DESC
    );

COMMENT ON TABLE commercial_billing_manual_square_invoices IS
    'One immutable external Square reference and optional explicit sent-via-Square audit per approved EOM commercial invoice.';

COMMENT ON TABLE commercial_billing_manual_square_invoice_operations IS
    'Idempotent authenticated local provider operations for manual Square reference recording and sent-state confirmation.';
