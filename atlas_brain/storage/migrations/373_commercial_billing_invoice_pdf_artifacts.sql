-- Durable, immutable PDF artifacts for explicitly approved EOM commercial invoices.
--
-- This is additive.  It is intentionally separate from the financial approval
-- transaction and from Gmail/Square delivery state: an artifact row means only
-- that ATLAS retained one exact invoice PDF for a linked approved draft.
--
-- Rollback: stop the artifact route/service and retain the records.  Dropping
-- PII-bearing invoice artifacts or their audit receipts is a separately
-- authorized destructive retention action, never a mixed-version rollback.

CREATE TABLE IF NOT EXISTS commercial_billing_invoice_pdf_artifacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    approval_id UUID NOT NULL UNIQUE
        REFERENCES commercial_billing_candidate_approvals(id) ON DELETE RESTRICT,
    artifact_kind VARCHAR(32) NOT NULL DEFAULT 'invoice_pdf',
    state VARCHAR(32) NOT NULL DEFAULT 'ready',
    content_type VARCHAR(128) NOT NULL DEFAULT 'application/pdf',
    filename VARCHAR(128) NOT NULL,
    pdf_bytes BYTEA NOT NULL,
    byte_size INTEGER NOT NULL,
    pdf_sha256 VARCHAR(64) NOT NULL,
    render_fingerprint VARCHAR(64) NOT NULL,
    generated_by VARCHAR(128) NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT commercial_billing_invoice_pdf_artifacts_kind_check
        CHECK (artifact_kind = 'invoice_pdf'),
    CONSTRAINT commercial_billing_invoice_pdf_artifacts_state_check
        CHECK (state = 'ready'),
    CONSTRAINT commercial_billing_invoice_pdf_artifacts_content_type_check
        CHECK (content_type = 'application/pdf'),
    CONSTRAINT commercial_billing_invoice_pdf_artifacts_byte_size_check
        CHECK (byte_size > 0 AND byte_size <= 10485760),
    CONSTRAINT commercial_billing_invoice_pdf_artifacts_bytes_match_size_check
        CHECK (octet_length(pdf_bytes) = byte_size),
    CONSTRAINT commercial_billing_invoice_pdf_artifacts_sha256_check
        CHECK (pdf_sha256 ~ '^[0-9a-f]{64}$'),
    CONSTRAINT commercial_billing_invoice_pdf_artifacts_render_fingerprint_check
        CHECK (render_fingerprint ~ '^[0-9a-f]{64}$')
);

CREATE TABLE IF NOT EXISTS commercial_billing_invoice_pdf_operations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    artifact_id UUID NOT NULL
        REFERENCES commercial_billing_invoice_pdf_artifacts(id) ON DELETE RESTRICT,
    source VARCHAR(32) NOT NULL DEFAULT 'eom_admin',
    idempotency_key VARCHAR(128) NOT NULL,
    request_fingerprint VARCHAR(64) NOT NULL,
    requested_by VARCHAR(128) NOT NULL,
    requested_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT commercial_billing_invoice_pdf_operations_source_key
        UNIQUE (source, idempotency_key),
    CONSTRAINT commercial_billing_invoice_pdf_operations_request_fingerprint_check
        CHECK (request_fingerprint ~ '^[0-9a-f]{64}$')
);

CREATE INDEX IF NOT EXISTS idx_commercial_billing_invoice_pdf_operations_artifact
    ON commercial_billing_invoice_pdf_operations (artifact_id, requested_at DESC);

COMMENT ON TABLE commercial_billing_invoice_pdf_artifacts IS
    'One immutable ready PDF byte artifact per explicitly approved EOM commercial draft invoice; no delivery state.';

COMMENT ON TABLE commercial_billing_invoice_pdf_operations IS
    'Idempotent authenticated requests that created or reused approved commercial invoice PDF artifacts.';
