-- Explicit approval evidence for one commercial billing candidate -> draft invoice.
--
-- This is additive. It neither changes retained review snapshots nor sends an
-- invoice. A successful row proves one exact candidate became one draft ATLAS
-- invoice; PDF/Gmail/Square delivery state belongs to later migrations.
--
-- Rollback: revert the reader/writer first and retain this financial audit
-- evidence. Dropping either object is a separately authorized destructive
-- operation, not a normal mixed-version rollback.

CREATE UNIQUE INDEX IF NOT EXISTS idx_commercial_billing_run_candidates_exact_source
    ON commercial_billing_run_candidates (
        billing_run_id, candidate_key, source_fingerprint
    );

CREATE TABLE IF NOT EXISTS commercial_billing_candidate_approvals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    billing_run_id UUID NOT NULL
        REFERENCES commercial_billing_runs(id) ON DELETE RESTRICT,
    candidate_key VARCHAR(512) NOT NULL,
    source_fingerprint VARCHAR(64) NOT NULL,
    source VARCHAR(32) NOT NULL DEFAULT 'eom_admin',
    idempotency_key VARCHAR(128) NOT NULL,
    request_fingerprint VARCHAR(64) NOT NULL,
    invoice_id UUID NOT NULL REFERENCES invoices(id) ON DELETE RESTRICT,
    state VARCHAR(32) NOT NULL DEFAULT 'invoice_created',
    approved_by VARCHAR(128) NOT NULL,
    approved_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT commercial_billing_candidate_approvals_fingerprint_check
        CHECK (source_fingerprint ~ '^[0-9a-f]{64}$'),
    CONSTRAINT commercial_billing_candidate_approvals_request_fingerprint_check
        CHECK (request_fingerprint ~ '^[0-9a-f]{64}$'),
    CONSTRAINT commercial_billing_candidate_approvals_state_check
        CHECK (state = 'invoice_created'),
    CONSTRAINT commercial_billing_candidate_approvals_source_key
        UNIQUE (source, idempotency_key),
    CONSTRAINT commercial_billing_candidate_approvals_candidate_fingerprint_key
        UNIQUE (candidate_key, source_fingerprint),
    CONSTRAINT commercial_billing_candidate_approvals_invoice_key
        UNIQUE (invoice_id),
    CONSTRAINT commercial_billing_candidate_approvals_snapshot_fkey
        FOREIGN KEY (billing_run_id, candidate_key, source_fingerprint)
        REFERENCES commercial_billing_run_candidates (
            billing_run_id, candidate_key, source_fingerprint
        ) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_commercial_billing_candidate_approvals_run
    ON commercial_billing_candidate_approvals (billing_run_id, approved_at DESC);

CREATE UNIQUE INDEX IF NOT EXISTS idx_invoices_eom_commercial_billing_source_ref
    ON invoices (source, source_ref)
    WHERE source = 'eom_commercial_billing' AND source_ref IS NOT NULL;

COMMENT ON TABLE commercial_billing_candidate_approvals IS
    'One immutable approval audit row per exact commercial candidate fingerprint; creates a draft invoice only.';
