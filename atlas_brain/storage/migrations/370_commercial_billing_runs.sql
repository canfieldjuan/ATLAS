-- Durable, pre-approval EOM commercial billing-review evidence.
--
-- This migration stores only the immutable preview Juan reviewed.  It creates
-- no invoice, PDF, Gmail draft, sent state, service-invoiced marker, or email.
-- The parent/child relationship is deliberately restrictive: review evidence
-- must outlive application rollback and cannot be silently deleted by a later
-- invoice or customer operation.
--
-- Rollback evidence:
--   Revert application code first and retain these additive audit rows.
--   A separately authorized destructive teardown, only after every reader and
--   writer is removed, may run:
--       DROP TABLE commercial_billing_run_candidates;
--       DROP TABLE commercial_billing_runs;
--   That is not an ordinary rollback: it destroys reviewed source evidence and
--   breaks a mixed-version deployment.
--
-- Roll-forward safety:
--   Existing candidate preview, payment, deposit, invoice, MCP, scheduler, PDF,
--   Gmail, and mail paths do not reference these tables.  The new writer is an
--   additive authenticated provider route and remains safe before tracker/UI
--   consumers are deployed.

CREATE TABLE IF NOT EXISTS commercial_billing_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    billing_period VARCHAR(7) NOT NULL,
    calendar_id TEXT,
    state VARCHAR(16) NOT NULL DEFAULT 'draft',
    candidate_contract_version INTEGER NOT NULL,
    snapshot_fingerprint VARCHAR(64) NOT NULL,
    source VARCHAR(32) NOT NULL DEFAULT 'eom_admin',
    idempotency_key VARCHAR(128) NOT NULL,
    request_fingerprint VARCHAR(64) NOT NULL,
    created_by VARCHAR(128) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT commercial_billing_runs_period_check
        CHECK (billing_period ~ '^[0-9]{4}-(0[1-9]|1[0-2])$'),
    CONSTRAINT commercial_billing_runs_state_check
        CHECK (state = 'draft'),
    CONSTRAINT commercial_billing_runs_contract_version_check
        CHECK (candidate_contract_version > 0),
    CONSTRAINT commercial_billing_runs_snapshot_fingerprint_check
        CHECK (snapshot_fingerprint ~ '^[0-9a-f]{64}$'),
    CONSTRAINT commercial_billing_runs_request_fingerprint_check
        CHECK (request_fingerprint ~ '^[0-9a-f]{64}$'),
    CONSTRAINT commercial_billing_runs_source_idempotency_key_key
        UNIQUE (source, idempotency_key)
);

CREATE INDEX IF NOT EXISTS idx_commercial_billing_runs_period_created
    ON commercial_billing_runs (billing_period, created_at DESC);

CREATE TABLE IF NOT EXISTS commercial_billing_run_candidates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    billing_run_id UUID NOT NULL
        REFERENCES commercial_billing_runs(id) ON DELETE RESTRICT,
    candidate_key VARCHAR(512) NOT NULL,
    source_fingerprint VARCHAR(64) NOT NULL,
    display_order INTEGER NOT NULL,
    snapshot JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT commercial_billing_run_candidates_source_fingerprint_check
        CHECK (source_fingerprint ~ '^[0-9a-f]{64}$'),
    CONSTRAINT commercial_billing_run_candidates_display_order_check
        CHECK (display_order >= 0),
    CONSTRAINT commercial_billing_run_candidates_run_key_key
        UNIQUE (billing_run_id, candidate_key),
    CONSTRAINT commercial_billing_run_candidates_run_order_key
        UNIQUE (billing_run_id, display_order)
);

CREATE INDEX IF NOT EXISTS idx_commercial_billing_run_candidates_run_order
    ON commercial_billing_run_candidates (billing_run_id, display_order);

COMMENT ON TABLE commercial_billing_runs IS
    'Immutable EOM commercial billing-review snapshots. A draft run is not an invoice or delivery operation.';
COMMENT ON TABLE commercial_billing_run_candidates IS
    'Complete generated candidate evidence retained for stale-source reconciliation before approval.';
