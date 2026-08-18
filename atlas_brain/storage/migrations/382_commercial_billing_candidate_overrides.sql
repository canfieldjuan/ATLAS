-- atlas: atomic-bookkeeping
-- Append-only, one-run EOM commercial billing candidate overrides.
--
-- The source candidate snapshot remains immutable.  A revision captures the
-- separately approved one-time effective evidence, and its review fingerprint
-- prevents an Include decision for prior money/recipient data from approving a
-- newer revision.  This migration is additive and never creates an invoice,
-- PDF, Gmail draft, email, Square record, or sent state.

CREATE TABLE IF NOT EXISTS commercial_billing_candidate_overrides (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    billing_run_id UUID NOT NULL
        REFERENCES commercial_billing_runs(id) ON DELETE RESTRICT,
    candidate_key VARCHAR(512) NOT NULL,
    source_fingerprint VARCHAR(64) NOT NULL,
    revision INTEGER NOT NULL,
    review_fingerprint VARCHAR(64) NOT NULL,
    effective_snapshot JSONB NOT NULL,
    reason_code VARCHAR(64) NOT NULL,
    reason VARCHAR(1000) NOT NULL,
    source VARCHAR(32) NOT NULL DEFAULT 'eom_admin',
    idempotency_key VARCHAR(128) NOT NULL,
    request_fingerprint VARCHAR(64) NOT NULL,
    overridden_by VARCHAR(128) NOT NULL,
    overridden_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT commercial_billing_candidate_overrides_source_fingerprint_check
        CHECK (source_fingerprint ~ '^[0-9a-f]{64}$'),
    CONSTRAINT commercial_billing_candidate_overrides_review_fingerprint_check
        CHECK (review_fingerprint ~ '^[0-9a-f]{64}$'),
    CONSTRAINT commercial_billing_candidate_overrides_revision_check
        CHECK (revision > 0),
    CONSTRAINT commercial_billing_candidate_overrides_reason_code_check
        CHECK (reason_code IN (
            'one_time_service_variation',
            'partial_or_missed_service',
            'approved_pricing_exception',
            'customer_credit',
            'additional_charge',
            'source_correction_pending',
            'billing_delivery_exception'
        )),
    CONSTRAINT commercial_billing_candidate_overrides_reason_check
        CHECK (length(btrim(reason)) > 0),
    CONSTRAINT commercial_billing_candidate_overrides_request_fingerprint_check
        CHECK (request_fingerprint ~ '^[0-9a-f]{64}$'),
    CONSTRAINT commercial_billing_candidate_overrides_actor_check
        CHECK (length(btrim(overridden_by)) > 0),
    CONSTRAINT commercial_billing_candidate_overrides_source_key
        UNIQUE (source, idempotency_key),
    CONSTRAINT commercial_billing_candidate_overrides_revision_key
        UNIQUE (billing_run_id, candidate_key, source_fingerprint, revision),
    CONSTRAINT commercial_billing_candidate_overrides_review_key
        UNIQUE (review_fingerprint),
    CONSTRAINT commercial_billing_candidate_overrides_snapshot_fkey
        FOREIGN KEY (billing_run_id, candidate_key, source_fingerprint)
        REFERENCES commercial_billing_run_candidates (
            billing_run_id, candidate_key, source_fingerprint
        ) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_commercial_billing_candidate_overrides_active
    ON commercial_billing_candidate_overrides (
        billing_run_id, candidate_key, source_fingerprint, revision DESC
    );

-- Migration 381 made these audit rows append-only. The temporary drop is
-- enclosed by this atomic migration and its table lock; it only permits the
-- deterministic derived identity backfill below, after which the same guard
-- is restored before commit.
DROP TRIGGER IF EXISTS trg_prevent_commercial_billing_review_decision_mutation
    ON commercial_billing_candidate_review_decisions;
DROP TRIGGER IF EXISTS trg_prevent_commercial_billing_review_decision_truncate
    ON commercial_billing_candidate_review_decisions;

ALTER TABLE commercial_billing_candidate_review_decisions
    ADD COLUMN IF NOT EXISTS review_fingerprint VARCHAR(64);
UPDATE commercial_billing_candidate_review_decisions
SET review_fingerprint = source_fingerprint
WHERE review_fingerprint IS NULL;
ALTER TABLE commercial_billing_candidate_review_decisions
    ALTER COLUMN review_fingerprint SET NOT NULL;
ALTER TABLE commercial_billing_candidate_review_decisions
    DROP CONSTRAINT IF EXISTS commercial_billing_candidate_review_decisions_review_fingerprint_check;
ALTER TABLE commercial_billing_candidate_review_decisions
    ADD CONSTRAINT commercial_billing_candidate_review_decisions_review_fingerprint_check
    CHECK (review_fingerprint ~ '^[0-9a-f]{64}$');
CREATE INDEX IF NOT EXISTS idx_commercial_billing_candidate_review_decisions_review
    ON commercial_billing_candidate_review_decisions (
        candidate_key, source_fingerprint, review_fingerprint, revision DESC
    );

CREATE OR REPLACE FUNCTION prevent_commercial_billing_review_decision_mutation()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'commercial_billing_candidate_review_decisions is append-only';
END;
$$;

CREATE TRIGGER trg_prevent_commercial_billing_review_decision_mutation
    BEFORE UPDATE OR DELETE ON commercial_billing_candidate_review_decisions
    FOR EACH ROW
    EXECUTE FUNCTION prevent_commercial_billing_review_decision_mutation();
CREATE TRIGGER trg_prevent_commercial_billing_review_decision_truncate
    BEFORE TRUNCATE ON commercial_billing_candidate_review_decisions
    FOR EACH STATEMENT
    EXECUTE FUNCTION prevent_commercial_billing_review_decision_mutation();

ALTER TABLE commercial_billing_candidate_approvals
    ADD COLUMN IF NOT EXISTS review_fingerprint VARCHAR(64);
UPDATE commercial_billing_candidate_approvals
SET review_fingerprint = source_fingerprint
WHERE review_fingerprint IS NULL;
ALTER TABLE commercial_billing_candidate_approvals
    ALTER COLUMN review_fingerprint SET NOT NULL;
ALTER TABLE commercial_billing_candidate_approvals
    DROP CONSTRAINT IF EXISTS commercial_billing_candidate_approvals_review_fingerprint_check;
ALTER TABLE commercial_billing_candidate_approvals
    ADD CONSTRAINT commercial_billing_candidate_approvals_review_fingerprint_check
    CHECK (review_fingerprint ~ '^[0-9a-f]{64}$');

-- A temporarily mixed Atlas rollout can still have a prior provider binary
-- writing its old INSERT column list. Keep that writer safe for unoverridden
-- candidates by deriving the legacy review identity at the database boundary.
-- A later override still cannot be approved by that legacy identity because
-- the invoice trigger below compares it to the active effective identity.
CREATE OR REPLACE FUNCTION default_commercial_billing_review_fingerprint()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.review_fingerprint IS NULL THEN
        NEW.review_fingerprint := NEW.source_fingerprint;
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_default_commercial_billing_review_decision_fingerprint
    ON commercial_billing_candidate_review_decisions;
CREATE TRIGGER trg_default_commercial_billing_review_decision_fingerprint
    BEFORE INSERT ON commercial_billing_candidate_review_decisions
    FOR EACH ROW
    EXECUTE FUNCTION default_commercial_billing_review_fingerprint();

DROP TRIGGER IF EXISTS trg_default_commercial_billing_approval_fingerprint
    ON commercial_billing_candidate_approvals;
CREATE TRIGGER trg_default_commercial_billing_approval_fingerprint
    BEFORE INSERT ON commercial_billing_candidate_approvals
    FOR EACH ROW
    EXECUTE FUNCTION default_commercial_billing_review_fingerprint();

CREATE OR REPLACE FUNCTION prevent_commercial_billing_candidate_override_mutation()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'commercial_billing_candidate_overrides is append-only';
END;
$$;

DROP TRIGGER IF EXISTS trg_prevent_commercial_billing_candidate_override_mutation
    ON commercial_billing_candidate_overrides;
CREATE TRIGGER trg_prevent_commercial_billing_candidate_override_mutation
    BEFORE UPDATE OR DELETE ON commercial_billing_candidate_overrides
    FOR EACH ROW
    EXECUTE FUNCTION prevent_commercial_billing_candidate_override_mutation();

DROP TRIGGER IF EXISTS trg_prevent_commercial_billing_candidate_override_truncate
    ON commercial_billing_candidate_overrides;
CREATE TRIGGER trg_prevent_commercial_billing_candidate_override_truncate
    BEFORE TRUNCATE ON commercial_billing_candidate_overrides
    FOR EACH STATEMENT
    EXECUTE FUNCTION prevent_commercial_billing_candidate_override_mutation();

-- Replace the earlier excluded-only trigger.  Legacy, unoverridden candidates
-- retain their default-include behavior; an overridden candidate instead needs
-- an explicit Include recorded for its current effective review fingerprint.
CREATE OR REPLACE FUNCTION prevent_commercial_billing_invoice_for_excluded_candidate()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    candidate_identity_key TEXT;
    candidate_identity_fingerprint TEXT;
    candidate_identity_billing_run_id UUID;
    review_identity_fingerprint TEXT;
    active_review_fingerprint TEXT;
    current_decision VARCHAR(16);
BEGIN
    IF NEW.source IS DISTINCT FROM 'eom_commercial_billing' THEN
        RETURN NEW;
    END IF;

    IF jsonb_typeof(NEW.metadata -> 'candidateKey') IS DISTINCT FROM 'string'
       OR jsonb_typeof(NEW.metadata -> 'commercialBillingRunId') IS DISTINCT FROM 'string'
       OR jsonb_typeof(NEW.metadata -> 'sourceFingerprint') IS DISTINCT FROM 'string' THEN
        RAISE EXCEPTION 'Commercial billing invoice review identity is invalid';
    END IF;
    candidate_identity_key := NEW.metadata ->> 'candidateKey';
    BEGIN
        candidate_identity_billing_run_id :=
            (NEW.metadata ->> 'commercialBillingRunId')::UUID;
    EXCEPTION WHEN invalid_text_representation THEN
        RAISE EXCEPTION 'Commercial billing invoice review identity is invalid';
    END;
    candidate_identity_fingerprint := NEW.metadata ->> 'sourceFingerprint';
    review_identity_fingerprint := COALESCE(
        NEW.metadata ->> 'reviewFingerprint', candidate_identity_fingerprint
    );
    IF candidate_identity_key IS NULL
       OR candidate_identity_key = ''
       OR length(candidate_identity_key) > 512
       OR candidate_identity_fingerprint IS NULL
       OR candidate_identity_fingerprint !~ '^[0-9a-f]{64}$'
       OR review_identity_fingerprint !~ '^[0-9a-f]{64}$' THEN
        RAISE EXCEPTION 'Commercial billing invoice review identity is invalid';
    END IF;

    PERFORM 1
    FROM commercial_billing_run_candidates
    WHERE billing_run_id = candidate_identity_billing_run_id
      AND candidate_key = candidate_identity_key
      AND source_fingerprint = candidate_identity_fingerprint
    LIMIT 1;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Commercial billing invoice review identity is invalid';
    END IF;

    PERFORM pg_advisory_xact_lock(
        hashtextextended(
            'commercial-billing-approval:candidate:'
            || candidate_identity_key || ':' || candidate_identity_fingerprint,
            0
        )
    );

    SELECT review_fingerprint
    INTO active_review_fingerprint
    FROM commercial_billing_candidate_overrides
    WHERE billing_run_id = candidate_identity_billing_run_id
      AND candidate_key = candidate_identity_key
      AND source_fingerprint = candidate_identity_fingerprint
    ORDER BY revision DESC
    LIMIT 1;
    active_review_fingerprint := COALESCE(
        active_review_fingerprint, candidate_identity_fingerprint
    );
    IF review_identity_fingerprint IS DISTINCT FROM active_review_fingerprint THEN
        RAISE EXCEPTION 'Commercial billing candidate review identity is stale';
    END IF;

    SELECT decision
    INTO current_decision
    FROM commercial_billing_candidate_review_decisions
    WHERE candidate_key = candidate_identity_key
      AND source_fingerprint = candidate_identity_fingerprint
      AND review_fingerprint = active_review_fingerprint
    ORDER BY revision DESC
    LIMIT 1;

    IF active_review_fingerprint <> candidate_identity_fingerprint
       AND current_decision IS DISTINCT FROM 'included' THEN
        RAISE EXCEPTION 'Commercial billing candidate override requires an explicit include decision';
    END IF;
    IF active_review_fingerprint = candidate_identity_fingerprint
       AND current_decision = 'excluded' THEN
        RAISE EXCEPTION 'Commercial billing candidate is excluded; include it before invoice creation';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_prevent_commercial_billing_invoice_for_excluded_candidate
    ON invoices;
CREATE TRIGGER trg_prevent_commercial_billing_invoice_for_excluded_candidate
    BEFORE INSERT ON invoices
    FOR EACH ROW
    EXECUTE FUNCTION prevent_commercial_billing_invoice_for_excluded_candidate();

COMMENT ON TABLE commercial_billing_candidate_overrides IS
    'Append-only actor-audited one-run EOM candidate overrides; source evidence remains immutable and no delivery side effect is created.';
