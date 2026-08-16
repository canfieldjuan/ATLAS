-- Append-only operator review decisions for durable EOM commercial billing
-- candidate snapshots.
--
-- A decision is deliberately not an invoice, delivery, calendar, or service
-- marker.  It records whether one globally approved candidate identity is
-- currently included or excluded from later explicit approval. The retained
-- run row proves the source snapshot that recorded the event, while the
-- candidate key plus fingerprint scope matches the existing global approval
-- identity. The lack of a row is the backward-compatible included default.
-- Re-inclusion is a new event rather than an update so the complete
-- actor/reason history remains available.
--
-- Dependency: migration 372 establishes the exact retained-snapshot unique
-- key used by this foreign key.  Migration runners apply numeric migrations in
-- order, so a mixed-version reader that has not yet received this migration
-- safely treats the additive review state as unavailable rather than creating
-- an invoice.
--
-- Rollback: remove the reader/writer before rolling back application code and
-- retain these audit rows.  Dropping this table would destroy review history;
-- that is a separately authorized destructive operation, not a normal
-- mixed-version rollback.

CREATE TABLE IF NOT EXISTS commercial_billing_candidate_review_decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    billing_run_id UUID NOT NULL
        REFERENCES commercial_billing_runs(id) ON DELETE RESTRICT,
    candidate_key VARCHAR(512) NOT NULL,
    source_fingerprint VARCHAR(64) NOT NULL,
    revision INTEGER NOT NULL,
    decision VARCHAR(16) NOT NULL,
    reason VARCHAR(1000) NOT NULL,
    source VARCHAR(32) NOT NULL DEFAULT 'eom_admin',
    idempotency_key VARCHAR(128) NOT NULL,
    request_fingerprint VARCHAR(64) NOT NULL,
    decided_by VARCHAR(128) NOT NULL,
    decided_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT commercial_billing_candidate_review_decisions_fingerprint_check
        CHECK (source_fingerprint ~ '^[0-9a-f]{64}$'),
    CONSTRAINT commercial_billing_candidate_review_decisions_revision_check
        CHECK (revision > 0),
    CONSTRAINT commercial_billing_candidate_review_decisions_decision_check
        CHECK (decision IN ('included', 'excluded')),
    CONSTRAINT commercial_billing_candidate_review_decisions_reason_check
        CHECK (length(btrim(reason)) > 0),
    CONSTRAINT commercial_billing_candidate_review_decisions_request_fingerprint_check
        CHECK (request_fingerprint ~ '^[0-9a-f]{64}$'),
    CONSTRAINT commercial_billing_candidate_review_decisions_actor_check
        CHECK (length(btrim(decided_by)) > 0),
    CONSTRAINT commercial_billing_candidate_review_decisions_source_key
        UNIQUE (source, idempotency_key),
    CONSTRAINT commercial_billing_candidate_review_decisions_revision_key
        UNIQUE (candidate_key, source_fingerprint, revision),
    CONSTRAINT commercial_billing_candidate_review_decisions_snapshot_fkey
        FOREIGN KEY (billing_run_id, candidate_key, source_fingerprint)
        REFERENCES commercial_billing_run_candidates (
            billing_run_id, candidate_key, source_fingerprint
        ) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_commercial_billing_candidate_review_decisions_run_candidate
    ON commercial_billing_candidate_review_decisions (
        billing_run_id,
        candidate_key,
        source_fingerprint,
        revision DESC
    );

CREATE INDEX IF NOT EXISTS idx_commercial_billing_run_candidates_identity
    ON commercial_billing_run_candidates (candidate_key, source_fingerprint);

CREATE OR REPLACE FUNCTION prevent_commercial_billing_review_decision_mutation()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'commercial_billing_candidate_review_decisions is append-only';
END;
$$;

DROP TRIGGER IF EXISTS trg_prevent_commercial_billing_review_decision_mutation
    ON commercial_billing_candidate_review_decisions;
CREATE TRIGGER trg_prevent_commercial_billing_review_decision_mutation
    BEFORE UPDATE OR DELETE ON commercial_billing_candidate_review_decisions
    FOR EACH ROW
    EXECUTE FUNCTION prevent_commercial_billing_review_decision_mutation();

DROP TRIGGER IF EXISTS trg_prevent_commercial_billing_review_decision_truncate
    ON commercial_billing_candidate_review_decisions;
CREATE TRIGGER trg_prevent_commercial_billing_review_decision_truncate
    BEFORE TRUNCATE ON commercial_billing_candidate_review_decisions
    FOR EACH STATEMENT
    EXECUTE FUNCTION prevent_commercial_billing_review_decision_mutation();

CREATE OR REPLACE FUNCTION prevent_commercial_billing_invoice_for_excluded_candidate()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    candidate_identity_key TEXT;
    candidate_identity_fingerprint TEXT;
    current_decision VARCHAR(16);
BEGIN
    IF NEW.source IS DISTINCT FROM 'eom_commercial_billing' THEN
        RETURN NEW;
    END IF;

    IF jsonb_typeof(NEW.metadata -> 'candidateKey') IS DISTINCT FROM 'string'
       OR jsonb_typeof(NEW.metadata -> 'sourceFingerprint') IS DISTINCT FROM 'string' THEN
        RAISE EXCEPTION 'Commercial billing invoice review identity is invalid';
    END IF;

    candidate_identity_key := NEW.metadata ->> 'candidateKey';
    candidate_identity_fingerprint := NEW.metadata ->> 'sourceFingerprint';
    IF candidate_identity_key IS NULL
       OR candidate_identity_key = ''
       OR length(candidate_identity_key) > 512
       OR candidate_identity_fingerprint IS NULL
       OR candidate_identity_fingerprint !~ '^[0-9a-f]{64}$' THEN
        RAISE EXCEPTION 'Commercial billing invoice review identity is invalid';
    END IF;

    PERFORM 1
    FROM commercial_billing_run_candidates
    WHERE candidate_key = candidate_identity_key
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

    SELECT decision
    INTO current_decision
    FROM commercial_billing_candidate_review_decisions
    WHERE candidate_key = candidate_identity_key
      AND source_fingerprint = candidate_identity_fingerprint
    ORDER BY revision DESC
    LIMIT 1;

    IF current_decision = 'excluded' THEN
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

COMMENT ON TABLE commercial_billing_candidate_review_decisions IS
    'Append-only actor-audited include/exclude decisions for globally approved EOM commercial candidate identities; no invoice or delivery side effect.';
