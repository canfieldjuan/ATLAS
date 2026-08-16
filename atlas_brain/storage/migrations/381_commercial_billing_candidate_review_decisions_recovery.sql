-- atlas: atomic-bookkeeping
-- Recover commercial billing review-decision enforcement where migration 380 was
-- recorded before its later global revision identity and database safety objects
-- were added.
--
-- This is forward-only and data-preserving. The recorded table is retained. A
-- legacy per-run revision key is replaced only if every historical row already
-- has a unique global (candidate_key, source_fingerprint, revision) identity.
-- Ambiguous history stops the migration before its ledger row is recorded.

DO $$
DECLARE
    revision_key_columns TEXT[];
    revision_key_exists BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid = 'commercial_billing_candidate_review_decisions'::regclass
          AND conname = 'commercial_billing_candidate_review_decisions_revision_key'
    )
    INTO revision_key_exists;

    SELECT ARRAY_AGG(attribute.attname ORDER BY key_column.ordinality)
    INTO revision_key_columns
    FROM pg_constraint AS constraint_state
    JOIN UNNEST(constraint_state.conkey) WITH ORDINALITY
        AS key_column(attnum, ordinality)
        ON TRUE
    JOIN pg_attribute AS attribute
        ON attribute.attrelid = constraint_state.conrelid
       AND attribute.attnum = key_column.attnum
    WHERE constraint_state.conrelid =
              'commercial_billing_candidate_review_decisions'::regclass
      AND constraint_state.conname =
              'commercial_billing_candidate_review_decisions_revision_key'
      AND constraint_state.contype = 'u';

    IF revision_key_columns IS DISTINCT FROM
            ARRAY['candidate_key', 'source_fingerprint', 'revision'] THEN
        IF EXISTS (
            SELECT 1
            FROM commercial_billing_candidate_review_decisions
            GROUP BY candidate_key, source_fingerprint, revision
            HAVING COUNT(*) > 1
        ) THEN
            RAISE EXCEPTION
                'Cannot recover commercial billing review decisions with duplicate global revision identities';
        END IF;

        IF revision_key_exists THEN
            ALTER TABLE commercial_billing_candidate_review_decisions
                DROP CONSTRAINT commercial_billing_candidate_review_decisions_revision_key;
        END IF;

        ALTER TABLE commercial_billing_candidate_review_decisions
            ADD CONSTRAINT commercial_billing_candidate_review_decisions_revision_key
            UNIQUE (candidate_key, source_fingerprint, revision);
    END IF;
END;
$$;

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
