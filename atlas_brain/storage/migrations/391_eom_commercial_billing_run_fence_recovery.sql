-- atlas: atomic-bookkeeping
-- Recover the current run-scoped commercial-billing invoice fence after the
-- historical 379 review-decision source was recorded under a synthetic version
-- and the target retained an older global-lookup function body.
--
-- This migration is intentionally forward-only and data-preserving. It changes
-- only the database function/trigger boundary after proving the exact observed
-- legacy catalog state. It never rewrites review decisions, overrides,
-- invoices, payments, or historical migration receipts.

DO $recovery$
DECLARE
    pgcrypto_schema TEXT;
    legacy_function_body TEXT;
    observed_fence_sha256 TEXT;
BEGIN
    SELECT namespace_state.nspname
    INTO pgcrypto_schema
    FROM pg_catalog.pg_extension AS extension_state
    JOIN pg_catalog.pg_namespace AS namespace_state
      ON namespace_state.oid = extension_state.extnamespace
    WHERE extension_state.extname = 'pgcrypto'
    LIMIT 1;

    IF pgcrypto_schema IS NULL THEN
        RAISE EXCEPTION
            'Cannot recover commercial billing run fence without pgcrypto SHA-256 support';
    END IF;

    SELECT function_state.prosrc
    INTO legacy_function_body
    FROM pg_catalog.pg_proc AS function_state
    JOIN pg_catalog.pg_namespace AS namespace_state
      ON namespace_state.oid = function_state.pronamespace
    WHERE namespace_state.nspname = pg_catalog.current_schema()
      AND function_state.proname =
          'prevent_commercial_billing_invoice_for_excluded_candidate'
      AND function_state.pronargs = 0
      AND function_state.prorettype = 'trigger'::pg_catalog.regtype
    LIMIT 1;

    IF legacy_function_body IS NULL THEN
        RAISE EXCEPTION
            'Cannot recover commercial billing run fence without the legacy invoice fence function';
    END IF;

    EXECUTE format(
        'SELECT encode(%1$I.digest($1::text, ''sha256''), ''hex'')',
        pgcrypto_schema
    )
    INTO observed_fence_sha256
    USING legacy_function_body;

    IF observed_fence_sha256 IS DISTINCT FROM
            'b71db37ee1906ca26788be21deb716092052fc3197d4b72762d57892fbc77851' THEN
        RAISE EXCEPTION
            'Cannot recover commercial billing run fence from an unrecognized invoice fence body';
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_catalog.pg_class AS relation_state
        JOIN pg_catalog.pg_namespace AS namespace_state
          ON namespace_state.oid = relation_state.relnamespace
        WHERE namespace_state.nspname = pg_catalog.current_schema()
          AND relation_state.relname IN (
              'commercial_billing_candidate_review_decisions',
              'commercial_billing_candidate_overrides',
              'commercial_billing_run_candidates',
              'invoices'
          )
          AND relation_state.relkind = 'r'
          AND relation_state.relpersistence = 'p'
          AND NOT relation_state.relispartition
        GROUP BY namespace_state.nspname
        HAVING COUNT(*) = 4
    ) THEN
        RAISE EXCEPTION
            'Cannot recover commercial billing run fence without the reviewed billing catalog';
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_catalog.pg_attribute AS attribute_state
        JOIN pg_catalog.pg_class AS relation_state
          ON relation_state.oid = attribute_state.attrelid
        JOIN pg_catalog.pg_namespace AS namespace_state
          ON namespace_state.oid = relation_state.relnamespace
        WHERE namespace_state.nspname = pg_catalog.current_schema()
          AND relation_state.relname =
              'commercial_billing_candidate_review_decisions'
          AND attribute_state.attname = 'review_fingerprint'
          AND NOT attribute_state.attisdropped
          AND attribute_state.attnotnull
    ) THEN
        RAISE EXCEPTION
            'Cannot recover commercial billing run fence without the reviewed decision identity';
    END IF;

    IF (
        SELECT COUNT(*)
        FROM (
            VALUES
                ('commercial_billing_candidate_review_decisions', 'candidate_key', 'varchar'),
                ('commercial_billing_candidate_review_decisions', 'source_fingerprint', 'varchar'),
                ('commercial_billing_candidate_review_decisions', 'review_fingerprint', 'varchar'),
                ('commercial_billing_candidate_review_decisions', 'decision', 'varchar'),
                ('commercial_billing_candidate_overrides', 'billing_run_id', 'uuid'),
                ('commercial_billing_candidate_overrides', 'candidate_key', 'varchar'),
                ('commercial_billing_candidate_overrides', 'source_fingerprint', 'varchar'),
                ('commercial_billing_candidate_overrides', 'review_fingerprint', 'varchar'),
                ('commercial_billing_run_candidates', 'billing_run_id', 'uuid'),
                ('commercial_billing_run_candidates', 'candidate_key', 'varchar'),
                ('commercial_billing_run_candidates', 'source_fingerprint', 'varchar')
        ) AS expected_column(relation_name, column_name, type_name)
        JOIN pg_catalog.pg_class AS relation_state
          ON relation_state.relname = expected_column.relation_name
        JOIN pg_catalog.pg_namespace AS namespace_state
          ON namespace_state.oid = relation_state.relnamespace
        JOIN pg_catalog.pg_attribute AS attribute_state
          ON attribute_state.attrelid = relation_state.oid
         AND attribute_state.attname = expected_column.column_name
        JOIN pg_catalog.pg_type AS type_state
          ON type_state.oid = attribute_state.atttypid
        WHERE namespace_state.nspname = pg_catalog.current_schema()
          AND relation_state.relkind = 'r'
          AND NOT relation_state.relispartition
          AND NOT attribute_state.attisdropped
          AND attribute_state.attnotnull
          AND type_state.typname = expected_column.type_name
    ) <> 11 THEN
        RAISE EXCEPTION
            'Cannot recover commercial billing run fence without the final reviewed column contract';
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_catalog.pg_trigger AS trigger_state
        JOIN pg_catalog.pg_class AS relation_state
          ON relation_state.oid = trigger_state.tgrelid
        JOIN pg_catalog.pg_namespace AS namespace_state
          ON namespace_state.oid = relation_state.relnamespace
        JOIN pg_catalog.pg_proc AS function_state
          ON function_state.oid = trigger_state.tgfoid
        WHERE namespace_state.nspname = pg_catalog.current_schema()
          AND relation_state.relname = 'invoices'
          AND trigger_state.tgname =
              'trg_prevent_commercial_billing_invoice_for_excluded_candidate'
          AND trigger_state.tgtype = 7
          AND trigger_state.tgenabled = 'O'
          AND NOT trigger_state.tgisinternal
          AND function_state.proname =
              'prevent_commercial_billing_invoice_for_excluded_candidate'
    ) THEN
        RAISE EXCEPTION
            'Cannot recover commercial billing run fence without the legacy invoice trigger';
    END IF;

    IF (
        SELECT COUNT(*)
        FROM pg_catalog.pg_trigger AS trigger_state
        JOIN pg_catalog.pg_class AS relation_state
          ON relation_state.oid = trigger_state.tgrelid
        JOIN pg_catalog.pg_namespace AS namespace_state
          ON namespace_state.oid = relation_state.relnamespace
        JOIN pg_catalog.pg_proc AS function_state
          ON function_state.oid = trigger_state.tgfoid
        WHERE namespace_state.nspname = pg_catalog.current_schema()
          AND NOT trigger_state.tgisinternal
          AND trigger_state.tgenabled = 'O'
          AND (
              relation_state.relname,
              trigger_state.tgname,
              function_state.proname,
              trigger_state.tgtype
          ) IN (
              (
                  'commercial_billing_candidate_review_decisions',
                  'trg_prevent_commercial_billing_review_decision_mutation',
                  'prevent_commercial_billing_review_decision_mutation',
                  27
              ),
              (
                  'commercial_billing_candidate_review_decisions',
                  'trg_prevent_commercial_billing_review_decision_truncate',
                  'prevent_commercial_billing_review_decision_mutation',
                  34
              ),
              (
                  'commercial_billing_candidate_overrides',
                  'trg_prevent_commercial_billing_candidate_override_mutation',
                  'prevent_commercial_billing_candidate_override_mutation',
                  27
              ),
              (
                  'commercial_billing_candidate_overrides',
                  'trg_prevent_commercial_billing_candidate_override_truncate',
                  'prevent_commercial_billing_candidate_override_mutation',
                  34
              )
          )
    ) <> 4 THEN
        RAISE EXCEPTION
            'Cannot recover commercial billing run fence without immutable review history guards';
    END IF;

    IF (SELECT COUNT(*)
        FROM schema_migrations
        WHERE name = '379_commercial_billing_candidate_review_decisions'
          AND version = -10
          AND content_sha256 IS NULL
          AND applied_at = TIMESTAMPTZ '2026-08-16 18:04:47.984357+00') <> 1
       OR
       (SELECT COUNT(*)
        FROM schema_migrations
        WHERE name = '380_commercial_billing_candidate_review_decisions'
          AND version = 380
          AND content_sha256 IS NULL
          AND applied_at = TIMESTAMPTZ '2026-08-16 18:22:56.919633+00') <> 1
       OR
       (SELECT COUNT(*)
        FROM schema_migrations
        WHERE name = '381_commercial_billing_candidate_review_decisions_recovery'
          AND version = 381
          AND content_sha256 IS NULL
          AND applied_at = TIMESTAMPTZ '2026-08-16 23:18:24.384279+00') <> 1
       OR
       (SELECT COUNT(*)
        FROM schema_migrations
        WHERE name = '382_commercial_billing_candidate_overrides'
          AND version = 382
          AND content_sha256 IS NULL
          AND applied_at = TIMESTAMPTZ '2026-08-17 19:09:25.208581+00') <> 1 THEN
        RAISE EXCEPTION
            'Cannot recover commercial billing run fence without exact historical ledger receipts';
    END IF;
END;
$recovery$;

CREATE OR REPLACE FUNCTION prevent_commercial_billing_invoice_for_excluded_candidate()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $function$
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
$function$;

DROP TRIGGER IF EXISTS trg_prevent_commercial_billing_invoice_for_excluded_candidate
    ON invoices;
CREATE TRIGGER trg_prevent_commercial_billing_invoice_for_excluded_candidate
    BEFORE INSERT ON invoices
    FOR EACH ROW
    EXECUTE FUNCTION prevent_commercial_billing_invoice_for_excluded_candidate();

COMMENT ON FUNCTION prevent_commercial_billing_invoice_for_excluded_candidate() IS
    'Forward recovery for the historical commercial-billing review fence; binds review identity to billing run.';
