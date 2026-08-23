-- atlas: atomic-bookkeeping
-- Bind the recovered 391 commercial-billing invoice fence to the schema whose
-- catalog it was reviewed against. 391 is immutable target evidence and its
-- function body deliberately remains byte-identical; this forward-only receipt
-- fixes the ambient-search-path execution gap without rewriting that history.
--
-- The migration is reserved for the exact 391 recovery state. The reconciliation
-- selector re-checks that state before admitting this file, and this SQL repeats
-- the function-body/configuration boundary before recording its own receipt.

DO $schema_binding$
DECLARE
    schema_name TEXT := pg_catalog.current_schema();
    pgcrypto_schema TEXT;
    observed_fence_body TEXT;
    observed_fence_config TEXT[];
    expected_fence_config TEXT[];
    observed_fence_sha256 TEXT;
BEGIN
    IF schema_name IS NULL THEN
        RAISE EXCEPTION
            'Cannot bind commercial billing run fence without an active schema';
    END IF;

    SELECT namespace_state.nspname
    INTO pgcrypto_schema
    FROM pg_catalog.pg_extension AS extension_state
    JOIN pg_catalog.pg_namespace AS namespace_state
      ON namespace_state.oid = extension_state.extnamespace
    WHERE extension_state.extname = 'pgcrypto'
    LIMIT 1;

    IF pgcrypto_schema IS NULL THEN
        RAISE EXCEPTION
            'Cannot bind commercial billing run fence without pgcrypto SHA-256 support';
    END IF;

    SELECT function_state.prosrc, function_state.proconfig
    INTO observed_fence_body, observed_fence_config
    FROM pg_catalog.pg_proc AS function_state
    JOIN pg_catalog.pg_namespace AS namespace_state
      ON namespace_state.oid = function_state.pronamespace
    WHERE namespace_state.nspname = schema_name
      AND function_state.proname =
          'prevent_commercial_billing_invoice_for_excluded_candidate'
      AND function_state.pronargs = 0
      AND function_state.prorettype = 'trigger'::pg_catalog.regtype
    LIMIT 1;

    IF observed_fence_body IS NULL THEN
        RAISE EXCEPTION
            'Cannot bind commercial billing run fence without the recovered invoice fence function';
    END IF;

    EXECUTE format(
        'SELECT encode(%1$I.digest($1::text, ''sha256''), ''hex'')',
        pgcrypto_schema
    )
    INTO observed_fence_sha256
    USING observed_fence_body;

    IF observed_fence_sha256 IS DISTINCT FROM
            '04b99e4a3ff2b18f2d58d3e1e610a4b2079fcbbd0d5ce51d97c212daaefd0477' THEN
        RAISE EXCEPTION
            'Cannot bind commercial billing run fence from an unrecognized recovered invoice fence body';
    END IF;

    expected_fence_config := ARRAY[
        format('search_path=pg_catalog, %I, pg_temp', schema_name)
    ];
    IF COALESCE(observed_fence_config, ARRAY[]::text[]) <> ARRAY[]::text[]
       AND observed_fence_config IS DISTINCT FROM expected_fence_config THEN
        RAISE EXCEPTION
            'Cannot bind commercial billing run fence with an unrecognized function configuration';
    END IF;

    EXECUTE format(
        'ALTER FUNCTION %1$I.prevent_commercial_billing_invoice_for_excluded_candidate() '
        || 'SET search_path = pg_catalog, %1$I, pg_temp',
        schema_name
    );
END;
$schema_binding$;
