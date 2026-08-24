-- atlas: atomic-bookkeeping
-- Forward-only privilege repair for durable EOM missed-call recovery.
--
-- Migration 389 is intentionally additive, but it did not assign the recovery
-- objects to EOM's no-login guard or materialize the normal runtime ACL. If a
-- DBA applies 389, PostgreSQL otherwise leaves those tables owned by that DBA
-- and Atlas cannot serve the recovery route or worker. Preserve 389's evidence
-- and receipt; repair ownership/privileges in this new, recorded migration.
--
-- A database administrator must run this migration. The normal Atlas runtime
-- must never receive membership in atlas_eom_handoff_owner just to operate the
-- recovery service, and NocoDB must never receive direct recovery-table access.

DO $$
DECLARE
    schema_name TEXT := current_schema();
    executor_is_superuser BOOLEAN;
    trusted_guard_role_ready BOOLEAN;
    runtime_role_ready BOOLEAN;
    nocodb_role_ready BOOLEAN;
    relation_name TEXT;
    column_name TEXT;
    function_signature TEXT;
    append_only_triggers_ready BOOLEAN;
BEGIN
    SELECT COALESCE(executor_role.rolsuper, FALSE)
      INTO executor_is_superuser
      FROM pg_catalog.pg_roles AS executor_role
     WHERE executor_role.rolname = current_user;

    IF NOT executor_is_superuser THEN
        RAISE EXCEPTION
            'database administrator must run 393_eom_missed_call_recovery_runtime_privileges';
    END IF;

    SELECT EXISTS (
        SELECT 1
          FROM pg_catalog.pg_roles AS guard_role
         WHERE guard_role.rolname = 'atlas_eom_handoff_owner'
           AND NOT guard_role.rolcanlogin
           AND NOT guard_role.rolinherit
           AND NOT guard_role.rolsuper
           AND NOT guard_role.rolcreaterole
           AND NOT guard_role.rolcreatedb
           AND NOT guard_role.rolreplication
           AND NOT guard_role.rolbypassrls
           AND NOT EXISTS (
               SELECT 1
                 FROM pg_catalog.pg_roles AS member_role
                WHERE member_role.rolcanlogin
                  AND NOT member_role.rolsuper
                  AND pg_catalog.pg_has_role(
                      member_role.oid, guard_role.oid, 'MEMBER'
                  )
           )
    )
      INTO trusted_guard_role_ready;

    IF NOT trusted_guard_role_ready THEN
        RAISE EXCEPTION
            'atlas_eom_handoff_owner must be a no-login, membership-isolated guard role before running 393_eom_missed_call_recovery_runtime_privileges';
    END IF;

    SELECT EXISTS (
        SELECT 1
          FROM pg_catalog.pg_roles AS runtime_role
         WHERE runtime_role.rolname = 'atlas'
           AND runtime_role.rolcanlogin
    )
      INTO runtime_role_ready;

    IF NOT runtime_role_ready THEN
        RAISE EXCEPTION
            'atlas must be a login runtime role before running 393_eom_missed_call_recovery_runtime_privileges';
    END IF;

    SELECT EXISTS (
        SELECT 1
          FROM pg_catalog.pg_roles AS nocodb_role
         WHERE nocodb_role.rolname = 'atlas_nocodb'
           AND nocodb_role.rolcanlogin
           AND NOT nocodb_role.rolsuper
           AND NOT nocodb_role.rolcreaterole
           AND NOT nocodb_role.rolcreatedb
           AND NOT nocodb_role.rolreplication
           AND NOT nocodb_role.rolbypassrls
           AND NOT nocodb_role.rolinherit
           AND NOT EXISTS (
               SELECT 1
                 FROM pg_catalog.pg_auth_members AS membership
                WHERE membership.member = nocodb_role.oid
           )
    )
      INTO nocodb_role_ready;

    IF NOT nocodb_role_ready THEN
        RAISE EXCEPTION
            'atlas_nocodb must be an unprivileged LOGIN NOINHERIT role before running 393_eom_missed_call_recovery_runtime_privileges';
    END IF;

    FOR relation_name IN
        SELECT unnest(ARRAY[
            'eom_missed_call_operation_receipts',
            'eom_missed_call_attempts',
            'eom_missed_call_contact_suppressions',
            'eom_missed_call_sequences',
            'eom_missed_call_sequence_steps',
            'eom_missed_call_sequence_events'
        ])
    LOOP
        IF to_regclass(format('%I.%I', schema_name, relation_name)) IS NULL THEN
            RAISE EXCEPTION
                'required EOM missed-call relation %.% is absent before privilege repair',
                schema_name,
                relation_name;
        END IF;
    END LOOP;

    -- UPDATE is needed for the worker's existing FOR UPDATE locks. Do not
    -- grant it until the immutable-evidence triggers from migration 389 are
    -- present, enabled, and still bound to their rejecting functions.
    SELECT EXISTS (
        SELECT 1
          FROM pg_catalog.pg_trigger AS trigger
         WHERE trigger.tgrelid = to_regclass(
                   format(
                       '%I.%I',
                       schema_name,
                       'eom_missed_call_operation_receipts'
                   )
               )
           AND trigger.tgname = 'trg_prevent_eom_missed_call_operation_receipt_mutation'
           AND trigger.tgfoid = to_regprocedure(
               format(
                   '%I.%I()',
                   schema_name,
                   'prevent_eom_missed_call_operation_receipt_mutation'
               )
           )
           AND trigger.tgtype = 27
           AND trigger.tgenabled = 'O'
           AND NOT trigger.tgisinternal
    )
    AND EXISTS (
        SELECT 1
          FROM pg_catalog.pg_trigger AS trigger
         WHERE trigger.tgrelid = to_regclass(
                   format('%I.%I', schema_name, 'eom_missed_call_attempts')
               )
           AND trigger.tgname = 'trg_prevent_eom_missed_call_attempt_mutation'
           AND trigger.tgfoid = to_regprocedure(
               format(
                   '%I.%I()',
                   schema_name,
                   'prevent_eom_missed_call_attempt_mutation'
               )
           )
           AND trigger.tgtype = 27
           AND trigger.tgenabled = 'O'
           AND NOT trigger.tgisinternal
    )
      INTO append_only_triggers_ready;

    IF NOT append_only_triggers_ready THEN
        RAISE EXCEPTION
            'append-only receipt and attempt triggers must be intact before running 393_eom_missed_call_recovery_runtime_privileges';
    END IF;

    -- PostgreSQL requires the target owner to hold CREATE on the schema. The
    -- guard remains no-login and membership-isolated, so this grant does not
    -- create an executable path for Atlas or NocoDB.
    EXECUTE format(
        'GRANT USAGE, CREATE ON SCHEMA %I TO atlas_eom_handoff_owner',
        schema_name
    );

    FOR relation_name IN
        SELECT unnest(ARRAY[
            'eom_missed_call_operation_receipts',
            'eom_missed_call_attempts',
            'eom_missed_call_contact_suppressions',
            'eom_missed_call_sequences',
            'eom_missed_call_sequence_steps',
            'eom_missed_call_sequence_events'
        ])
    LOOP
        EXECUTE format(
            'REVOKE ALL PRIVILEGES ON TABLE %I.%I FROM PUBLIC',
            schema_name,
            relation_name
        );
        EXECUTE format(
            'REVOKE ALL PRIVILEGES ON TABLE %I.%I FROM atlas_nocodb',
            schema_name,
            relation_name
        );
        -- Rebuild the runtime allowlist below rather than preserving a stale
        -- DBA-era direct grant such as DELETE after ownership changes.
        EXECUTE format(
            'REVOKE ALL PRIVILEGES ON TABLE %I.%I FROM atlas',
            schema_name,
            relation_name
        );
        -- Table and column ACLs are separate PostgreSQL catalogs. Rebuild the
        -- entire direct access surface so an old column grant cannot survive
        -- the table-level revoke and expose recovery evidence to NocoDB.
        FOR column_name IN
            SELECT attribute.attname
              FROM pg_catalog.pg_attribute AS attribute
             WHERE attribute.attrelid = to_regclass(
                       format('%I.%I', schema_name, relation_name)
                   )
               AND attribute.attnum > 0
               AND NOT attribute.attisdropped
        LOOP
            EXECUTE format(
                'REVOKE ALL PRIVILEGES (%I) ON TABLE %I.%I FROM PUBLIC',
                column_name,
                schema_name,
                relation_name
            );
            EXECUTE format(
                'REVOKE ALL PRIVILEGES (%I) ON TABLE %I.%I FROM atlas_nocodb',
                column_name,
                schema_name,
                relation_name
            );
            EXECUTE format(
                'REVOKE ALL PRIVILEGES (%I) ON TABLE %I.%I FROM atlas',
                column_name,
                schema_name,
                relation_name
            );
        END LOOP;
        EXECUTE format(
            'ALTER TABLE %I.%I OWNER TO atlas_eom_handoff_owner',
            schema_name,
            relation_name
        );
    END LOOP;

    -- The guarded trigger chain reads CRM evidence and uses a contact row lock.
    -- Its owner cannot log in and receives no broad CRM write capability.
    EXECUTE format(
        'GRANT SELECT, UPDATE ON TABLE %I.contacts TO atlas_eom_handoff_owner',
        schema_name
    );
    EXECUTE format(
        'GRANT SELECT ON TABLE %I.contact_interactions TO atlas_eom_handoff_owner',
        schema_name
    );

    -- These six functions are the only recovery helpers reached from CRM table
    -- triggers. Run them as the isolated owner with a deterministic search path
    -- so an allowed NocoDB CRM mutation can cancel a sequence without granting
    -- NocoDB direct access to recovery evidence.
    EXECUTE format(
        'ALTER FUNCTION %I.cancel_eom_missed_call_sequences_for_contact(UUID, VARCHAR, VARCHAR) SECURITY DEFINER',
        schema_name
    );
    EXECUTE format(
        'ALTER FUNCTION %I.cancel_eom_missed_call_sequences_for_contact(UUID, VARCHAR, VARCHAR) SET search_path = pg_catalog, %I, pg_temp',
        schema_name,
        schema_name
    );
    EXECUTE format(
        'REVOKE ALL ON FUNCTION %I.cancel_eom_missed_call_sequences_for_contact(UUID, VARCHAR, VARCHAR) FROM PUBLIC',
        schema_name
    );
    EXECUTE format(
        'ALTER FUNCTION %I.cancel_eom_missed_call_sequences_for_contact(UUID, VARCHAR, VARCHAR) OWNER TO atlas_eom_handoff_owner',
        schema_name
    );

    EXECUTE format(
        'ALTER FUNCTION %I.lock_eom_missed_call_interaction_contact() SECURITY DEFINER',
        schema_name
    );
    EXECUTE format(
        'ALTER FUNCTION %I.lock_eom_missed_call_interaction_contact() SET search_path = pg_catalog, %I, pg_temp',
        schema_name,
        schema_name
    );
    EXECUTE format(
        'REVOKE ALL ON FUNCTION %I.lock_eom_missed_call_interaction_contact() FROM PUBLIC',
        schema_name
    );
    EXECUTE format(
        'ALTER FUNCTION %I.lock_eom_missed_call_interaction_contact() OWNER TO atlas_eom_handoff_owner',
        schema_name
    );

    EXECUTE format(
        'ALTER FUNCTION %I.eom_missed_call_effective_recipient(UUID, TEXT) SECURITY DEFINER',
        schema_name
    );
    EXECUTE format(
        'ALTER FUNCTION %I.eom_missed_call_effective_recipient(UUID, TEXT) SET search_path = pg_catalog, %I, pg_temp',
        schema_name,
        schema_name
    );
    EXECUTE format(
        'REVOKE ALL ON FUNCTION %I.eom_missed_call_effective_recipient(UUID, TEXT) FROM PUBLIC',
        schema_name
    );
    EXECUTE format(
        'ALTER FUNCTION %I.eom_missed_call_effective_recipient(UUID, TEXT) OWNER TO atlas_eom_handoff_owner',
        schema_name
    );

    EXECUTE format(
        'ALTER FUNCTION %I.cancel_eom_missed_call_on_recipient_change(UUID) SECURITY DEFINER',
        schema_name
    );
    EXECUTE format(
        'ALTER FUNCTION %I.cancel_eom_missed_call_on_recipient_change(UUID) SET search_path = pg_catalog, %I, pg_temp',
        schema_name,
        schema_name
    );
    EXECUTE format(
        'REVOKE ALL ON FUNCTION %I.cancel_eom_missed_call_on_recipient_change(UUID) FROM PUBLIC',
        schema_name
    );
    EXECUTE format(
        'ALTER FUNCTION %I.cancel_eom_missed_call_on_recipient_change(UUID) OWNER TO atlas_eom_handoff_owner',
        schema_name
    );

    EXECUTE format(
        'ALTER FUNCTION %I.cancel_eom_missed_call_on_contact_change() SECURITY DEFINER',
        schema_name
    );
    EXECUTE format(
        'ALTER FUNCTION %I.cancel_eom_missed_call_on_contact_change() SET search_path = pg_catalog, %I, pg_temp',
        schema_name,
        schema_name
    );
    EXECUTE format(
        'REVOKE ALL ON FUNCTION %I.cancel_eom_missed_call_on_contact_change() FROM PUBLIC',
        schema_name
    );
    EXECUTE format(
        'ALTER FUNCTION %I.cancel_eom_missed_call_on_contact_change() OWNER TO atlas_eom_handoff_owner',
        schema_name
    );

    EXECUTE format(
        'ALTER FUNCTION %I.cancel_eom_missed_call_on_interaction() SECURITY DEFINER',
        schema_name
    );
    EXECUTE format(
        'ALTER FUNCTION %I.cancel_eom_missed_call_on_interaction() SET search_path = pg_catalog, %I, pg_temp',
        schema_name,
        schema_name
    );
    EXECUTE format(
        'REVOKE ALL ON FUNCTION %I.cancel_eom_missed_call_on_interaction() FROM PUBLIC',
        schema_name
    );
    EXECUTE format(
        'ALTER FUNCTION %I.cancel_eom_missed_call_on_interaction() OWNER TO atlas_eom_handoff_owner',
        schema_name
    );

    -- REVOKE FROM PUBLIC does not remove an explicit old NocoDB EXECUTE
    -- grant. Clear that direct path on all SECURITY DEFINER bridge helpers.
    FOR function_signature IN
        SELECT unnest(ARRAY[
            'cancel_eom_missed_call_sequences_for_contact(UUID, VARCHAR, VARCHAR)',
            'lock_eom_missed_call_interaction_contact()',
            'eom_missed_call_effective_recipient(UUID, TEXT)',
            'cancel_eom_missed_call_on_recipient_change(UUID)',
            'cancel_eom_missed_call_on_contact_change()',
            'cancel_eom_missed_call_on_interaction()'
        ])
    LOOP
        EXECUTE format(
            'REVOKE ALL ON FUNCTION %I.%s FROM atlas_nocodb',
            schema_name,
            function_signature
        );
    END LOOP;

    -- Direct runtime access intentionally remains narrower than object owner
    -- access. UPDATE on immutable receipt/attempt rows is required for the
    -- existing SELECT ... FOR UPDATE locks; their append-only triggers reject
    -- any attempted data mutation.
    EXECUTE format('GRANT USAGE ON SCHEMA %I TO atlas', schema_name);
    EXECUTE format(
        'GRANT SELECT, INSERT, UPDATE ON TABLE %I.eom_missed_call_operation_receipts TO atlas',
        schema_name
    );
    EXECUTE format(
        'GRANT SELECT, INSERT, UPDATE ON TABLE %I.eom_missed_call_attempts TO atlas',
        schema_name
    );
    EXECUTE format(
        'GRANT SELECT, INSERT ON TABLE %I.eom_missed_call_contact_suppressions TO atlas',
        schema_name
    );
    EXECUTE format(
        'GRANT SELECT, INSERT, UPDATE ON TABLE %I.eom_missed_call_sequences TO atlas',
        schema_name
    );
    EXECUTE format(
        'GRANT SELECT, INSERT, UPDATE ON TABLE %I.eom_missed_call_sequence_steps TO atlas',
        schema_name
    );
    EXECUTE format(
        'GRANT SELECT, INSERT ON TABLE %I.eom_missed_call_sequence_events TO atlas',
        schema_name
    );
END;
$$;
