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
    runtime_role TEXT := NULLIF(
        btrim(
            pg_catalog.current_setting(
                'atlas.eom_missed_call_recovery_runtime_role',
                TRUE
            )
        ),
        ''
    );
    executor_is_superuser BOOLEAN;
    trusted_guard_role_ready BOOLEAN;
    runtime_role_ready BOOLEAN;
    nocodb_role_ready BOOLEAN;
    relation_name TEXT;
    column_name TEXT;
    function_signature TEXT;
    acl_role TEXT;
    expected_function_language TEXT;
    expected_function_body_sha256 TEXT;
    observed_function_language TEXT;
    observed_function_body TEXT;
    observed_function_body_sha256 TEXT;
    pgcrypto_schema TEXT;
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

    IF runtime_role IS NULL THEN
        RAISE EXCEPTION
            'configured EOM runtime role is required before running 393_eom_missed_call_recovery_runtime_privileges';
    END IF;

    SELECT EXISTS (
        SELECT 1
          FROM pg_catalog.pg_roles AS runtime_role_state
         WHERE runtime_role_state.rolname = runtime_role
           AND runtime_role_state.rolcanlogin
           AND runtime_role_state.rolname <> 'atlas_nocodb'
           AND NOT runtime_role_state.rolsuper
           AND NOT runtime_role_state.rolcreaterole
           AND NOT runtime_role_state.rolcreatedb
           AND NOT runtime_role_state.rolreplication
           AND NOT runtime_role_state.rolbypassrls
           AND NOT EXISTS (
               SELECT 1
                 FROM pg_catalog.pg_auth_members AS membership
                 JOIN pg_catalog.pg_roles AS guard_role
                   ON guard_role.oid = membership.roleid
                WHERE membership.member = runtime_role_state.oid
                  AND guard_role.rolname = 'atlas_eom_handoff_owner'
           )
           AND NOT EXISTS (
               SELECT 1
                 FROM pg_catalog.pg_roles AS delegating_login
                WHERE delegating_login.rolcanlogin
                  AND NOT delegating_login.rolsuper
                  AND delegating_login.oid <> runtime_role_state.oid
                  AND pg_catalog.pg_has_role(
                      delegating_login.oid, runtime_role_state.oid, 'MEMBER'
                  )
           )
    )
      INTO runtime_role_ready;

    IF NOT runtime_role_ready THEN
        RAISE EXCEPTION
            'configured EOM runtime role % must be an unprivileged, guard-isolated login before running 393_eom_missed_call_recovery_runtime_privileges',
            runtime_role;
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
           AND NOT EXISTS (
               SELECT 1
                 FROM pg_catalog.pg_roles AS delegating_login
                WHERE delegating_login.rolcanlogin
                  AND NOT delegating_login.rolsuper
                  AND delegating_login.oid <> nocodb_role.oid
                  AND pg_catalog.pg_has_role(
                      delegating_login.oid, nocodb_role.oid, 'MEMBER'
                  )
           )
    )
      INTO nocodb_role_ready;

    IF NOT nocodb_role_ready THEN
        RAISE EXCEPTION
            'atlas_nocodb must be an unprivileged LOGIN NOINHERIT role before running 393_eom_missed_call_recovery_runtime_privileges';
    END IF;

    -- The protected DBA is the only actor allowed to establish this
    -- database-level prerequisite. Do it only after every role boundary is
    -- valid, rather than relying on an undocumented deployment or test fixture.
    EXECUTE 'CREATE EXTENSION IF NOT EXISTS pgcrypto';

    -- 393 is the first migration to give the CRM bridge functions definer
    -- authority. Never elevate an object merely because its signature still
    -- exists: its stored body and language must be the exact trusted migration-
    -- 389 source before this migration changes its owner or security mode.
    SELECT namespace_state.nspname
      INTO pgcrypto_schema
      FROM pg_catalog.pg_extension AS extension_state
      JOIN pg_catalog.pg_namespace AS namespace_state
        ON namespace_state.oid = extension_state.extnamespace
     WHERE extension_state.extname = 'pgcrypto'
     LIMIT 1;

    IF pgcrypto_schema IS NULL THEN
        RAISE EXCEPTION
            'pgcrypto SHA-256 support is required before elevating EOM missed-call bridge functions';
    END IF;

    FOR function_signature, expected_function_language, expected_function_body_sha256 IN
        SELECT expected_function.signature,
               expected_function.language_name,
               expected_function.body_sha256
          FROM (
              VALUES
                  (
                      'cancel_eom_missed_call_sequences_for_contact(UUID, VARCHAR, VARCHAR)',
                      'plpgsql',
                      'd4c08e5d18af991621648f89297fb83170f7355948c52f5da85749cfeb0e535b'
                  ),
                  (
                      'lock_eom_missed_call_interaction_contact()',
                      'plpgsql',
                      'f7b6b3a3057c2b68b46c1854a64a693105bda4732d0dea5b8c772400d314e18e'
                  ),
                  (
                      'eom_missed_call_effective_recipient(UUID, TEXT)',
                      'sql',
                      '128b147313d03f1bf05a4ce086e225f09647078dfdf43bced4f2b47c0d3f4371'
                  ),
                  (
                      'cancel_eom_missed_call_on_recipient_change(UUID)',
                      'plpgsql',
                      '9f099f30ae2303e1f8610217222d40f1bde51ea5adbc2964b72b4b2581c18c0f'
                  ),
                  (
                      'cancel_eom_missed_call_on_contact_change()',
                      'plpgsql',
                      'ae2cac20a094b376008dc249098aafef8592aa46ea4e224ba6effc8271cebe8d'
                  ),
                  (
                      'cancel_eom_missed_call_on_interaction()',
                      'plpgsql',
                      '0751a416b37e46a3d1960edbc6f4c65496b46465244bae04eeeaf45bd555feee'
                  )
          ) AS expected_function(signature, language_name, body_sha256)
    LOOP
        SELECT procedure.prosrc, language_state.lanname
          INTO observed_function_body, observed_function_language
          FROM pg_catalog.pg_proc AS procedure
          JOIN pg_catalog.pg_language AS language_state
            ON language_state.oid = procedure.prolang
         WHERE procedure.oid = pg_catalog.to_regprocedure(
                   format('%I.%s', schema_name, function_signature)
               );

        IF observed_function_body IS NULL
           OR observed_function_language IS DISTINCT FROM expected_function_language THEN
            RAISE EXCEPTION
                'required EOM missed-call bridge function % is not the trusted migration-389 function',
                function_signature;
        END IF;

        EXECUTE format(
            'SELECT encode(%1$I.digest($1::text, ''sha256''), ''hex'')',
            pgcrypto_schema
        )
          INTO observed_function_body_sha256
          USING observed_function_body;

        IF observed_function_body_sha256 IS DISTINCT FROM expected_function_body_sha256 THEN
            RAISE EXCEPTION
                'required EOM missed-call bridge function % does not match its trusted migration-389 body',
                function_signature;
        END IF;
    END LOOP;

    -- PostgreSQL preserves a trigger function's OID across CREATE OR REPLACE
    -- FUNCTION, so trigger bindings cannot prove that their bodies still
    -- enforce the recovery append-only boundary by themselves.
    FOR function_signature, expected_function_language, expected_function_body_sha256 IN
        SELECT expected_function.signature,
               expected_function.language_name,
               expected_function.body_sha256
          FROM (
              VALUES
                  (
                      'prevent_eom_missed_call_operation_receipt_mutation()',
                      'plpgsql',
                      '7bc869fec7fe5493b5859da8ab1a26da547d53db518b6e32fa2ab9e3cd9d08a7'
                  ),
                  (
                      'prevent_eom_missed_call_attempt_mutation()',
                      'plpgsql',
                      '7647fbfe7a9642e0434e0a0e3930aa221f6d93323205cd0633ba809cf31c579d'
                  ),
                  (
                      'prevent_eom_missed_call_sequence_event_mutation()',
                      'plpgsql',
                      '3f88a357faf419060b4c01046a49e06be87302e573fe831f8ab1f5c9dfe072a4'
                  ),
                  (
                      'prevent_eom_missed_call_suppression_mutation()',
                      'plpgsql',
                      'eb3762cb1c676d7527d3637c8850920c2f486dd30b34a6ca541f77072c618528'
                  )
          ) AS expected_function(signature, language_name, body_sha256)
    LOOP
        SELECT procedure.prosrc, language_state.lanname
          INTO observed_function_body, observed_function_language
          FROM pg_catalog.pg_proc AS procedure
          JOIN pg_catalog.pg_language AS language_state
            ON language_state.oid = procedure.prolang
         WHERE procedure.oid = pg_catalog.to_regprocedure(
                   format('%I.%s', schema_name, function_signature)
               );

        IF observed_function_body IS NULL
           OR observed_function_language IS DISTINCT FROM expected_function_language THEN
            RAISE EXCEPTION
                'required EOM missed-call append-only fence function % is not the trusted migration-389 function',
                function_signature;
        END IF;

        EXECUTE format(
            'SELECT encode(%1$I.digest($1::text, ''sha256''), ''hex'')',
            pgcrypto_schema
        )
          INTO observed_function_body_sha256
          USING observed_function_body;

        IF observed_function_body_sha256 IS DISTINCT FROM expected_function_body_sha256 THEN
            RAISE EXCEPTION
                'required EOM missed-call append-only fence function % does not match its trusted migration-389 body',
                function_signature;
        END IF;
    END LOOP;

    -- Scope validators run on every recovery write, and the inbound-SMS helper
    -- runs inside a SECURITY DEFINER CRM trigger. They therefore share the
    -- guard boundary even though they are not independently security-definer.
    FOR function_signature, expected_function_language, expected_function_body_sha256 IN
        SELECT expected_function.signature,
               expected_function.language_name,
               expected_function.body_sha256
          FROM (
              VALUES
                  (
                      'validate_eom_missed_call_contact_scope()',
                      'plpgsql',
                      '685cbc52f37986f0603795132bd1f71962f40912e3e294194f08ade00174fad9'
                  ),
                  (
                      'validate_eom_missed_call_sequence_scope()',
                      'plpgsql',
                      '7d6e532059f3a7862eb8fe339c0166766d7188c71e71526c49f65fd25395152d'
                  ),
                  (
                      'eom_missed_call_has_proven_inbound_sms(JSONB)',
                      'sql',
                      '8e988b596b85d2e62de3cfb2f49ec6463b8a38116855809db78801c735890567'
                  )
          ) AS expected_function(signature, language_name, body_sha256)
    LOOP
        SELECT procedure.prosrc, language_state.lanname
          INTO observed_function_body, observed_function_language
          FROM pg_catalog.pg_proc AS procedure
          JOIN pg_catalog.pg_language AS language_state
            ON language_state.oid = procedure.prolang
         WHERE procedure.oid = pg_catalog.to_regprocedure(
                   format('%I.%s', schema_name, function_signature)
               );

        IF observed_function_body IS NULL
           OR observed_function_language IS DISTINCT FROM expected_function_language THEN
            RAISE EXCEPTION
                'required EOM missed-call guard function % is not the trusted migration-389 function',
                function_signature;
        END IF;

        EXECUTE format(
            'SELECT encode(%1$I.digest($1::text, ''sha256''), ''hex'')',
            pgcrypto_schema
        )
          INTO observed_function_body_sha256
          USING observed_function_body;

        IF observed_function_body_sha256 IS DISTINCT FROM expected_function_body_sha256 THEN
            RAISE EXCEPTION
                'required EOM missed-call guard function % does not match its trusted migration-389 body',
                function_signature;
        END IF;
    END LOOP;

    -- The application schema can remain writable by the normal runtime for
    -- ordinary migrations. Its guard-owned callbacks therefore cannot rely on
    -- search-path position alone for user-defined function dispatch: a runtime
    -- overload with a closer VARCHAR or unknown-literal signature could win.
    -- The trusted migration-389 bodies above establish the source we are
    -- replacing. Recreate only the callbacks with user-defined calls so every
    -- such dispatch names this schema and its exact trusted argument types.
    EXECUTE format(
        $definition$
        CREATE OR REPLACE FUNCTION %1$I.cancel_eom_missed_call_on_recipient_change(
            target_contact_id UUID
        )
        RETURNS VOID
        LANGUAGE plpgsql
        AS $body$
        DECLARE
            effective_recipient TEXT;
        BEGIN
            SELECT %1$I.eom_missed_call_effective_recipient(
                c.id::UUID,
                c.email::TEXT
            )
              INTO effective_recipient
              FROM contacts AS c
             WHERE c.id = target_contact_id
               AND c.business_context_id = 'effingham_maids';

            IF NOT FOUND THEN
                RETURN;
            END IF;

            IF EXISTS (
                SELECT 1
                  FROM eom_missed_call_sequences AS sequence
                 WHERE sequence.contact_id = target_contact_id
                   AND sequence.state IN ('active', 'blocked_configuration')
                   AND lower(btrim(sequence.recipient_email)) IS DISTINCT FROM
                       effective_recipient
            ) THEN
                PERFORM %1$I.cancel_eom_missed_call_sequences_for_contact(
                    target_contact_id::UUID,
                    'recipient_changed'::VARCHAR,
                    'interaction_trigger'::VARCHAR
                );
            END IF;
        END;
        $body$;
        $definition$,
        schema_name
    );

    EXECUTE format(
        $definition$
        CREATE OR REPLACE FUNCTION %1$I.cancel_eom_missed_call_on_contact_change()
        RETURNS TRIGGER
        LANGUAGE plpgsql
        AS $body$
        DECLARE
            cancellation_reason VARCHAR(64);
        BEGIN
            IF OLD.business_context_id = 'effingham_maids'
               AND NEW.business_context_id IS DISTINCT FROM 'effingham_maids' THEN
                cancellation_reason := 'tenant_changed';
            ELSIF NEW.business_context_id IS DISTINCT FROM 'effingham_maids' THEN
                RETURN NEW;
            ELSIF NEW.contact_type <> 'lead' THEN
                cancellation_reason := 'became_customer';
            ELSIF NEW.status <> 'active' THEN
                cancellation_reason := 'contact_inactive';
            ELSIF NEW.lead_stage IS DISTINCT FROM 'new' THEN
                cancellation_reason := 'lead_advanced';
            ELSIF NEW.customer_type = 'commercial' THEN
                cancellation_reason := 'non_residential';
            ELSIF NEW.email IS DISTINCT FROM OLD.email
               AND EXISTS (
                   SELECT 1
                     FROM eom_missed_call_sequences AS sequence
                    WHERE sequence.contact_id = NEW.id
                      AND sequence.state IN ('active', 'blocked_configuration')
                      AND lower(btrim(sequence.recipient_email)) IS DISTINCT FROM
                          %1$I.eom_missed_call_effective_recipient(
                              NEW.id::UUID,
                              NEW.email::TEXT
                          )
               ) THEN
                cancellation_reason := 'recipient_changed';
            ELSE
                RETURN NEW;
            END IF;

            PERFORM %1$I.cancel_eom_missed_call_sequences_for_contact(
                NEW.id::UUID,
                cancellation_reason::VARCHAR,
                'contact_trigger'::VARCHAR
            );
            RETURN NEW;
        END;
        $body$;
        $definition$,
        schema_name
    );

    EXECUTE format(
        $definition$
        CREATE OR REPLACE FUNCTION %1$I.cancel_eom_missed_call_on_interaction()
        RETURNS TRIGGER
        LANGUAGE plpgsql
        AS $body$
        DECLARE
            cancellation_reason VARCHAR(64);
        BEGIN
            IF TG_OP <> 'INSERT' THEN
                PERFORM %1$I.cancel_eom_missed_call_on_recipient_change(
                    OLD.contact_id::UUID
                );
                IF TG_OP = 'UPDATE'
                   AND NEW.contact_id IS DISTINCT FROM OLD.contact_id THEN
                    PERFORM %1$I.cancel_eom_missed_call_on_recipient_change(
                        NEW.contact_id::UUID
                    );
                END IF;
                IF TG_OP = 'DELETE' THEN
                    RETURN OLD;
                END IF;
                RETURN NEW;
            END IF;

            IF NOT EXISTS (
                SELECT 1
                  FROM contacts
                 WHERE id = NEW.contact_id
                   AND business_context_id = 'effingham_maids'
            ) THEN
                RETURN NEW;
            END IF;

            IF NEW.metadata->>'missed_call_recovery_cancel_reason' IN (
                'callback_recorded', 'response_recorded', 'opt_out', 'manual'
            ) THEN
                cancellation_reason := NEW.metadata->>'missed_call_recovery_cancel_reason';
            ELSIF NEW.interaction_type = 'sms'
               AND %1$I.eom_missed_call_has_proven_inbound_sms(
                   NEW.metadata::JSONB
               ) THEN
                cancellation_reason := 'tracked_inbound_response';
            ELSIF NEW.interaction_type IN (
                'email_inbound', 'lead_response', 'callback_completed',
                'conversation_completed', 'opt_out'
            ) THEN
                cancellation_reason := NEW.interaction_type;
            ELSIF NEW.interaction_type = 'web_form'
               AND NEW.intent = 'estimate_request' THEN
                cancellation_reason := 'new_estimate_request';
            ELSE
                RETURN NEW;
            END IF;

            IF NEW.interaction_type = 'opt_out' THEN
                INSERT INTO eom_missed_call_contact_suppressions (
                    contact_id, reason_code, actor_name, source
                ) VALUES (
                    NEW.contact_id, 'opt_out', 'system', 'interaction_trigger'
                ) ON CONFLICT (contact_id) DO NOTHING;
            END IF;

            IF EXISTS (
                SELECT 1
                FROM eom_missed_call_sequences AS sequence
                WHERE sequence.contact_id = NEW.contact_id
                  AND sequence.state IN ('active', 'blocked_configuration')
                  AND NEW.occurred_at > sequence.created_at
            ) THEN
                PERFORM %1$I.cancel_eom_missed_call_sequences_for_contact(
                    NEW.contact_id::UUID,
                    cancellation_reason::VARCHAR,
                    'interaction_trigger'::VARCHAR
                );
            END IF;
            RETURN NEW;
        END;
        $body$;
        $definition$,
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
        -- Transfer ownership before revoking legacy role ACLs. Revoking an
        -- old owner first creates an explicit empty ACL entry, and PostgreSQL
        -- carries that entry to the isolated guard on ownership transfer. That
        -- would prevent the guard's SECURITY DEFINER validators from reading
        -- their own protected relations.
        EXECUTE format(
            'ALTER TABLE %I.%I OWNER TO atlas_eom_handoff_owner',
            schema_name,
            relation_name
        );
        -- An old owner can carry an explicit empty ACL through ownership
        -- transfer. Make the no-login guard's table authority explicit before
        -- rebuilding the direct-grantee allowlist below.
        EXECUTE format(
            'GRANT ALL PRIVILEGES ON TABLE %I.%I TO atlas_eom_handoff_owner',
            schema_name,
            relation_name
        );
        EXECUTE format(
            'REVOKE ALL PRIVILEGES ON TABLE %I.%I FROM PUBLIC',
            schema_name,
            relation_name
        );
        -- Rebuild the direct table allowlist from every catalog grantee, not
        -- just the current runtime's named predecessor. A stale login or a
        -- group inherited by one can otherwise retain recovery evidence access
        -- after ownership transfer.
        FOR acl_role IN
            SELECT DISTINCT grantee_role.rolname
              FROM pg_catalog.pg_class AS relation
              CROSS JOIN LATERAL pg_catalog.aclexplode(
                  COALESCE(
                      relation.relacl,
                      pg_catalog.acldefault('r', relation.relowner)
                  )
              ) AS relation_acl
              JOIN pg_catalog.pg_roles AS grantee_role
                ON grantee_role.oid = relation_acl.grantee
             WHERE relation.oid = to_regclass(
                       format('%I.%I', schema_name, relation_name)
                   )
               AND grantee_role.rolname <> 'atlas_eom_handoff_owner'
        LOOP
            EXECUTE format(
                'REVOKE ALL PRIVILEGES ON TABLE %I.%I FROM %I',
                schema_name,
                relation_name,
                acl_role
            );
        END LOOP;
        -- Table and column ACLs are separate PostgreSQL catalogs. Rebuild the
        -- entire direct access surface so no stale column grant can survive
        -- the table-level rebuild and expose recovery evidence to a login or
        -- one of its inherited groups.
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
            FOR acl_role IN
                SELECT DISTINCT grantee_role.rolname
                  FROM pg_catalog.pg_attribute AS attribute
                  CROSS JOIN LATERAL pg_catalog.aclexplode(
                      attribute.attacl
                  ) AS column_acl
                  JOIN pg_catalog.pg_roles AS grantee_role
                    ON grantee_role.oid = column_acl.grantee
                 WHERE attribute.attrelid = to_regclass(
                           format('%I.%I', schema_name, relation_name)
                       )
                   AND attribute.attname = column_name
                   AND attribute.attnum > 0
                   AND NOT attribute.attisdropped
            LOOP
                EXECUTE format(
                    'REVOKE ALL PRIVILEGES (%I) ON TABLE %I.%I FROM %I',
                    column_name,
                    schema_name,
                    relation_name,
                    acl_role
                );
            END LOOP;
        END LOOP;
    END LOOP;

    -- The remaining recovery functions either enforce immutable/scope fences
    -- or execute inside the definer chain. A normal runtime owner could replace
    -- one after this migration, so transfer every attested function before any
    -- runtime grant is rebuilt.
    FOR function_signature IN
        SELECT unnest(ARRAY[
            'prevent_eom_missed_call_operation_receipt_mutation()',
            'prevent_eom_missed_call_attempt_mutation()',
            'prevent_eom_missed_call_sequence_event_mutation()',
            'prevent_eom_missed_call_suppression_mutation()',
            'validate_eom_missed_call_contact_scope()',
            'validate_eom_missed_call_sequence_scope()',
            'eom_missed_call_has_proven_inbound_sms(JSONB)'
        ])
    LOOP
        EXECUTE format(
            'ALTER FUNCTION %I.%s OWNER TO atlas_eom_handoff_owner',
            schema_name,
            function_signature
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

    -- These table-trigger validators must read contacts while the configured
    -- runtime writes recovery rows. Keep that CRM read under the no-login guard
    -- rather than widening the runtime's direct CRM-table access.
    FOR function_signature IN
        SELECT unnest(ARRAY[
            'validate_eom_missed_call_contact_scope()',
            'validate_eom_missed_call_sequence_scope()'
        ])
    LOOP
        EXECUTE format(
            'ALTER FUNCTION %I.%s SECURITY DEFINER',
            schema_name,
            function_signature
        );
        EXECUTE format(
            'ALTER FUNCTION %I.%s SET search_path = pg_catalog, %I, pg_temp',
            schema_name,
            function_signature,
            schema_name
        );
        EXECUTE format(
            'REVOKE ALL ON FUNCTION %I.%s FROM PUBLIC',
            schema_name,
            function_signature
        );
    END LOOP;

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

    -- REVOKE FROM PUBLIC does not remove an explicit role grant. Rebuild the
    -- complete direct-execution deny surface on every SECURITY DEFINER helper;
    -- CRM table triggers remain the only supported invocation path.
    FOR function_signature IN
        SELECT unnest(ARRAY[
            'cancel_eom_missed_call_sequences_for_contact(UUID, VARCHAR, VARCHAR)',
            'lock_eom_missed_call_interaction_contact()',
            'eom_missed_call_effective_recipient(UUID, TEXT)',
            'cancel_eom_missed_call_on_recipient_change(UUID)',
            'cancel_eom_missed_call_on_contact_change()',
            'cancel_eom_missed_call_on_interaction()',
            'validate_eom_missed_call_contact_scope()',
            'validate_eom_missed_call_sequence_scope()'
        ])
    LOOP
        FOR acl_role IN
            SELECT grantee_role.rolname
              FROM pg_catalog.pg_proc AS procedure
              JOIN pg_catalog.pg_namespace AS function_namespace
                ON function_namespace.oid = procedure.pronamespace
              CROSS JOIN LATERAL pg_catalog.aclexplode(
                  COALESCE(
                      procedure.proacl,
                      pg_catalog.acldefault('f', procedure.proowner)
                  )
              ) AS function_acl
              JOIN pg_catalog.pg_roles AS grantee_role
                ON grantee_role.oid = function_acl.grantee
             WHERE function_namespace.nspname = schema_name
               AND procedure.oid = pg_catalog.to_regprocedure(
                   format('%I.%s', schema_name, function_signature)
               )
               AND function_acl.privilege_type = 'EXECUTE'
               AND grantee_role.rolname <> 'atlas_eom_handoff_owner'
        LOOP
            EXECUTE format(
                'REVOKE ALL ON FUNCTION %I.%s FROM %I',
                schema_name,
                function_signature,
                acl_role
            );
        END LOOP;
    END LOOP;

    -- Direct runtime access intentionally remains narrower than object owner
    -- access. UPDATE on immutable receipt/attempt rows is required for the
    -- existing SELECT ... FOR UPDATE locks; their append-only triggers reject
    -- any attempted data mutation.
    EXECUTE format('GRANT USAGE ON SCHEMA %I TO %I', schema_name, runtime_role);
    EXECUTE format(
        'GRANT SELECT, INSERT, UPDATE ON TABLE %I.eom_missed_call_operation_receipts TO %I',
        schema_name,
        runtime_role
    );
    EXECUTE format(
        'GRANT SELECT, INSERT, UPDATE ON TABLE %I.eom_missed_call_attempts TO %I',
        schema_name,
        runtime_role
    );
    EXECUTE format(
        'GRANT SELECT, INSERT ON TABLE %I.eom_missed_call_contact_suppressions TO %I',
        schema_name,
        runtime_role
    );
    EXECUTE format(
        'GRANT SELECT, INSERT, UPDATE ON TABLE %I.eom_missed_call_sequences TO %I',
        schema_name,
        runtime_role
    );
    EXECUTE format(
        'GRANT SELECT, INSERT, UPDATE ON TABLE %I.eom_missed_call_sequence_steps TO %I',
        schema_name,
        runtime_role
    );
    EXECUTE format(
        'GRANT SELECT, INSERT ON TABLE %I.eom_missed_call_sequence_events TO %I',
        schema_name,
        runtime_role
    );
END;
$$;
