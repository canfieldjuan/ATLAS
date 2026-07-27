-- atlas: atomic-bookkeeping
-- Keep the EOM handoff guards outside the Atlas/NocoDB runtime login.
--
-- This migration must run through a database administrator or a role with
-- CREATEROLE plus membership in atlas_eom_handoff_owner. The administrator
-- that granted a non-superuser's temporary membership must revoke it after the
-- atomic migration commits; the enabled startup guard refuses to serve until
-- that happens. This deliberately fails closed rather than leaving the
-- protected table owned by a shared operator login.

DO $$
DECLARE
    schema_name TEXT := current_schema();
    runtime_role TEXT := current_user;
    runtime_is_superuser BOOLEAN;
    runtime_can_create_role BOOLEAN;
    runtime_is_guard_member BOOLEAN;
    runtime_has_guard_admin BOOLEAN;
    runtime_owns_schema BOOLEAN;
    runtime_owns_all_tables BOOLEAN;
    runtime_owns_protected_functions BOOLEAN;
    table_row RECORD;
BEGIN
    SELECT rolsuper, rolcreaterole
    INTO runtime_is_superuser, runtime_can_create_role
    FROM pg_roles
    WHERE rolname = runtime_role;

    IF NOT EXISTS (
        SELECT 1 FROM pg_roles WHERE rolname = 'atlas_eom_handoff_owner'
    ) THEN
        IF NOT runtime_is_superuser AND NOT runtime_can_create_role THEN
            RAISE EXCEPTION
                'bootstrap atlas_eom_handoff_owner with a database administrator before running this migration';
        END IF;
        CREATE ROLE atlas_eom_handoff_owner NOLOGIN NOINHERIT;
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM pg_roles AS nocodb_role
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
              FROM pg_auth_members AS membership
              WHERE membership.member = nocodb_role.oid
          )
    ) THEN
        RAISE EXCEPTION
            'database administrator must provision atlas_nocodb as an unprivileged, membership-free LOGIN NOINHERIT role with its NocoDB password before running this migration';
    END IF;

    SELECT pg_has_role(runtime_role, 'atlas_eom_handoff_owner', 'MEMBER')
    INTO runtime_is_guard_member;
    SELECT COALESCE(admin_option, FALSE)
    INTO runtime_has_guard_admin
    FROM pg_auth_members AS membership
    JOIN pg_roles AS member_role ON member_role.oid = membership.member
    JOIN pg_roles AS guard_role ON guard_role.oid = membership.roleid
    WHERE member_role.rolname = runtime_role
      AND guard_role.rolname = 'atlas_eom_handoff_owner';
    runtime_has_guard_admin := COALESCE(runtime_has_guard_admin, FALSE);

    IF NOT runtime_is_superuser AND (
        NOT runtime_is_guard_member OR NOT runtime_has_guard_admin
    ) THEN
        RAISE EXCEPTION
            'migration executor % must be a superuser or an admin member of atlas_eom_handoff_owner',
            runtime_role;
    END IF;

    -- This migration revokes direct NocoDB grants on every current table and
    -- transfers two protected functions. A non-superuser must own each object
    -- it changes; fail before making any partial privilege change otherwise.
    SELECT schema_row.nspowner = runtime_role_row.oid
    INTO runtime_owns_schema
    FROM pg_namespace AS schema_row
    JOIN pg_roles AS runtime_role_row ON runtime_role_row.rolname = runtime_role
    WHERE schema_row.nspname = schema_name;
    runtime_owns_schema := COALESCE(runtime_owns_schema, FALSE);

    SELECT NOT EXISTS (
        SELECT 1
        FROM pg_tables AS schema_table
        JOIN pg_class AS table_class
          ON table_class.oid = to_regclass(
              format('%I.%I', schema_name, schema_table.tablename)
          )
        JOIN pg_roles AS runtime_role_row ON runtime_role_row.rolname = runtime_role
        WHERE schema_table.schemaname = schema_name
          AND table_class.relowner <> runtime_role_row.oid
    )
    INTO runtime_owns_all_tables;

    SELECT COUNT(*) = 2
       AND BOOL_AND(protected_function.proowner = runtime_role_row.oid)
    INTO runtime_owns_protected_functions
    FROM pg_proc AS protected_function
    JOIN pg_namespace AS function_schema
      ON function_schema.oid = protected_function.pronamespace
    JOIN pg_roles AS runtime_role_row ON runtime_role_row.rolname = runtime_role
    WHERE function_schema.nspname = schema_name
      AND protected_function.proname IN (
          'require_eom_customer_handoff_finalization',
          'prevent_eom_customer_handoff_mutation'
      );
    runtime_owns_protected_functions := COALESCE(runtime_owns_protected_functions, FALSE);

    IF NOT runtime_is_superuser AND (
        NOT runtime_owns_schema
        OR NOT runtime_owns_all_tables
        OR NOT runtime_owns_protected_functions
    ) THEN
        RAISE EXCEPTION
            'migration executor % must own the schema, every current table, and protected handoff functions',
            runtime_role;
    END IF;

    -- Preserve Atlas finalization access explicitly before ownership moves.
    EXECUTE format(
        'GRANT SELECT, INSERT, UPDATE, DELETE, TRUNCATE ON TABLE %I.eom_customer_handoffs TO %I',
        schema_name,
        runtime_role
    );
    EXECUTE format(
        'GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE %I.eom_lead_lifecycle_events TO %I',
        schema_name,
        runtime_role
    );

    -- NocoDB is a browser UI for only the documented CRM tables. It receives
    -- neither generic database-operator access nor evidence/DDL authority.
    EXECUTE format('REVOKE CREATE ON SCHEMA %I FROM PUBLIC', schema_name);
    -- PostgreSQL requires the prospective owner to hold CREATE on this schema
    -- before a non-superuser executor can transfer protected-object ownership.
    EXECUTE format(
        'GRANT USAGE, CREATE ON SCHEMA %I TO atlas_eom_handoff_owner',
        schema_name
    );
    EXECUTE format('GRANT USAGE ON SCHEMA %I TO atlas_nocodb', schema_name);
    FOR table_row IN
        SELECT tablename
        FROM pg_tables
        WHERE schemaname = schema_name
    LOOP
        -- Remove grants made by older versions of this migration as well as
        -- any ad-hoc direct grant. The allowlist below is the entire UI scope.
        EXECUTE format(
            'REVOKE ALL PRIVILEGES ON TABLE %I.%I FROM atlas_nocodb',
            schema_name,
            table_row.tablename
        );
    END LOOP;
    EXECUTE format(
        'REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA %I FROM atlas_nocodb',
        schema_name
    );

    -- EOM's NocoDB setup documents exactly these CRM tables. UUID defaults
    -- mean none of them needs a sequence grant.
    FOR table_row IN
        SELECT tablename
        FROM pg_tables
        WHERE schemaname = schema_name
          AND tablename IN ('contacts', 'contact_interactions', 'appointments')
    LOOP
        IF table_row.tablename = 'contacts' THEN
            -- EOM ownership/type/stage are lifecycle authority, not ordinary
            -- CRM-edit fields. NocoDB can still create and edit generic CRM
            -- data, but cannot write those columns directly.
            EXECUTE format(
                'REVOKE INSERT, UPDATE ON TABLE %I.contacts FROM atlas_nocodb',
                schema_name
            );
            EXECUTE format(
                'GRANT SELECT, DELETE ON TABLE %I.contacts TO atlas_nocodb',
                schema_name
            );
            EXECUTE format(
                'GRANT INSERT (id, full_name, first_name, last_name, email, phone, '
                || 'address, city, state, zip, status, tags, notes, source, source_ref, '
                || 'lead_owner, next_follow_up_at, created_at, updated_at, metadata) '
                || 'ON TABLE %I.contacts TO atlas_nocodb',
                schema_name
            );
            EXECUTE format(
                'GRANT UPDATE (full_name, first_name, last_name, email, phone, '
                || 'address, city, state, zip, status, tags, notes, source, source_ref, '
                || 'lead_owner, next_follow_up_at, updated_at, metadata) '
                || 'ON TABLE %I.contacts TO atlas_nocodb',
                schema_name
            );
        ELSE
            EXECUTE format(
                'GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE %I.%I TO atlas_nocodb',
                schema_name,
                table_row.tablename
            );
        END IF;
    END LOOP;
END;
$$;

ALTER TABLE eom_customer_handoffs OWNER TO atlas_eom_handoff_owner;
ALTER FUNCTION require_eom_customer_handoff_finalization() OWNER TO atlas_eom_handoff_owner;
ALTER FUNCTION prevent_eom_customer_handoff_mutation() OWNER TO atlas_eom_handoff_owner;

-- A non-superuser cannot revoke a membership whose grantor is the database
-- administrator. Do not pretend otherwise: the full-app startup preflight
-- blocks while that membership remains, and the grantor revokes it only after
-- this atomic migration and its ledger row have committed.
