-- Keep the EOM handoff guards outside the Atlas/NocoDB runtime login.
--
-- This migration must run through a database administrator or a role with
-- CREATEROLE plus membership in atlas_eom_handoff_owner. It deliberately fails
-- rather than leaving the protected table owned by a shared operator login.

DO $$
DECLARE
    schema_name TEXT := current_schema();
    runtime_role TEXT := current_user;
    runtime_is_superuser BOOLEAN;
    runtime_can_create_role BOOLEAN;
    runtime_is_guard_member BOOLEAN;
    runtime_has_guard_admin BOOLEAN;
    table_row RECORD;
    sequence_row RECORD;
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
        SELECT 1 FROM pg_roles WHERE rolname = 'atlas_nocodb'
    ) THEN
        IF NOT runtime_is_superuser AND NOT runtime_can_create_role THEN
            RAISE EXCEPTION
                'bootstrap atlas_nocodb with a database administrator before running this migration';
        END IF;
        CREATE ROLE atlas_nocodb NOLOGIN NOINHERIT;
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

    -- NocoDB retains its existing ordinary CRM access, but never evidence or
    -- handoff mutation rights. It also receives no schema ownership/DDL right.
    EXECUTE format('REVOKE CREATE ON SCHEMA %I FROM PUBLIC', schema_name);
    EXECUTE format('GRANT USAGE ON SCHEMA %I TO atlas_eom_handoff_owner', schema_name);
    EXECUTE format('GRANT USAGE ON SCHEMA %I TO atlas_nocodb', schema_name);
    FOR table_row IN
        SELECT tablename
        FROM pg_tables
        WHERE schemaname = schema_name
          AND tablename NOT IN (
              'eom_customer_handoffs',
              'eom_lead_lifecycle_events'
          )
    LOOP
        EXECUTE format(
            'GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE %I.%I TO atlas_nocodb',
            schema_name,
            table_row.tablename
        );
    END LOOP;
    FOR sequence_row IN
        SELECT relname
        FROM pg_class
        WHERE relnamespace = to_regnamespace(schema_name)
          AND relkind = 'S'
    LOOP
        EXECUTE format(
            'GRANT USAGE, SELECT, UPDATE ON SEQUENCE %I.%I TO atlas_nocodb',
            schema_name,
            sequence_row.relname
        );
    END LOOP;

    EXECUTE format('REVOKE ALL PRIVILEGES ON TABLE %I.eom_customer_handoffs FROM PUBLIC', schema_name);
    EXECUTE format('REVOKE ALL PRIVILEGES ON TABLE %I.eom_lead_lifecycle_events FROM PUBLIC', schema_name);
    EXECUTE format('REVOKE ALL PRIVILEGES ON TABLE %I.eom_customer_handoffs FROM atlas_nocodb', schema_name);
    EXECUTE format('REVOKE ALL PRIVILEGES ON TABLE %I.eom_lead_lifecycle_events FROM atlas_nocodb', schema_name);
END;
$$;

ALTER TABLE eom_customer_handoffs OWNER TO atlas_eom_handoff_owner;
ALTER FUNCTION require_eom_customer_handoff_finalization() OWNER TO atlas_eom_handoff_owner;
ALTER FUNCTION prevent_eom_customer_handoff_mutation() OWNER TO atlas_eom_handoff_owner;

DO $$
BEGIN
    -- The temporary admin membership lets a non-superuser migration executor
    -- transfer ownership, then is removed before the application serves.
    EXECUTE format('REVOKE atlas_eom_handoff_owner FROM %I', current_user);
END;
$$;
