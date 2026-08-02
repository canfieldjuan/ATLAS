-- atlas: atomic-bookkeeping
-- atlas: out-of-band-bootstrap
-- Repair Atlas runtime handoff access after protected ownership transfer.
--
-- Migration 354 predates this repair and may already be recorded in
-- schema_migrations. Keep this as a separate out-of-band bootstrap migration so
-- databases that have applied 354 can still receive the post-owner grant.

DO $$
DECLARE
    schema_name TEXT := current_schema();
    migration_executor TEXT := current_user;
    runtime_role TEXT := NULLIF(
        btrim(current_setting('atlas.eom_funnel_runtime_role', true)),
        ''
    );
    runtime_can_login BOOLEAN;
    runtime_is_superuser BOOLEAN;
    runtime_can_create_role BOOLEAN;
    runtime_can_create_db BOOLEAN;
    runtime_can_bypass_rls BOOLEAN;
    runtime_owns_database BOOLEAN;
    executor_is_superuser BOOLEAN;
    executor_is_guard_member BOOLEAN;
    executor_has_guard_admin BOOLEAN;
    handoff_table REGCLASS := to_regclass(format('%I.eom_customer_handoffs', current_schema()));
    handoff_owner TEXT;
BEGIN
    IF runtime_role IS NULL THEN
        RAISE EXCEPTION
            'migration 356 requires atlas.eom_funnel_runtime_role to name the serving login';
    END IF;

    IF runtime_role = 'atlas_eom_handoff_owner' THEN
        RAISE EXCEPTION
            'migration 356 target runtime role must be a serving login, not atlas_eom_handoff_owner';
    END IF;

    SELECT rolcanlogin, rolsuper, rolcreaterole, rolcreatedb, rolbypassrls
    INTO
        runtime_can_login,
        runtime_is_superuser,
        runtime_can_create_role,
        runtime_can_create_db,
        runtime_can_bypass_rls
    FROM pg_roles
    WHERE rolname = runtime_role;
    IF runtime_can_login IS NULL THEN
        RAISE EXCEPTION
            'migration 356 target runtime role % does not exist',
            runtime_role;
    END IF;
    IF NOT runtime_can_login THEN
        RAISE EXCEPTION
            'migration 356 target runtime role % must be a LOGIN role',
            runtime_role;
    END IF;
    IF runtime_is_superuser
        OR runtime_can_create_role
        OR runtime_can_create_db
        OR runtime_can_bypass_rls
    THEN
        RAISE EXCEPTION
            'migration 356 target runtime role % must not be superuser, CREATEROLE, CREATEDB, or BYPASSRLS',
            runtime_role;
    END IF;
    SELECT database_row.datdba = runtime_role_row.oid
    INTO runtime_owns_database
    FROM pg_database AS database_row
    JOIN pg_roles AS runtime_role_row ON runtime_role_row.rolname = runtime_role
    WHERE database_row.datname = current_database();
    IF COALESCE(runtime_owns_database, FALSE) THEN
        RAISE EXCEPTION
            'migration 356 target runtime role % must not own the database',
            runtime_role;
    END IF;

    SELECT rolsuper
    INTO executor_is_superuser
    FROM pg_roles
    WHERE rolname = migration_executor;
    executor_is_superuser := COALESCE(executor_is_superuser, FALSE);

    IF handoff_table IS NULL THEN
        RAISE EXCEPTION
            'migration 356 requires eom_customer_handoffs from migration 353';
    END IF;

    SELECT owner_role.rolname
    INTO handoff_owner
    FROM pg_class AS handoff_class
    JOIN pg_roles AS owner_role ON owner_role.oid = handoff_class.relowner
    WHERE handoff_class.oid = handoff_table;

    IF handoff_owner <> 'atlas_eom_handoff_owner' THEN
        RAISE EXCEPTION
            'migration 356 requires migration 354 to transfer eom_customer_handoffs to atlas_eom_handoff_owner';
    END IF;

    SELECT pg_has_role(migration_executor, 'atlas_eom_handoff_owner', 'MEMBER')
    INTO executor_is_guard_member;
    SELECT COALESCE(admin_option, FALSE)
    INTO executor_has_guard_admin
    FROM pg_auth_members AS membership
    JOIN pg_roles AS member_role ON member_role.oid = membership.member
    JOIN pg_roles AS guard_role ON guard_role.oid = membership.roleid
    WHERE member_role.rolname = migration_executor
      AND guard_role.rolname = 'atlas_eom_handoff_owner';
    executor_has_guard_admin := COALESCE(executor_has_guard_admin, FALSE);

    IF NOT executor_is_superuser AND (
        NOT executor_is_guard_member OR NOT executor_has_guard_admin
    ) THEN
        RAISE EXCEPTION
            'migration executor % must be a superuser or an admin member of atlas_eom_handoff_owner',
            migration_executor;
    END IF;

    EXECUTE 'SET LOCAL ROLE atlas_eom_handoff_owner';
    EXECUTE format(
        'GRANT SELECT, INSERT, UPDATE ON TABLE %I.eom_customer_handoffs TO %I',
        schema_name,
        runtime_role
    );
    EXECUTE 'RESET ROLE';
END;
$$;
