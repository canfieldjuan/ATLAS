-- atlas: atomic-bookkeeping
-- Immutable, bilingual EOM Terms releases. This migration stores no customer
-- acceptance and seeds no Terms content; later invitation/acceptance slices
-- reference the published version selected by the singleton pointer.
--
-- This is a controlled DBA-only migration. The direct Atlas runtime must be
-- able to create drafts and publish them, but it must not own the relations or
-- trigger functions that make published history immutable. Apply this through
-- the dedicated Terms-authority preflight/apply command, never ordinary Atlas
-- startup migrations.
--
-- Post-deployment rollback: stop the Terms write routes and every consumer
-- before rolling the application back. Retain the guard-owned tables,
-- functions, triggers, grants, and migration record as legal/audit history;
-- do not drop, truncate, delete, or unrecord them. The complete operational
-- sequence and security-containment exception live in
-- .agent/runbooks/database.md under "EOM Terms authority migration".

DO $$
DECLARE
    schema_name TEXT := current_schema();
    executor_is_superuser BOOLEAN;
    runtime_role_ready BOOLEAN;
    guard_role_ready BOOLEAN;
BEGIN
    SELECT COALESCE(role.rolsuper, FALSE)
      INTO executor_is_superuser
      FROM pg_roles AS role
     WHERE role.rolname = current_user;
    IF NOT executor_is_superuser THEN
        RAISE EXCEPTION
            'database administrator must run 396_eom_terms_authority';
    END IF;

    SELECT EXISTS (
        SELECT 1
          FROM pg_roles AS runtime_role
         WHERE runtime_role.rolname = 'atlas'
           AND runtime_role.rolcanlogin
           AND NOT runtime_role.rolsuper
           AND NOT runtime_role.rolcreaterole
           AND NOT runtime_role.rolcreatedb
           AND NOT runtime_role.rolreplication
           AND NOT runtime_role.rolbypassrls
           AND NOT EXISTS (
               SELECT 1
                 FROM pg_database AS database
                WHERE database.datname = current_database()
                  AND database.datdba = runtime_role.oid
           )
           AND NOT EXISTS (
               SELECT 1
                 FROM pg_roles AS elevated_role
                WHERE elevated_role.oid <> runtime_role.oid
                  AND (
                      elevated_role.rolsuper
                      OR elevated_role.rolcreaterole
                      OR elevated_role.rolcreatedb
                      OR elevated_role.rolreplication
                      OR elevated_role.rolbypassrls
                  )
                  AND pg_has_role(
                      runtime_role.oid, elevated_role.oid, 'MEMBER'
                  )
           )
    )
      INTO runtime_role_ready;
    IF NOT runtime_role_ready THEN
        RAISE EXCEPTION
            'atlas must be an unprivileged login runtime role before running 396_eom_terms_authority';
    END IF;

    SELECT EXISTS (
        SELECT 1
          FROM pg_roles AS guard_role
         WHERE guard_role.rolname = 'atlas_eom_handoff_owner'
           AND NOT guard_role.rolcanlogin
           AND NOT guard_role.rolinherit
           AND NOT guard_role.rolsuper
           AND NOT guard_role.rolcreaterole
           AND NOT guard_role.rolcreatedb
           AND NOT guard_role.rolreplication
           AND NOT guard_role.rolbypassrls
    )
      INTO guard_role_ready;
    IF NOT guard_role_ready THEN
        RAISE EXCEPTION
            'atlas_eom_handoff_owner must already be the isolated no-login EOM guard role before running 396_eom_terms_authority';
    END IF;

    IF EXISTS (
        SELECT 1
          FROM pg_stat_activity AS activity
          JOIN pg_roles AS guard_role
            ON guard_role.rolname = 'atlas_eom_handoff_owner'
         WHERE activity.usesysid = guard_role.oid
           AND activity.datid = (
               SELECT database.oid
                 FROM pg_database AS database
                WHERE database.datname = current_database()
           )
    ) THEN
        RAISE EXCEPTION
            'atlas_eom_handoff_owner must have no live sessions before running 396_eom_terms_authority';
    END IF;

    IF EXISTS (
        SELECT 1
          FROM pg_roles AS member_role
          JOIN pg_roles AS guard_role
            ON guard_role.rolname = 'atlas_eom_handoff_owner'
         WHERE member_role.rolcanlogin
           AND NOT member_role.rolsuper
           AND pg_has_role(member_role.oid, guard_role.oid, 'MEMBER')
    ) THEN
        RAISE EXCEPTION
            'no non-superuser login may retain membership in atlas_eom_handoff_owner';
    END IF;

    IF NOT EXISTS (
        SELECT 1
          FROM pg_namespace AS namespace
         WHERE namespace.nspname = schema_name
           AND namespace.nspowner = (
               SELECT oid
                 FROM pg_roles
                WHERE rolname = 'atlas_eom_handoff_owner'
           )
    ) THEN
        RAISE EXCEPTION
            'the EOM funnel schema must already be owned by atlas_eom_handoff_owner; apply the guarded EOM schema boundary before 396_eom_terms_authority';
    END IF;

    IF to_regclass(format('%I.eom_terms_versions', schema_name)) IS NOT NULL
       OR to_regclass(format('%I.eom_terms_current_version', schema_name)) IS NOT NULL
       OR to_regprocedure(format('%I.protect_eom_terms_version()', schema_name))
            IS NOT NULL
       OR to_regprocedure(format(
            '%I.require_published_eom_terms_current_version()', schema_name
          )) IS NOT NULL
       OR to_regprocedure(format(
            '%I.prevent_eom_terms_current_removal()', schema_name
          )) IS NOT NULL THEN
        RAISE EXCEPTION
            'refusing to adopt a pre-existing EOM Terms authority object';
    END IF;
END;
$$;

CREATE TABLE eom_terms_versions (
    id UUID CONSTRAINT pk_eom_terms_versions PRIMARY KEY DEFAULT gen_random_uuid(),
    business_context_id VARCHAR(64) NOT NULL DEFAULT 'effingham_maids',
    version_label VARCHAR(64) NOT NULL,
    status VARCHAR(16) NOT NULL DEFAULT 'draft',
    material_change BOOLEAN NOT NULL,
    documents JSONB NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    created_by_id BIGINT NOT NULL,
    created_by_name VARCHAR(128) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    published_by_id BIGINT,
    published_by_name VARCHAR(128),
    published_at TIMESTAMPTZ,
    CONSTRAINT uq_eom_terms_version_label UNIQUE (version_label),
    CONSTRAINT ck_eom_terms_context
        CHECK (business_context_id = 'effingham_maids'),
    CONSTRAINT ck_eom_terms_version_label
        CHECK (version_label ~ '^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$'),
    CONSTRAINT ck_eom_terms_status
        CHECK (status IN ('draft', 'published')),
    CONSTRAINT ck_eom_terms_documents_object
        CHECK (jsonb_typeof(documents) = 'object'),
    CONSTRAINT ck_eom_terms_content_hash
        CHECK (content_hash ~ '^[0-9a-f]{64}$'),
    CONSTRAINT ck_eom_terms_creator
        CHECK (created_by_id > 0 AND length(btrim(created_by_name)) > 0),
    CONSTRAINT ck_eom_terms_publication
        CHECK (
            (status = 'draft'
                AND published_by_id IS NULL
                AND published_by_name IS NULL
                AND published_at IS NULL)
            OR
            (status = 'published'
                AND published_by_id > 0
                AND length(btrim(published_by_name)) > 0
                AND published_at IS NOT NULL)
        )
);

CREATE TABLE eom_terms_current_version (
    singleton BOOLEAN NOT NULL DEFAULT TRUE,
    version_id UUID NOT NULL,
    selected_by_id BIGINT NOT NULL,
    selected_by_name VARCHAR(128) NOT NULL,
    selected_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT pk_eom_terms_current_version PRIMARY KEY (singleton),
    CONSTRAINT ck_eom_terms_current_singleton CHECK (singleton),
    CONSTRAINT uq_eom_terms_current_version UNIQUE (version_id),
    CONSTRAINT fk_eom_terms_current_version
        FOREIGN KEY (version_id)
        REFERENCES eom_terms_versions(id) ON DELETE RESTRICT,
    CONSTRAINT ck_eom_terms_current_selector_id CHECK (selected_by_id > 0),
    CONSTRAINT ck_eom_terms_current_selector_name
        CHECK (length(btrim(selected_by_name)) > 0)
);

CREATE FUNCTION protect_eom_terms_version()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY INVOKER
AS $$
BEGIN
    IF TG_OP = 'TRUNCATE' THEN
        RAISE EXCEPTION 'EOM Terms version history is append-only';
    END IF;
    IF OLD.status = 'published' THEN
        RAISE EXCEPTION 'Published EOM Terms versions are immutable';
    END IF;
    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    END IF;
    IF NEW.status = 'draft' THEN
        RAISE EXCEPTION 'EOM Terms drafts cannot be edited; create a new version';
    END IF;
    IF NEW.id IS DISTINCT FROM OLD.id
       OR NEW.business_context_id IS DISTINCT FROM OLD.business_context_id
       OR NEW.version_label IS DISTINCT FROM OLD.version_label
       OR NEW.material_change IS DISTINCT FROM OLD.material_change
       OR NEW.documents IS DISTINCT FROM OLD.documents
       OR NEW.content_hash IS DISTINCT FROM OLD.content_hash
       OR NEW.created_by_id IS DISTINCT FROM OLD.created_by_id
       OR NEW.created_by_name IS DISTINCT FROM OLD.created_by_name
       OR NEW.created_at IS DISTINCT FROM OLD.created_at THEN
        RAISE EXCEPTION 'Publishing EOM Terms cannot rewrite draft content';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_protect_eom_terms_version
    BEFORE UPDATE OR DELETE ON eom_terms_versions
    FOR EACH ROW EXECUTE FUNCTION protect_eom_terms_version();

CREATE TRIGGER trg_protect_eom_terms_version_truncate
    BEFORE TRUNCATE ON eom_terms_versions
    FOR EACH STATEMENT EXECUTE FUNCTION protect_eom_terms_version();

CREATE FUNCTION require_published_eom_terms_current_version()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY INVOKER
AS $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM eom_terms_versions AS version
        WHERE version.id = NEW.version_id
          AND version.status = 'published'
    ) THEN
        RAISE EXCEPTION 'Current EOM Terms version must be published';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_require_published_eom_terms_current_version
    BEFORE INSERT OR UPDATE ON eom_terms_current_version
    FOR EACH ROW EXECUTE FUNCTION require_published_eom_terms_current_version();

CREATE FUNCTION prevent_eom_terms_current_removal()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY INVOKER
AS $$
BEGIN
    RAISE EXCEPTION 'Current EOM Terms authority cannot be removed';
END;
$$;

CREATE TRIGGER trg_prevent_eom_terms_current_truncate
    BEFORE TRUNCATE ON eom_terms_current_version
    FOR EACH STATEMENT EXECUTE FUNCTION prevent_eom_terms_current_removal();

CREATE TRIGGER trg_prevent_eom_terms_current_delete
    BEFORE DELETE ON eom_terms_current_version
    FOR EACH ROW EXECUTE FUNCTION prevent_eom_terms_current_removal();

CREATE INDEX idx_eom_terms_versions_created
    ON eom_terms_versions (created_at DESC, id DESC);

DO $$
DECLARE
    schema_name TEXT := current_schema();
    table_name TEXT;
    function_name TEXT;
    grantee_name TEXT;
BEGIN
    FOREACH function_name IN ARRAY ARRAY[
        'protect_eom_terms_version',
        'require_published_eom_terms_current_version',
        'prevent_eom_terms_current_removal'
    ]
    LOOP
        EXECUTE format(
            'ALTER FUNCTION %I.%I() RESET ALL', schema_name, function_name
        );
        EXECUTE format(
            'ALTER FUNCTION %I.%I() SET search_path TO pg_catalog, %I, pg_temp',
            schema_name,
            function_name,
            schema_name
        );
        EXECUTE format(
            'ALTER FUNCTION %I.%I() OWNER TO atlas_eom_handoff_owner',
            schema_name,
            function_name
        );
    END LOOP;

    FOREACH table_name IN ARRAY ARRAY[
        'eom_terms_versions',
        'eom_terms_current_version'
    ]
    LOOP
        EXECUTE format(
            'ALTER TABLE %I.%I OWNER TO atlas_eom_handoff_owner',
            schema_name,
            table_name
        );
        EXECUTE format(
            'REVOKE ALL PRIVILEGES ON TABLE %I.%I FROM PUBLIC',
            schema_name,
            table_name
        );
        FOR grantee_name IN
            SELECT DISTINCT grantee_role.rolname
              FROM pg_class AS relation
              JOIN pg_namespace AS namespace
                ON namespace.oid = relation.relnamespace
              CROSS JOIN LATERAL aclexplode(
                  COALESCE(relation.relacl, ARRAY[]::aclitem[])
              ) AS acl
              JOIN pg_roles AS grantee_role ON grantee_role.oid = acl.grantee
             WHERE namespace.nspname = schema_name
               AND relation.relname = table_name
               AND grantee_role.rolname <> 'atlas_eom_handoff_owner'
        LOOP
            EXECUTE format(
                'REVOKE ALL PRIVILEGES ON TABLE %I.%I FROM %I',
                schema_name,
                table_name,
                grantee_name
            );
        END LOOP;
    END LOOP;

    FOREACH function_name IN ARRAY ARRAY[
        'protect_eom_terms_version',
        'require_published_eom_terms_current_version',
        'prevent_eom_terms_current_removal'
    ]
    LOOP
        EXECUTE format(
            'REVOKE ALL PRIVILEGES ON FUNCTION %I.%I() FROM PUBLIC',
            schema_name,
            function_name
        );
        FOR grantee_name IN
            SELECT DISTINCT grantee_role.rolname
              FROM pg_proc AS protected_function
              JOIN pg_namespace AS namespace
                ON namespace.oid = protected_function.pronamespace
              CROSS JOIN LATERAL aclexplode(
                  COALESCE(protected_function.proacl, ARRAY[]::aclitem[])
              ) AS acl
              JOIN pg_roles AS grantee_role ON grantee_role.oid = acl.grantee
             WHERE namespace.nspname = schema_name
               AND protected_function.proname = function_name
               AND protected_function.pronargs = 0
               AND grantee_role.rolname <> 'atlas_eom_handoff_owner'
        LOOP
            EXECUTE format(
                'REVOKE ALL PRIVILEGES ON FUNCTION %I.%I() FROM %I',
                schema_name,
                function_name,
                grantee_name
            );
        END LOOP;
    END LOOP;

    EXECUTE format(
        'GRANT SELECT ON TABLE %I.eom_terms_versions TO atlas',
        schema_name
    );
    EXECUTE format(
        'GRANT INSERT (id, version_label, material_change, documents, '
        || 'content_hash, created_by_id, created_by_name) '
        || 'ON TABLE %I.eom_terms_versions TO atlas',
        schema_name
    );
    EXECUTE format(
        'GRANT UPDATE (status, published_by_id, published_by_name, published_at) '
        || 'ON TABLE %I.eom_terms_versions TO atlas',
        schema_name
    );
    EXECUTE format(
        'GRANT SELECT ON TABLE %I.eom_terms_current_version TO atlas',
        schema_name
    );
    EXECUTE format(
        'GRANT INSERT (singleton, version_id, selected_by_id, selected_by_name) '
        || 'ON TABLE %I.eom_terms_current_version TO atlas',
        schema_name
    );
    EXECUTE format(
        'GRANT UPDATE (version_id, selected_by_id, selected_by_name, selected_at) '
        || 'ON TABLE %I.eom_terms_current_version TO atlas',
        schema_name
    );
END;
$$;
