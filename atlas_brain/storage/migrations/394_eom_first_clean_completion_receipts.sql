-- atlas: atomic-bookkeeping
-- Immutable, tenant-scoped first-clean completion evidence for EOM.
--
-- `first_clean_booked` is Calendar booking evidence, not service completion.
-- These rows admit only an authenticated tracker report for an already
-- canonicalized active residential customer. They deliberately create no
-- customer email, token, payment, or Stripe side effect.
--
-- This is a controlled DBA-only migration. The normal Atlas runtime no longer
-- owns eom_customer_handoffs after migration 354, so it cannot safely create
-- a foreign key to that guard-owned table. More importantly, the new immutable
-- evidence must itself be guard-owned; a runtime-owned table could disable its
-- own append-only triggers. Use the dedicated 394 preflight/apply command,
-- never the slim EOM profile's ordinary migration startup.
--
-- Rollback: stop the completion route and its tracker consumer first, then
-- retain these append-only receipts as audit evidence. Do not delete a receipt
-- merely because an application deployment is rolled back.

DO $$
DECLARE
    schema_name TEXT := current_schema();
    executor_is_superuser BOOLEAN;
    guard_is_trusted BOOLEAN;
    runtime_role_ready BOOLEAN;
BEGIN
    SELECT COALESCE(role.rolsuper, FALSE)
      INTO executor_is_superuser
      FROM pg_roles AS role
     WHERE role.rolname = current_user;
    IF NOT executor_is_superuser THEN
        RAISE EXCEPTION
            'database administrator must run 394_eom_first_clean_completion_receipts';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_roles WHERE rolname = 'atlas_eom_handoff_owner'
    ) THEN
        CREATE ROLE atlas_eom_handoff_owner NOLOGIN NOINHERIT;
    END IF;
    ALTER ROLE atlas_eom_handoff_owner
        NOLOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE
        NOREPLICATION NOBYPASSRLS;

    SELECT EXISTS (
        SELECT 1
          FROM pg_roles AS guard_role
         WHERE guard_role.rolname = 'atlas_eom_handoff_owner'
           AND NOT guard_role.rolcanlogin
           AND NOT guard_role.rolinherit
           AND NOT guard_role.rolsuper
           AND NOT guard_role.rolcreaterole
    )
      INTO guard_is_trusted;
    IF NOT guard_is_trusted THEN
        RAISE EXCEPTION
            'atlas_eom_handoff_owner must be a no-login, membership-isolated guard role before running 394_eom_first_clean_completion_receipts';
    END IF;

    SELECT EXISTS (
        SELECT 1
          FROM pg_roles AS runtime_role
         WHERE runtime_role.rolname = 'atlas'
           AND runtime_role.rolcanlogin
    )
      INTO runtime_role_ready;
    IF NOT runtime_role_ready THEN
        RAISE EXCEPTION
            'atlas must be a login runtime role before running 394_eom_first_clean_completion_receipts';
    END IF;

    IF to_regclass(format('%I.contacts', schema_name)) IS NULL
       OR to_regclass(format('%I.eom_lead_lifecycle_events', schema_name)) IS NULL
       OR to_regclass(format('%I.eom_customer_handoffs', schema_name)) IS NULL THEN
        RAISE EXCEPTION
            'contacts, eom_lead_lifecycle_events, and eom_customer_handoffs must exist before running 394_eom_first_clean_completion_receipts';
    END IF;

    -- A no-login guard role can own protected objects only when it can create
    -- them in this schema. Runtime and NocoDB must never retain a path to set
    -- that role after the migration commits.
    EXECUTE format(
        'GRANT USAGE, CREATE ON SCHEMA %I TO atlas_eom_handoff_owner',
        schema_name
    );
    EXECUTE format('GRANT USAGE ON SCHEMA %I TO atlas', schema_name);
    EXECUTE format('REVOKE %I FROM %I', 'atlas_eom_handoff_owner', 'atlas');
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'atlas_nocodb') THEN
        EXECUTE format(
            'REVOKE %I FROM %I',
            'atlas_eom_handoff_owner',
            'atlas_nocodb'
        );
    END IF;
    IF EXISTS (
        WITH RECURSIVE role_chain(roleid) AS (
            SELECT membership.roleid
              FROM pg_auth_members AS membership
             WHERE membership.member = (
                 SELECT oid FROM pg_roles WHERE rolname = 'atlas'
             )
            UNION
            SELECT membership.roleid
              FROM pg_auth_members AS membership
              JOIN role_chain ON membership.member = role_chain.roleid
        )
        SELECT 1
          FROM role_chain
         WHERE roleid = (
             SELECT oid
               FROM pg_roles
              WHERE rolname = 'atlas_eom_handoff_owner'
         )
    ) THEN
        RAISE EXCEPTION
            'atlas must not retain direct or inherited membership in atlas_eom_handoff_owner';
    END IF;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'atlas_nocodb')
       AND EXISTS (
           WITH RECURSIVE role_chain(roleid) AS (
               SELECT membership.roleid
                 FROM pg_auth_members AS membership
                WHERE membership.member = (
                    SELECT oid FROM pg_roles WHERE rolname = 'atlas_nocodb'
                )
               UNION
               SELECT membership.roleid
                 FROM pg_auth_members AS membership
                 JOIN role_chain ON membership.member = role_chain.roleid
           )
           SELECT 1
             FROM role_chain
            WHERE roleid = (
                SELECT oid
                  FROM pg_roles
                 WHERE rolname = 'atlas_eom_handoff_owner'
            )
       ) THEN
        RAISE EXCEPTION
            'atlas_nocodb must not retain direct or inherited membership in atlas_eom_handoff_owner';
    END IF;
END;
$$;

CREATE TABLE IF NOT EXISTS eom_first_clean_completion_operation_receipts (
    operation_key VARCHAR(128) PRIMARY KEY,
    contact_id UUID NOT NULL REFERENCES contacts(id) ON DELETE RESTRICT,
    business_context_id VARCHAR(64) NOT NULL DEFAULT 'effingham_maids',
    operation_kind VARCHAR(32) NOT NULL DEFAULT 'first_clean_completion',
    request_fingerprint VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT ck_eom_first_clean_completion_operation_context
        CHECK (business_context_id = 'effingham_maids'),
    CONSTRAINT ck_eom_first_clean_completion_operation_key
        CHECK (length(btrim(operation_key)) BETWEEN 16 AND 128),
    CONSTRAINT ck_eom_first_clean_completion_operation_kind
        CHECK (operation_kind = 'first_clean_completion'),
    CONSTRAINT ck_eom_first_clean_completion_operation_fingerprint
        CHECK (request_fingerprint ~ '^[0-9a-f]{64}$')
);

CREATE TABLE IF NOT EXISTS eom_first_clean_completion_receipts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    contact_id UUID NOT NULL REFERENCES contacts(id) ON DELETE RESTRICT,
    handoff_id UUID NOT NULL REFERENCES eom_customer_handoffs(id)
        ON DELETE RESTRICT,
    tracker_customer_id BIGINT NOT NULL CHECK (tracker_customer_id > 0),
    tracker_site_id BIGINT NOT NULL CHECK (tracker_site_id > 0),
    tracker_service_kind VARCHAR(32) NOT NULL,
    tracker_service_id BIGINT NOT NULL CHECK (tracker_service_id > 0),
    completed_at TIMESTAMPTZ NOT NULL,
    operation_key VARCHAR(128) NOT NULL
        REFERENCES eom_first_clean_completion_operation_receipts(operation_key)
        ON DELETE RESTRICT,
    request_fingerprint VARCHAR(64) NOT NULL,
    actor_id BIGINT NOT NULL CHECK (actor_id > 0),
    actor_name VARCHAR(128) NOT NULL,
    source VARCHAR(32) NOT NULL DEFAULT 'time_tracker',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT ck_eom_first_clean_completion_service_kind
        CHECK (tracker_service_kind IN ('job', 'planned_visit')),
    CONSTRAINT ck_eom_first_clean_completion_actor_name
        CHECK (length(btrim(actor_name)) > 0),
    CONSTRAINT ck_eom_first_clean_completion_source
        CHECK (source = 'time_tracker'),
    CONSTRAINT ck_eom_first_clean_completion_fingerprint
        CHECK (request_fingerprint ~ '^[0-9a-f]{64}$'),
    CONSTRAINT uq_eom_first_clean_completion_receipt_contact
        UNIQUE (contact_id),
    CONSTRAINT uq_eom_first_clean_completion_receipt_handoff
        UNIQUE (handoff_id),
    CONSTRAINT uq_eom_first_clean_completion_receipt_operation
        UNIQUE (operation_key),
    CONSTRAINT uq_eom_first_clean_completion_tracker_service
        UNIQUE (tracker_service_kind, tracker_service_id)
);

CREATE INDEX IF NOT EXISTS idx_eom_first_clean_completion_receipts_completed
    ON eom_first_clean_completion_receipts (completed_at DESC, id DESC);

CREATE OR REPLACE FUNCTION prevent_eom_first_clean_completion_mutation()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'eom first-clean completion evidence is append-only';
END;
$$;

DROP TRIGGER IF EXISTS trg_prevent_eom_first_clean_completion_operation_mutation
    ON eom_first_clean_completion_operation_receipts;
CREATE TRIGGER trg_prevent_eom_first_clean_completion_operation_mutation
    BEFORE UPDATE OR DELETE ON eom_first_clean_completion_operation_receipts
    FOR EACH ROW
    EXECUTE FUNCTION prevent_eom_first_clean_completion_mutation();

DROP TRIGGER IF EXISTS trg_prevent_eom_first_clean_completion_operation_truncate
    ON eom_first_clean_completion_operation_receipts;
CREATE TRIGGER trg_prevent_eom_first_clean_completion_operation_truncate
    BEFORE TRUNCATE ON eom_first_clean_completion_operation_receipts
    FOR EACH STATEMENT
    EXECUTE FUNCTION prevent_eom_first_clean_completion_mutation();

DROP TRIGGER IF EXISTS trg_prevent_eom_first_clean_completion_receipt_mutation
    ON eom_first_clean_completion_receipts;
CREATE TRIGGER trg_prevent_eom_first_clean_completion_receipt_mutation
    BEFORE UPDATE OR DELETE ON eom_first_clean_completion_receipts
    FOR EACH ROW
    EXECUTE FUNCTION prevent_eom_first_clean_completion_mutation();

DROP TRIGGER IF EXISTS trg_prevent_eom_first_clean_completion_receipt_truncate
    ON eom_first_clean_completion_receipts;
CREATE TRIGGER trg_prevent_eom_first_clean_completion_receipt_truncate
    BEFORE TRUNCATE ON eom_first_clean_completion_receipts
    FOR EACH STATEMENT
    EXECUTE FUNCTION prevent_eom_first_clean_completion_mutation();

CREATE OR REPLACE FUNCTION require_eom_first_clean_completion_operation_scope()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
          FROM contacts AS contact
         WHERE contact.id = NEW.contact_id
           AND contact.business_context_id = 'effingham_maids'
    ) THEN
        RAISE EXCEPTION
            'eom first-clean completion operation requires an EOM contact';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_require_eom_first_clean_completion_operation_scope
    ON eom_first_clean_completion_operation_receipts;
CREATE TRIGGER trg_require_eom_first_clean_completion_operation_scope
    BEFORE INSERT OR UPDATE ON eom_first_clean_completion_operation_receipts
    FOR EACH ROW
    EXECUTE FUNCTION require_eom_first_clean_completion_operation_scope();

CREATE OR REPLACE FUNCTION require_eom_first_clean_completion_receipt()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.completed_at > CURRENT_TIMESTAMP THEN
        RAISE EXCEPTION
            'eom first-clean completion cannot be recorded in the future';
    END IF;

    IF NOT EXISTS (
        SELECT 1
          FROM contacts AS contact
          JOIN eom_customer_handoffs AS handoff
            ON handoff.id = NEW.handoff_id
           AND handoff.contact_id = NEW.contact_id
           AND handoff.tracker_customer_id = NEW.tracker_customer_id
           AND handoff.tracker_site_id = NEW.tracker_site_id
          JOIN eom_first_clean_completion_operation_receipts AS operation
            ON operation.operation_key = NEW.operation_key
           AND operation.contact_id = NEW.contact_id
           AND operation.request_fingerprint = NEW.request_fingerprint
         WHERE contact.id = NEW.contact_id
           AND contact.business_context_id = 'effingham_maids'
           AND contact.contact_type = 'customer'
           AND contact.status = 'active'
           AND contact.customer_type = 'residential'
    ) THEN
        RAISE EXCEPTION
            'eom first-clean completion requires matching active residential customer handoff';
    END IF;

    IF NOT EXISTS (
        SELECT 1
          FROM eom_lead_lifecycle_events AS lifecycle
         WHERE lifecycle.contact_id = NEW.contact_id
           AND lifecycle.event_type = 'first_clean_completed'
           AND lifecycle.source = 'time_tracker'
           AND lifecycle.operation_key = NEW.operation_key
           AND lifecycle.actor = (
               'employee:' || NEW.actor_id::text || ':' || NEW.actor_name
           )
           AND lifecycle.occurred_at = NEW.completed_at
           AND lifecycle.metadata @> jsonb_build_object(
               'completion_receipt_id', NEW.id::text,
               'handoff_id', NEW.handoff_id::text,
               'tracker_customer_id', NEW.tracker_customer_id,
               'tracker_site_id', NEW.tracker_site_id,
               'tracker_service_kind', NEW.tracker_service_kind,
               'tracker_service_id', NEW.tracker_service_id
           )
    ) THEN
        RAISE EXCEPTION
            'eom first-clean completion requires matching lifecycle evidence';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_require_eom_first_clean_completion_receipt
    ON eom_first_clean_completion_receipts;
CREATE TRIGGER trg_require_eom_first_clean_completion_receipt
    BEFORE INSERT OR UPDATE ON eom_first_clean_completion_receipts
    FOR EACH ROW
    EXECUTE FUNCTION require_eom_first_clean_completion_receipt();

DO $$
DECLARE
    schema_name TEXT := current_schema();
    table_name TEXT;
    grantee_name TEXT;
BEGIN
    ALTER TABLE eom_first_clean_completion_operation_receipts
        OWNER TO atlas_eom_handoff_owner;
    ALTER TABLE eom_first_clean_completion_receipts
        OWNER TO atlas_eom_handoff_owner;
    ALTER FUNCTION prevent_eom_first_clean_completion_mutation()
        OWNER TO atlas_eom_handoff_owner;
    ALTER FUNCTION require_eom_first_clean_completion_operation_scope()
        OWNER TO atlas_eom_handoff_owner;
    ALTER FUNCTION require_eom_first_clean_completion_receipt()
        OWNER TO atlas_eom_handoff_owner;

    -- Clear any inherited/default ACLs before granting only the row-locking
    -- DML the service actually uses. In particular, do not grant DELETE,
    -- TRUNCATE, REFERENCES, TRIGGER, or ownership to the runtime.
    FOR table_name IN
        SELECT unnest(ARRAY[
            'eom_first_clean_completion_operation_receipts',
            'eom_first_clean_completion_receipts'
        ])
    LOOP
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
        LOOP
            EXECUTE format(
                'REVOKE ALL PRIVILEGES ON TABLE %I.%I FROM %I',
                schema_name,
                table_name,
                grantee_name
            );
        END LOOP;
    END LOOP;
    EXECUTE format(
        'GRANT SELECT, INSERT, UPDATE ON TABLE %I.eom_first_clean_completion_operation_receipts TO atlas',
        schema_name
    );
    EXECUTE format(
        'GRANT SELECT, INSERT, UPDATE ON TABLE %I.eom_first_clean_completion_receipts TO atlas',
        schema_name
    );
END;
$$;

COMMENT ON TABLE eom_first_clean_completion_operation_receipts IS
    'Globally unique operation-key ownership for EOM first-clean completion reports; retries cannot move a completion to another customer or service.';
COMMENT ON TABLE eom_first_clean_completion_receipts IS
    'Immutable EOM evidence that one active residential canonical customer completed a first service; it is not a booking, email, payment, or Stripe authorization.';
