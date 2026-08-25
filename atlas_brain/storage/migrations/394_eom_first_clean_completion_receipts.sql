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
-- a foreign key to that guard-owned table. More importantly, the receipt and
-- lifecycle evidence boundaries must be guard-owned; a runtime-owned table or
-- trigger function could disable or replace its append-only implementation.
-- Use the dedicated 394 preflight/apply command, never the slim EOM profile's
-- ordinary migration startup.
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

    -- Refuse an elevated runtime before this migration creates or changes any
    -- protected role/object. A later guard-owner ACL cannot constrain a login
    -- that can bypass or administer PostgreSQL privileges itself.
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
    )
      INTO runtime_role_ready;
    IF NOT runtime_role_ready THEN
        RAISE EXCEPTION
            'atlas must be an unprivileged login runtime role before running 394_eom_first_clean_completion_receipts';
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

    -- A no-login owner role is not an effective guard if any ordinary login
    -- can assume it, directly or through another role. Superusers already
    -- bypass this boundary, so match the repository's canonical guard
    -- predicate and reject every non-superuser login member before any receipt
    -- object is created or transferred.
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
            'no non-superuser login may retain direct or inherited membership in atlas_eom_handoff_owner';
    END IF;

    IF to_regclass(format('%I.contacts', schema_name)) IS NULL
       OR to_regclass(format('%I.eom_lead_lifecycle_events', schema_name)) IS NULL
       OR to_regclass(format(
           '%I.eom_lead_lifecycle_events_sequence_seq', schema_name
       )) IS NULL
       OR to_regclass(format('%I.eom_customer_handoffs', schema_name)) IS NULL THEN
        RAISE EXCEPTION
            'contacts, eom_lead_lifecycle_events, its ordering sequence, and eom_customer_handoffs must exist before running 394_eom_first_clean_completion_receipts';
    END IF;

    IF to_regclass(pg_get_serial_sequence(
        format('%I.%I', schema_name, 'eom_lead_lifecycle_events'),
        'lifecycle_sequence'
    )) IS DISTINCT FROM to_regclass(format(
        '%I.eom_lead_lifecycle_events_sequence_seq', schema_name
    )) THEN
        RAISE EXCEPTION
            'eom_lead_lifecycle_events.lifecycle_sequence must use its canonical ordering sequence before running 394_eom_first_clean_completion_receipts';
    END IF;

    IF NOT EXISTS (
        SELECT 1
          FROM pg_attrdef AS attribute_default
          JOIN pg_attribute AS attribute
            ON attribute.attrelid = attribute_default.adrelid
           AND attribute.attnum = attribute_default.adnum
          JOIN pg_depend AS dependency
            ON dependency.classid = 'pg_attrdef'::regclass
           AND dependency.objid = attribute_default.oid
           AND dependency.refclassid = 'pg_class'::regclass
          JOIN pg_class AS sequence ON sequence.oid = dependency.refobjid
          JOIN pg_namespace AS sequence_namespace
            ON sequence_namespace.oid = sequence.relnamespace
         WHERE attribute_default.adrelid = to_regclass(
                   format('%I.%I', schema_name, 'eom_lead_lifecycle_events')
               )
           AND attribute.attname = 'lifecycle_sequence'
           AND sequence_namespace.nspname = schema_name
           AND sequence.relkind = 'S'
           AND sequence.relname = 'eom_lead_lifecycle_events_sequence_seq'
    ) THEN
        RAISE EXCEPTION
            'eom_lead_lifecycle_events.lifecycle_sequence must retain its canonical nextval default before running 394_eom_first_clean_completion_receipts';
    END IF;

    -- Migration 353 creates the handoff relation, but migration 354 moves its
    -- mutable/append-only boundary out of the runtime role. Completion
    -- evidence depends on that immutable canonical bridge, so a relation or
    -- trigger function still owned by Atlas is not an admissible prerequisite.
    IF NOT EXISTS (
        SELECT 1
          FROM pg_class AS relation
          JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
         WHERE namespace.nspname = schema_name
           AND relation.relkind = 'r'
           AND relation.relname = 'eom_customer_handoffs'
           AND relation.relowner = (
               SELECT oid
                 FROM pg_roles
                WHERE rolname = 'atlas_eom_handoff_owner'
           )
    ) OR (
        SELECT COUNT(*)
          FROM pg_proc AS protected_function
          JOIN pg_namespace AS namespace
            ON namespace.oid = protected_function.pronamespace
         WHERE namespace.nspname = schema_name
           AND protected_function.proname IN (
               'require_eom_customer_handoff_finalization',
               'prevent_eom_customer_handoff_mutation'
           )
           AND protected_function.pronargs = 0
           AND protected_function.proowner = (
               SELECT oid
                 FROM pg_roles
                WHERE rolname = 'atlas_eom_handoff_owner'
           )
    ) <> 2 THEN
        RAISE EXCEPTION
            'eom_customer_handoffs and its protected functions must be guard-owned by atlas_eom_handoff_owner; apply 354_eom_customer_handoff_privileges before 394_eom_first_clean_completion_receipts';
    END IF;

    -- A no-login guard role can own protected objects only when it can create
    -- them in this schema. No non-superuser login may retain a path to set that
    -- role after the migration commits.
    EXECUTE format(
        'GRANT USAGE, CREATE ON SCHEMA %I TO atlas_eom_handoff_owner',
        schema_name
    );
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
BEGIN
    -- PostgreSQL prepends a caller's temporary schema unless pg_temp is named
    -- explicitly. These invoker trigger functions validate permanent evidence,
    -- so pin the target schema first and pg_temp last rather than trusting a
    -- runtime connection's mutable search_path.
    EXECUTE format(
        'ALTER FUNCTION %I.require_eom_first_clean_completion_operation_scope() '
        || 'SET search_path TO %I, pg_catalog, pg_temp',
        schema_name,
        schema_name
    );
    EXECUTE format(
        'ALTER FUNCTION %I.require_eom_first_clean_completion_receipt() '
        || 'SET search_path TO %I, pg_catalog, pg_temp',
        schema_name,
        schema_name
    );
END;
$$;

DO $$
DECLARE
    schema_name TEXT := current_schema();
    table_name TEXT;
    sequence_name TEXT := 'eom_lead_lifecycle_events_sequence_seq';
    grantee_name TEXT;
BEGIN
    -- Guarding individual relations is insufficient while the runtime owns
    -- their containing namespace: a schema owner can DROP SCHEMA ... CASCADE
    -- around every table/function ACL. Transfer the namespace after all
    -- prerequisite objects have been checked, then re-grant only the runtime
    -- schema privileges ordinary future migrations need. CREATE cannot alter
    -- an existing guard-owned object, and the admission functions use their
    -- fixed guarded search_path rather than caller-created objects.
    EXECUTE format(
        'ALTER SCHEMA %I OWNER TO atlas_eom_handoff_owner',
        schema_name
    );
    EXECUTE format('GRANT USAGE, CREATE ON SCHEMA %I TO atlas', schema_name);

    ALTER TABLE eom_first_clean_completion_operation_receipts
        OWNER TO atlas_eom_handoff_owner;
    ALTER TABLE eom_first_clean_completion_receipts
        OWNER TO atlas_eom_handoff_owner;
    ALTER TABLE eom_lead_lifecycle_events
        OWNER TO atlas_eom_handoff_owner;
    -- Migration 363 gives lifecycle_sequence a nextval() default.  The table
    -- transfer above does not grant its runtime writers the sequence access
    -- that default consumes, so bind the canonical sequence to the same guard
    -- and restore only nextval()/USAGE for the direct Atlas runtime.
    EXECUTE format(
        'ALTER SEQUENCE %I.%I OWNER TO atlas_eom_handoff_owner',
        schema_name,
        sequence_name
    );
    ALTER FUNCTION prevent_eom_lead_lifecycle_event_mutation()
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
               -- Foreign-key enforcement runs under the referencing table's
               -- owner. Do not revoke the new guard owner's implicit access
               -- to the operation receipt it must key-share lock.
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
    EXECUTE format(
        'REVOKE ALL PRIVILEGES ON SEQUENCE %I.%I FROM PUBLIC',
        schema_name,
        sequence_name
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
           AND relation.relkind = 'S'
           AND relation.relname = sequence_name
           AND grantee_role.rolname <> 'atlas_eom_handoff_owner'
    LOOP
        EXECUTE format(
            'REVOKE ALL PRIVILEGES ON SEQUENCE %I.%I FROM %I',
            schema_name,
            sequence_name,
            grantee_name
        );
    END LOOP;
    EXECUTE format(
        'GRANT SELECT, INSERT, UPDATE ON TABLE %I.eom_first_clean_completion_operation_receipts TO atlas',
        schema_name
    );
    EXECUTE format(
        'GRANT SELECT, INSERT, UPDATE ON TABLE %I.eom_first_clean_completion_receipts TO atlas',
        schema_name
    );
    -- Existing lifecycle writers append evidence. They require UPDATE only to
    -- take row locks on idempotency/booking reads; the guard-owned append-only
    -- trigger still rejects every actual UPDATE or DELETE.
    EXECUTE format(
        'GRANT SELECT, INSERT, UPDATE ON TABLE %I.eom_lead_lifecycle_events TO atlas',
        schema_name
    );
    EXECUTE format(
        'GRANT USAGE ON SEQUENCE %I.%I TO atlas',
        schema_name,
        sequence_name
    );
END;
$$;

COMMENT ON TABLE eom_first_clean_completion_operation_receipts IS
    'Globally unique operation-key ownership for EOM first-clean completion reports; retries cannot move a completion to another customer or service.';
COMMENT ON TABLE eom_first_clean_completion_receipts IS
    'Immutable EOM evidence that one active residential canonical customer completed a first service; it is not a booking, email, payment, or Stripe authorization.';
