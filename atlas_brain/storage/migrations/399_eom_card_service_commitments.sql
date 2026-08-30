-- atlas: atomic-bookkeeping
-- Immutable operator classification for residential post-clean card policy.
-- A one-time customer still accepts Terms, but can never enter card setup.
--
-- Controlled DBA-only migration. Rollback is retention-only: stop consumers and
-- retain the decision evidence. Never delete or rewrite a recorded decision.

DO $$
DECLARE
    schema_name TEXT := current_schema();
    executor_is_superuser BOOLEAN;
BEGIN
    SELECT COALESCE(role.rolsuper, FALSE)
      INTO executor_is_superuser
      FROM pg_roles AS role
     WHERE role.rolname = current_user;
    IF NOT executor_is_superuser THEN
        RAISE EXCEPTION
            'database administrator must run 399_eom_card_service_commitments';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_roles
         WHERE rolname = 'atlas'
           AND rolcanlogin
           AND NOT rolsuper
           AND NOT rolcreaterole
           AND NOT rolcreatedb
           AND NOT rolreplication
           AND NOT rolbypassrls
    ) OR NOT EXISTS (
        SELECT 1 FROM pg_roles
         WHERE rolname = 'atlas_eom_handoff_owner'
           AND NOT rolcanlogin
           AND NOT rolinherit
           AND NOT rolsuper
           AND NOT rolcreaterole
           AND NOT rolcreatedb
           AND NOT rolreplication
           AND NOT rolbypassrls
    ) THEN
        RAISE EXCEPTION 'required EOM runtime and guard roles are unavailable';
    END IF;
    IF NOT EXISTS (
        SELECT 1
          FROM pg_namespace AS namespace
         WHERE namespace.nspname = schema_name
           AND namespace.nspowner = (
               SELECT oid FROM pg_roles
                WHERE rolname = 'atlas_eom_handoff_owner'
           )
    ) THEN
        RAISE EXCEPTION 'the EOM funnel schema must retain guard ownership';
    END IF;
    IF to_regclass(format(
           '%I.eom_post_clean_onboarding_candidates', schema_name
       )) IS NULL
       OR to_regclass(format('%I.eom_card_vault_enrollments', schema_name)) IS NULL
       OR to_regclass(format('%I.contacts', schema_name)) IS NULL
       OR NOT EXISTS (
           SELECT 1 FROM schema_migrations
            WHERE name = '395_eom_post_clean_onboarding_candidates'
       )
       OR NOT EXISTS (
           SELECT 1 FROM schema_migrations
            WHERE name = '398_eom_card_vault'
       ) THEN
        RAISE EXCEPTION 'recorded migrations 395 and 398 are required';
    END IF;
    IF to_regclass(format(
           '%I.eom_post_clean_service_commitments', schema_name
       )) IS NOT NULL
       OR to_regprocedure(format(
           '%I.protect_eom_card_service_commitment()', schema_name
       )) IS NOT NULL
       OR to_regprocedure(format(
           '%I.require_eom_recurring_card_commitment()', schema_name
       )) IS NOT NULL THEN
        RAISE EXCEPTION
            'refusing to adopt a pre-existing EOM service-commitment object';
    END IF;
    IF EXISTS (SELECT 1 FROM eom_card_vault_enrollments) THEN
        RAISE EXCEPTION
            'existing EOM card enrollments require explicit reconciliation';
    END IF;
END;
$$;

CREATE TABLE eom_post_clean_service_commitments (
    id UUID CONSTRAINT pk_eom_post_clean_service_commitments PRIMARY KEY,
    candidate_id UUID NOT NULL,
    contact_id UUID NOT NULL,
    operation_key VARCHAR(128) NOT NULL,
    request_fingerprint VARCHAR(64) NOT NULL,
    service_commitment VARCHAR(16) NOT NULL,
    business_context_id VARCHAR(64) NOT NULL DEFAULT 'effingham_maids',
    decided_by_employee_id BIGINT NOT NULL,
    decided_by_name VARCHAR(128) NOT NULL,
    decided_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_eom_post_clean_service_commitment_candidate
        UNIQUE (candidate_id),
    CONSTRAINT uq_eom_post_clean_service_commitment_contact
        UNIQUE (contact_id),
    CONSTRAINT uq_eom_post_clean_service_commitment_operation
        UNIQUE (operation_key),
    CONSTRAINT fk_eom_post_clean_service_commitment_candidate
        FOREIGN KEY (candidate_id)
        REFERENCES eom_post_clean_onboarding_candidates(id) ON DELETE RESTRICT,
    CONSTRAINT fk_eom_post_clean_service_commitment_contact
        FOREIGN KEY (contact_id) REFERENCES contacts(id) ON DELETE RESTRICT,
    CONSTRAINT ck_eom_post_clean_service_commitment_operation
        CHECK (operation_key ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{15,127}$'),
    CONSTRAINT ck_eom_post_clean_service_commitment_fingerprint
        CHECK (request_fingerprint ~ '^[0-9a-f]{64}$'),
    CONSTRAINT ck_eom_post_clean_service_commitment_value
        CHECK (service_commitment IN ('recurring', 'one_time')),
    CONSTRAINT ck_eom_post_clean_service_commitment_context
        CHECK (business_context_id = 'effingham_maids'),
    CONSTRAINT ck_eom_post_clean_service_commitment_actor
        CHECK (
            decided_by_employee_id > 0
            AND char_length(btrim(decided_by_name)) BETWEEN 1 AND 128
        )
);

CREATE OR REPLACE FUNCTION protect_eom_card_service_commitment()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP <> 'INSERT' THEN
        RAISE EXCEPTION 'EOM service-commitment history is append-only';
    END IF;
    IF NOT EXISTS (
        SELECT 1
          FROM eom_post_clean_onboarding_candidates AS candidate
          JOIN contacts AS contact ON contact.id = candidate.contact_id
         WHERE candidate.id = NEW.candidate_id
           AND candidate.contact_id = NEW.contact_id
           AND candidate.business_context_id = NEW.business_context_id
           AND candidate.status = 'pending'
           AND contact.business_context_id = 'effingham_maids'
           AND contact.contact_type = 'customer'
           AND contact.customer_type = 'residential'
           AND contact.status = 'active'
    ) THEN
        RAISE EXCEPTION 'EOM service-commitment subject is not eligible';
    END IF;
    IF EXISTS (
        SELECT 1
          FROM eom_card_vault_enrollments AS enrollment
         WHERE enrollment.candidate_id = NEW.candidate_id
            OR enrollment.contact_id = NEW.contact_id
    ) THEN
        RAISE EXCEPTION
            'EOM service commitment must precede card enrollment';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION require_eom_recurring_card_commitment()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
          FROM eom_post_clean_service_commitments AS commitment
         WHERE commitment.candidate_id = NEW.candidate_id
           AND commitment.contact_id = NEW.contact_id
           AND commitment.business_context_id = NEW.business_context_id
           AND commitment.service_commitment = 'recurring'
    ) THEN
        RAISE EXCEPTION
            'EOM card enrollment requires a recurring service commitment';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_protect_eom_card_service_commitment
    BEFORE INSERT OR UPDATE OR DELETE ON eom_post_clean_service_commitments
    FOR EACH ROW EXECUTE FUNCTION protect_eom_card_service_commitment();
CREATE TRIGGER trg_protect_eom_card_service_commitment_truncate
    BEFORE TRUNCATE ON eom_post_clean_service_commitments
    FOR EACH STATEMENT EXECUTE FUNCTION protect_eom_card_service_commitment();
CREATE TRIGGER trg_require_eom_recurring_card_commitment
    BEFORE INSERT ON eom_card_vault_enrollments
    FOR EACH ROW EXECUTE FUNCTION require_eom_recurring_card_commitment();

DO $$
DECLARE
    schema_name TEXT := current_schema();
    function_name TEXT;
BEGIN
    FOREACH function_name IN ARRAY ARRAY[
        'protect_eom_card_service_commitment',
        'require_eom_recurring_card_commitment'
    ] LOOP
        EXECUTE format(
            'ALTER FUNCTION %I.%I() RESET ALL', schema_name, function_name
        );
        EXECUTE format(
            'ALTER FUNCTION %I.%I() SET search_path TO pg_catalog, %I, pg_temp',
            schema_name, function_name, schema_name
        );
        EXECUTE format(
            'ALTER FUNCTION %I.%I() OWNER TO atlas_eom_handoff_owner',
            schema_name, function_name
        );
        EXECUTE format(
            'REVOKE ALL PRIVILEGES ON FUNCTION %I.%I() FROM PUBLIC, atlas',
            schema_name, function_name
        );
    END LOOP;
    EXECUTE format(
        'ALTER TABLE %I.eom_post_clean_service_commitments '
        || 'OWNER TO atlas_eom_handoff_owner', schema_name
    );
    EXECUTE format(
        'REVOKE ALL PRIVILEGES ON TABLE '
        || '%I.eom_post_clean_service_commitments FROM PUBLIC, atlas',
        schema_name
    );
    EXECUTE format(
        'GRANT SELECT ON TABLE %I.eom_post_clean_service_commitments TO atlas',
        schema_name
    );
    EXECUTE format(
        'GRANT INSERT (id, candidate_id, contact_id, operation_key, '
        || 'request_fingerprint, service_commitment, decided_by_employee_id, '
        || 'decided_by_name) ON TABLE '
        || '%I.eom_post_clean_service_commitments TO atlas', schema_name
    );
END;
$$;

COMMENT ON TABLE eom_post_clean_service_commitments IS
    'Immutable operator decision: recurring card-required or one-time Terms-only.';
