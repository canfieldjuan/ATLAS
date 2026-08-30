-- atlas: atomic-bookkeeping
-- Durable, provider-confirmed EOM card-on-file authority. This migration
-- stores Stripe object identifiers only; it stores no card number, CVC,
-- expiration, bank data, charge, invoice, or price.
--
-- Controlled DBA-only migration. Rollback is retention-only: stop card-vault
-- callers and retain these additive audit objects and migration receipt.

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
        RAISE EXCEPTION 'database administrator must run 398_eom_card_vault';
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
    ) THEN
        RAISE EXCEPTION 'atlas must be an unprivileged login runtime role';
    END IF;
    IF NOT EXISTS (
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
        RAISE EXCEPTION 'atlas_eom_handoff_owner must be the isolated guard role';
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
    IF to_regclass(format('%I.contacts', schema_name)) IS NULL
       OR to_regclass(format(
            '%I.eom_post_clean_onboarding_candidates', schema_name
          )) IS NULL
       OR to_regclass(format('%I.eom_terms_acceptances', schema_name)) IS NULL
       OR to_regclass(format('%I.eom_terms_versions', schema_name)) IS NULL
       OR to_regclass(format('%I.schema_migrations', schema_name)) IS NULL
       OR NOT EXISTS (
           SELECT 1 FROM schema_migrations
            WHERE name = '395_eom_post_clean_onboarding_candidates'
       )
       OR NOT EXISTS (
           SELECT 1 FROM schema_migrations
            WHERE name = '397_eom_terms_acceptance'
       ) THEN
        RAISE EXCEPTION 'recorded migrations 395 and 397 are required';
    END IF;
    IF to_regclass(format('%I.eom_card_vault_enrollments', schema_name)) IS NOT NULL
       OR to_regclass(format('%I.eom_card_vault_sessions', schema_name)) IS NOT NULL
       OR to_regclass(format('%I.eom_card_vault_events', schema_name)) IS NOT NULL
       OR to_regprocedure(format(
            '%I.protect_eom_card_vault_enrollment()', schema_name
          )) IS NOT NULL
       OR to_regprocedure(format(
            '%I.protect_eom_card_vault_session()', schema_name
          )) IS NOT NULL
       OR to_regprocedure(format(
            '%I.protect_eom_card_vault_event()', schema_name
          )) IS NOT NULL THEN
        RAISE EXCEPTION 'refusing to adopt a pre-existing EOM card-vault object';
    END IF;
END;
$$;

CREATE TABLE eom_card_vault_enrollments (
    id UUID CONSTRAINT pk_eom_card_vault_enrollments PRIMARY KEY,
    candidate_id UUID NOT NULL,
    contact_id UUID NOT NULL,
    initial_acceptance_id UUID NOT NULL,
    business_context_id VARCHAR(64) NOT NULL DEFAULT 'effingham_maids',
    stripe_customer_id VARCHAR(255),
    status VARCHAR(16) NOT NULL DEFAULT 'pending',
    stripe_setup_intent_id VARCHAR(255),
    stripe_payment_method_id VARCHAR(255),
    ready_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_eom_card_vault_enrollment_candidate UNIQUE (candidate_id),
    CONSTRAINT uq_eom_card_vault_enrollment_contact UNIQUE (contact_id),
    CONSTRAINT uq_eom_card_vault_stripe_customer UNIQUE (stripe_customer_id),
    CONSTRAINT uq_eom_card_vault_setup_intent UNIQUE (stripe_setup_intent_id),
    CONSTRAINT uq_eom_card_vault_payment_method UNIQUE (stripe_payment_method_id),
    CONSTRAINT fk_eom_card_vault_enrollment_candidate
        FOREIGN KEY (candidate_id)
        REFERENCES eom_post_clean_onboarding_candidates(id) ON DELETE RESTRICT,
    CONSTRAINT fk_eom_card_vault_enrollment_contact
        FOREIGN KEY (contact_id) REFERENCES contacts(id) ON DELETE RESTRICT,
    CONSTRAINT fk_eom_card_vault_enrollment_initial_acceptance
        FOREIGN KEY (initial_acceptance_id)
        REFERENCES eom_terms_acceptances(id) ON DELETE RESTRICT,
    CONSTRAINT ck_eom_card_vault_enrollment_context
        CHECK (business_context_id = 'effingham_maids'),
    CONSTRAINT ck_eom_card_vault_enrollment_customer
        CHECK (
            stripe_customer_id IS NULL
            OR stripe_customer_id ~ '^cus_[A-Za-z0-9_]+$'
        ),
    CONSTRAINT ck_eom_card_vault_enrollment_state
        CHECK (
            (status = 'pending'
                AND stripe_setup_intent_id IS NULL
                AND stripe_payment_method_id IS NULL
                AND ready_at IS NULL)
            OR
            (status = 'ready'
                AND stripe_customer_id IS NOT NULL
                AND stripe_customer_id ~ '^cus_[A-Za-z0-9_]+$'
                AND stripe_setup_intent_id IS NOT NULL
                AND stripe_setup_intent_id ~ '^seti_[A-Za-z0-9_]+$'
                AND stripe_payment_method_id IS NOT NULL
                AND stripe_payment_method_id ~ '^pm_[A-Za-z0-9_]+$'
                AND ready_at IS NOT NULL)
        )
);

CREATE TABLE eom_card_vault_sessions (
    id UUID CONSTRAINT pk_eom_card_vault_sessions PRIMARY KEY,
    enrollment_id UUID NOT NULL,
    acceptance_id UUID NOT NULL,
    state VARCHAR(16) NOT NULL DEFAULT 'creating',
    stripe_checkout_session_id VARCHAR(255),
    checkout_expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_eom_card_vault_checkout_session
        UNIQUE (stripe_checkout_session_id),
    CONSTRAINT fk_eom_card_vault_session_enrollment
        FOREIGN KEY (enrollment_id)
        REFERENCES eom_card_vault_enrollments(id) ON DELETE RESTRICT,
    CONSTRAINT fk_eom_card_vault_session_acceptance
        FOREIGN KEY (acceptance_id)
        REFERENCES eom_terms_acceptances(id) ON DELETE RESTRICT,
    CONSTRAINT ck_eom_card_vault_session_state
        CHECK (
            (state = 'creating'
                AND stripe_checkout_session_id IS NULL
                AND checkout_expires_at IS NULL)
            OR
            (state = 'open'
                AND stripe_checkout_session_id IS NOT NULL
                AND stripe_checkout_session_id ~ '^cs_[A-Za-z0-9_]+$'
                AND checkout_expires_at IS NOT NULL
                AND checkout_expires_at > created_at)
        )
);

CREATE TABLE eom_card_vault_events (
    stripe_event_id VARCHAR(255)
        CONSTRAINT pk_eom_card_vault_events PRIMARY KEY,
    enrollment_id UUID NOT NULL,
    session_id UUID NOT NULL,
    stripe_setup_intent_id VARCHAR(255) NOT NULL,
    stripe_payment_method_id VARCHAR(255) NOT NULL,
    received_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_eom_card_vault_event_session UNIQUE (session_id),
    CONSTRAINT fk_eom_card_vault_event_enrollment
        FOREIGN KEY (enrollment_id)
        REFERENCES eom_card_vault_enrollments(id) ON DELETE RESTRICT,
    CONSTRAINT fk_eom_card_vault_event_session
        FOREIGN KEY (session_id)
        REFERENCES eom_card_vault_sessions(id) ON DELETE RESTRICT,
    CONSTRAINT ck_eom_card_vault_event_id
        CHECK (stripe_event_id ~ '^evt_[A-Za-z0-9_]+$'),
    CONSTRAINT ck_eom_card_vault_event_setup_intent
        CHECK (stripe_setup_intent_id ~ '^seti_[A-Za-z0-9_]+$'),
    CONSTRAINT ck_eom_card_vault_event_payment_method
        CHECK (stripe_payment_method_id ~ '^pm_[A-Za-z0-9_]+$')
);

CREATE OR REPLACE FUNCTION protect_eom_card_vault_enrollment()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'TRUNCATE' THEN
        RAISE EXCEPTION 'EOM card-vault enrollment history is append-only';
    END IF;
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'EOM card-vault enrollment history cannot be deleted';
    END IF;
    IF TG_OP = 'INSERT' THEN
        IF NOT EXISTS (
            SELECT 1
              FROM eom_post_clean_onboarding_candidates AS candidate
              JOIN contacts AS contact ON contact.id = candidate.contact_id
              JOIN eom_terms_acceptances AS acceptance
                ON acceptance.id = NEW.initial_acceptance_id
               AND acceptance.contact_id = contact.id
              JOIN eom_terms_versions AS accepted_version
                ON accepted_version.id = acceptance.version_id
             WHERE candidate.id = NEW.candidate_id
               AND contact.id = NEW.contact_id
               AND candidate.status = 'pending'
               AND contact.business_context_id = 'effingham_maids'
               AND contact.contact_type = 'customer'
               AND contact.status = 'active'
               AND contact.customer_type = 'residential'
               AND acceptance.audience = 'residential'
               AND NOT EXISTS (
                   SELECT 1
                     FROM eom_terms_versions AS later
                    WHERE later.status = 'published'
                      AND later.material_change
                      AND later.publication_order
                          > accepted_version.publication_order
               )
        ) THEN
            RAISE EXCEPTION 'EOM card-vault enrollment subject is not eligible';
        END IF;
        RETURN NEW;
    END IF;
    IF ROW(
        NEW.id, NEW.candidate_id, NEW.contact_id, NEW.initial_acceptance_id,
        NEW.business_context_id, NEW.created_at
    ) IS DISTINCT FROM ROW(
        OLD.id, OLD.candidate_id, OLD.contact_id, OLD.initial_acceptance_id,
        OLD.business_context_id, OLD.created_at
    ) THEN
        RAISE EXCEPTION 'EOM card-vault enrollment identity is immutable';
    END IF;
    IF OLD.stripe_customer_id IS NOT NULL
       AND NEW.stripe_customer_id IS DISTINCT FROM OLD.stripe_customer_id THEN
        RAISE EXCEPTION 'EOM card-vault Stripe customer is immutable';
    END IF;
    IF OLD.status = 'ready' AND NEW IS DISTINCT FROM OLD THEN
        RAISE EXCEPTION 'ready EOM card-vault enrollment is immutable';
    END IF;
    IF OLD.status = 'pending' AND NEW.status = 'ready'
       AND NOT EXISTS (
           SELECT 1
             FROM eom_card_vault_events AS event
            WHERE event.enrollment_id = NEW.id
              AND event.stripe_setup_intent_id = NEW.stripe_setup_intent_id
              AND event.stripe_payment_method_id = NEW.stripe_payment_method_id
       ) THEN
        RAISE EXCEPTION 'ready EOM card-vault enrollment requires event evidence';
    END IF;
    IF OLD.status = 'pending' AND NEW.status IN ('pending', 'ready') THEN
        RETURN NEW;
    END IF;
    RAISE EXCEPTION 'EOM card-vault enrollment transition is invalid';
END;
$$;

CREATE OR REPLACE FUNCTION protect_eom_card_vault_session()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'TRUNCATE' THEN
        RAISE EXCEPTION 'EOM card-vault session history is append-only';
    END IF;
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'EOM card-vault session history cannot be deleted';
    END IF;
    IF TG_OP = 'INSERT' THEN
        IF NOT EXISTS (
            SELECT 1
              FROM eom_card_vault_enrollments AS enrollment
              JOIN eom_terms_acceptances AS acceptance
                ON acceptance.id = NEW.acceptance_id
               AND acceptance.contact_id = enrollment.contact_id
               AND acceptance.audience = 'residential'
             WHERE enrollment.id = NEW.enrollment_id
               AND enrollment.status = 'pending'
        ) THEN
            RAISE EXCEPTION 'EOM card-vault session requires pending enrollment';
        END IF;
        RETURN NEW;
    END IF;
    IF ROW(NEW.id, NEW.enrollment_id, NEW.acceptance_id, NEW.created_at)
       IS DISTINCT FROM ROW(
           OLD.id, OLD.enrollment_id, OLD.acceptance_id, OLD.created_at
       ) THEN
        RAISE EXCEPTION 'EOM card-vault session identity is immutable';
    END IF;
    IF OLD.state = 'creating' AND NEW.state = 'open' THEN
        RETURN NEW;
    END IF;
    IF NEW IS NOT DISTINCT FROM OLD THEN
        RETURN NEW;
    END IF;
    RAISE EXCEPTION 'EOM card-vault session transition is invalid';
END;
$$;

CREATE OR REPLACE FUNCTION protect_eom_card_vault_event()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP <> 'INSERT' THEN
        RAISE EXCEPTION 'EOM card-vault event evidence is append-only';
    END IF;
    IF NOT EXISTS (
        SELECT 1
          FROM eom_card_vault_sessions AS session
         WHERE session.id = NEW.session_id
           AND session.enrollment_id = NEW.enrollment_id
           AND session.state = 'open'
    ) THEN
        RAISE EXCEPTION 'EOM card-vault event subject does not match';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_protect_eom_card_vault_enrollment
    BEFORE INSERT OR UPDATE OR DELETE ON eom_card_vault_enrollments
    FOR EACH ROW EXECUTE FUNCTION protect_eom_card_vault_enrollment();
CREATE TRIGGER trg_protect_eom_card_vault_enrollment_truncate
    BEFORE TRUNCATE ON eom_card_vault_enrollments
    FOR EACH STATEMENT EXECUTE FUNCTION protect_eom_card_vault_enrollment();
CREATE TRIGGER trg_protect_eom_card_vault_session
    BEFORE INSERT OR UPDATE OR DELETE ON eom_card_vault_sessions
    FOR EACH ROW EXECUTE FUNCTION protect_eom_card_vault_session();
CREATE TRIGGER trg_protect_eom_card_vault_session_truncate
    BEFORE TRUNCATE ON eom_card_vault_sessions
    FOR EACH STATEMENT EXECUTE FUNCTION protect_eom_card_vault_session();
CREATE TRIGGER trg_protect_eom_card_vault_event
    BEFORE INSERT OR UPDATE OR DELETE ON eom_card_vault_events
    FOR EACH ROW EXECUTE FUNCTION protect_eom_card_vault_event();
CREATE TRIGGER trg_protect_eom_card_vault_event_truncate
    BEFORE TRUNCATE ON eom_card_vault_events
    FOR EACH STATEMENT EXECUTE FUNCTION protect_eom_card_vault_event();

CREATE INDEX idx_eom_card_vault_sessions_enrollment_created
    ON eom_card_vault_sessions (enrollment_id, created_at DESC, id DESC);
CREATE INDEX idx_eom_card_vault_enrollments_status
    ON eom_card_vault_enrollments (status, created_at, id);

DO $$
DECLARE
    schema_name TEXT := current_schema();
    table_name TEXT;
    function_name TEXT;
BEGIN
    FOREACH function_name IN ARRAY ARRAY[
        'protect_eom_card_vault_enrollment',
        'protect_eom_card_vault_session',
        'protect_eom_card_vault_event'
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
    FOREACH table_name IN ARRAY ARRAY[
        'eom_card_vault_enrollments',
        'eom_card_vault_sessions',
        'eom_card_vault_events'
    ] LOOP
        EXECUTE format(
            'ALTER TABLE %I.%I OWNER TO atlas_eom_handoff_owner',
            schema_name, table_name
        );
        EXECUTE format(
            'REVOKE ALL PRIVILEGES ON TABLE %I.%I FROM PUBLIC, atlas',
            schema_name, table_name
        );
    END LOOP;

    EXECUTE format(
        'GRANT SELECT ON TABLE %I.eom_card_vault_enrollments TO atlas',
        schema_name
    );
    EXECUTE format(
        'GRANT INSERT (id, candidate_id, contact_id, initial_acceptance_id) '
        || 'ON TABLE %I.eom_card_vault_enrollments TO atlas', schema_name
    );
    EXECUTE format(
        'GRANT UPDATE (stripe_customer_id, status, stripe_setup_intent_id, '
        || 'stripe_payment_method_id, ready_at) '
        || 'ON TABLE %I.eom_card_vault_enrollments TO atlas', schema_name
    );
    EXECUTE format(
        'GRANT SELECT ON TABLE %I.eom_card_vault_sessions TO atlas', schema_name
    );
    EXECUTE format(
        'GRANT INSERT (id, enrollment_id, acceptance_id) '
        || 'ON TABLE %I.eom_card_vault_sessions TO atlas', schema_name
    );
    EXECUTE format(
        'GRANT UPDATE (state, stripe_checkout_session_id, checkout_expires_at) '
        || 'ON TABLE %I.eom_card_vault_sessions TO atlas', schema_name
    );
    EXECUTE format(
        'GRANT SELECT ON TABLE %I.eom_card_vault_events TO atlas', schema_name
    );
    EXECUTE format(
        'GRANT INSERT (stripe_event_id, enrollment_id, session_id, '
        || 'stripe_setup_intent_id, stripe_payment_method_id) '
        || 'ON TABLE %I.eom_card_vault_events TO atlas', schema_name
    );
END;
$$;

COMMENT ON TABLE eom_card_vault_enrollments IS
    'One residential post-clean card-vault authority; ready only after provider confirmation.';
COMMENT ON TABLE eom_card_vault_sessions IS
    'Durable hosted Checkout setup attempts keyed before provider creation.';
COMMENT ON TABLE eom_card_vault_events IS
    'Immutable Stripe event evidence used for idempotent ready transitions.';
