-- atlas: atomic-bookkeeping
-- Customer-bound EOM Terms invitations, immutable acceptances, and stable
-- delivery evidence. The operator-published document authority remains in
-- migration 396; this migration stores references and exact rendered payloads
-- but seeds or edits no Terms prose.
--
-- This is a controlled DBA-only migration. The direct Atlas runtime may issue
-- an invitation, append one acceptance, and advance delivery/revocation state
-- through narrow one-way transitions, but it must not own or rewrite legal
-- evidence. Apply only through the dedicated controlled preflight/apply command.
--
-- Rollback: stop the invitation/acceptance routes and their consumers before
-- rolling application code back. Retain every table, trigger, function, grant,
-- and migration receipt as legal/audit history.

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
            'database administrator must run 397_eom_terms_acceptance';
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
            'atlas must be an unprivileged login runtime role before running 397_eom_terms_acceptance';
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
            'atlas_eom_handoff_owner must be the isolated no-login EOM guard role before running 397_eom_terms_acceptance';
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
            'atlas_eom_handoff_owner must have no live sessions before running 397_eom_terms_acceptance';
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
            'the EOM funnel schema must be owned by atlas_eom_handoff_owner before running 397_eom_terms_acceptance';
    END IF;

    IF to_regclass(format('%I.contacts', schema_name)) IS NULL
       OR to_regclass(format('%I.eom_terms_versions', schema_name)) IS NULL
       OR to_regclass(format('%I.eom_terms_current_version', schema_name)) IS NULL
       OR to_regclass(format('%I.schema_migrations', schema_name)) IS NULL
       OR NOT EXISTS (
           SELECT 1
             FROM schema_migrations
            WHERE name = '396_eom_terms_authority'
       ) THEN
        RAISE EXCEPTION
            'contacts and recorded migration 396_eom_terms_authority are required before running 397_eom_terms_acceptance';
    END IF;

    IF (
        SELECT count(*)
          FROM pg_class AS relation
          JOIN pg_namespace AS namespace
            ON namespace.oid = relation.relnamespace
         WHERE namespace.nspname = schema_name
           AND relation.relkind = 'r'
           AND relation.relname IN (
               'eom_terms_versions',
               'eom_terms_current_version'
           )
           AND relation.relowner = (
               SELECT oid
                 FROM pg_roles
                WHERE rolname = 'atlas_eom_handoff_owner'
           )
    ) <> 2 OR (
        SELECT count(*)
          FROM pg_proc AS guarded_function
          JOIN pg_namespace AS namespace
            ON namespace.oid = guarded_function.pronamespace
         WHERE namespace.nspname = schema_name
           AND guarded_function.proname IN (
               'protect_eom_terms_version',
               'require_published_eom_terms_current_version',
               'prevent_eom_terms_current_removal'
           )
           AND guarded_function.pronargs = 0
           AND guarded_function.proowner = (
               SELECT oid
                 FROM pg_roles
                WHERE rolname = 'atlas_eom_handoff_owner'
           )
    ) <> 3 THEN
        RAISE EXCEPTION
            'migration 396 Terms authority must retain guard ownership before running 397_eom_terms_acceptance';
    END IF;

    IF to_regclass(format('%I.eom_terms_invitations', schema_name)) IS NOT NULL
       OR to_regclass(format('%I.eom_terms_acceptances', schema_name)) IS NOT NULL
       OR to_regclass(format('%I.eom_terms_deliveries', schema_name)) IS NOT NULL
       OR to_regprocedure(format(
            '%I.validate_eom_terms_invitation()', schema_name
          )) IS NOT NULL
       OR to_regprocedure(format(
            '%I.protect_eom_terms_invitation()', schema_name
          )) IS NOT NULL
       OR to_regprocedure(format(
            '%I.validate_eom_terms_acceptance()', schema_name
          )) IS NOT NULL
       OR to_regprocedure(format(
            '%I.protect_eom_terms_acceptance()', schema_name
          )) IS NOT NULL
       OR to_regprocedure(format(
            '%I.protect_eom_terms_delivery()', schema_name
          )) IS NOT NULL THEN
        RAISE EXCEPTION
            'refusing to adopt a pre-existing EOM Terms acceptance object';
    END IF;

    IF EXISTS (
        SELECT 1
          FROM pg_attribute AS attribute
         WHERE attribute.attrelid = to_regclass(format(
                   '%I.eom_terms_versions', schema_name
               ))
           AND attribute.attname = 'publication_order'
           AND attribute.attnum > 0
           AND NOT attribute.attisdropped
    ) THEN
        RAISE EXCEPTION
            'refusing to adopt pre-existing EOM Terms publication ordering';
    END IF;
END;
$$;

ALTER TABLE eom_terms_versions
    ADD COLUMN publication_order BIGINT;

-- Migration 396 made published rows immutable. Temporarily suspend that one
-- guard inside this atomic DBA migration so existing releases can receive a
-- deterministic order. The selected current release is always placed last;
-- ties among historical releases are stable but make no claim about chronology
-- that was not recorded before this migration.
ALTER TABLE eom_terms_versions
    DISABLE TRIGGER trg_protect_eom_terms_version;

WITH ordered_release AS (
    SELECT version.id,
           row_number() OVER (
               ORDER BY
                   CASE WHEN version.id = (
                       SELECT selected.version_id
                         FROM eom_terms_current_version AS selected
                        WHERE selected.singleton
                   ) THEN 1 ELSE 0 END,
                   version.published_at,
                   version.created_at,
                   version.id
           ) AS publication_order
      FROM eom_terms_versions AS version
     WHERE version.status = 'published'
)
UPDATE eom_terms_versions AS version
   SET publication_order = ordered_release.publication_order
  FROM ordered_release
 WHERE version.id = ordered_release.id;

ALTER TABLE eom_terms_versions
    ENABLE TRIGGER trg_protect_eom_terms_version;

ALTER TABLE eom_terms_versions
    ADD CONSTRAINT uq_eom_terms_publication_order
        UNIQUE (publication_order),
    ADD CONSTRAINT ck_eom_terms_publication_order
        CHECK (
            (status = 'draft' AND publication_order IS NULL)
            OR
            (status = 'published' AND publication_order > 0)
        );

-- Extend migration 396's existing guard instead of adding another trigger.
-- The previous application release attests exactly five authority triggers;
-- keeping that boundary stable makes both a rolling deploy and app rollback
-- safe after this migration commits.
CREATE OR REPLACE FUNCTION protect_eom_terms_version()
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
    IF OLD.status = 'draft' AND NEW.status = 'published' THEN
        IF NEW.publication_order IS NOT NULL THEN
            RAISE EXCEPTION
                'EOM Terms publication order is database assigned';
        END IF;
        PERFORM pg_advisory_xact_lock(
            hashtextextended('eom-terms-current-version', 0)
        );
        SELECT COALESCE(max(version.publication_order), 0) + 1
          INTO NEW.publication_order
          FROM eom_terms_versions AS version
         WHERE version.status = 'published';
    ELSIF NEW.publication_order IS DISTINCT FROM OLD.publication_order THEN
        RAISE EXCEPTION 'EOM Terms publication order is immutable';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TABLE eom_terms_invitations (
    id UUID CONSTRAINT pk_eom_terms_invitations PRIMARY KEY,
    request_key VARCHAR(128) NOT NULL,
    business_context_id VARCHAR(64) NOT NULL DEFAULT 'effingham_maids',
    contact_id UUID NOT NULL,
    version_id UUID NOT NULL,
    audience VARCHAR(16) NOT NULL,
    locale VARCHAR(2) NOT NULL,
    customer_name VARCHAR(256) NOT NULL,
    recipient_email VARCHAR(256) NOT NULL,
    public_base_url VARCHAR(2048) NOT NULL,
    signing_key_fingerprint VARCHAR(64) NOT NULL,
    issued_at TIMESTAMPTZ NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    issued_by_id BIGINT NOT NULL,
    issued_by_name VARCHAR(128) NOT NULL,
    revoked_at TIMESTAMPTZ,
    revoked_by_id BIGINT,
    revoked_by_name VARCHAR(128),
    CONSTRAINT uq_eom_terms_invitations_request_key UNIQUE (request_key),
    CONSTRAINT fk_eom_terms_invitations_contact
        FOREIGN KEY (contact_id) REFERENCES contacts(id) ON DELETE RESTRICT,
    CONSTRAINT fk_eom_terms_invitations_version
        FOREIGN KEY (version_id) REFERENCES eom_terms_versions(id)
        ON DELETE RESTRICT,
    CONSTRAINT ck_eom_terms_invitations_request_key
        CHECK (request_key ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{15,127}$'),
    CONSTRAINT ck_eom_terms_invitations_context
        CHECK (business_context_id = 'effingham_maids'),
    CONSTRAINT ck_eom_terms_invitations_audience
        CHECK (audience IN ('residential', 'commercial')),
    CONSTRAINT ck_eom_terms_invitations_locale
        CHECK (locale IN ('en', 'es')),
    CONSTRAINT ck_eom_terms_invitations_customer_name
        CHECK (length(btrim(customer_name)) BETWEEN 1 AND 256),
    CONSTRAINT ck_eom_terms_invitations_recipient
        CHECK (
            recipient_email = lower(btrim(recipient_email))
            AND recipient_email ~ '^[^@[:space:]]+@[^@[:space:]]+\.[^@[:space:]]+$'
        ),
    CONSTRAINT ck_eom_terms_invitations_public_url
        CHECK (
            public_base_url = btrim(public_base_url)
            AND public_base_url ~ '^https://[^#]+$'
        ),
    CONSTRAINT ck_eom_terms_invitations_key_fingerprint
        CHECK (signing_key_fingerprint ~ '^[0-9a-f]{64}$'),
    CONSTRAINT ck_eom_terms_invitations_window
        CHECK (
            issued_at IS NOT NULL
            AND expires_at = issued_at + INTERVAL '30 days'
        ),
    CONSTRAINT ck_eom_terms_invitations_issuer
        CHECK (issued_by_id > 0 AND length(btrim(issued_by_name)) > 0),
    CONSTRAINT ck_eom_terms_invitations_revocation
        CHECK (
            (revoked_at IS NULL
                AND revoked_by_id IS NULL
                AND revoked_by_name IS NULL)
            OR
            (revoked_at IS NOT NULL
                AND revoked_by_id > 0
                AND length(btrim(revoked_by_name)) > 0)
        )
);

CREATE TABLE eom_terms_acceptances (
    id UUID CONSTRAINT pk_eom_terms_acceptances PRIMARY KEY,
    invitation_id UUID NOT NULL,
    business_context_id VARCHAR(64) NOT NULL,
    contact_id UUID NOT NULL,
    version_id UUID NOT NULL,
    audience VARCHAR(16) NOT NULL,
    locale VARCHAR(2) NOT NULL,
    recipient_email VARCHAR(256) NOT NULL,
    signer_name VARCHAR(256) NOT NULL,
    terms_accepted BOOLEAN NOT NULL,
    additional_work_accepted BOOLEAN NOT NULL,
    client_ip INET NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    accepted_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT uq_eom_terms_acceptances_invitation UNIQUE (invitation_id),
    CONSTRAINT fk_eom_terms_acceptances_invitation
        FOREIGN KEY (invitation_id) REFERENCES eom_terms_invitations(id)
        ON DELETE RESTRICT,
    CONSTRAINT fk_eom_terms_acceptances_contact
        FOREIGN KEY (contact_id) REFERENCES contacts(id) ON DELETE RESTRICT,
    CONSTRAINT fk_eom_terms_acceptances_version
        FOREIGN KEY (version_id) REFERENCES eom_terms_versions(id)
        ON DELETE RESTRICT,
    CONSTRAINT ck_eom_terms_acceptances_context
        CHECK (business_context_id = 'effingham_maids'),
    CONSTRAINT ck_eom_terms_acceptances_audience
        CHECK (audience IN ('residential', 'commercial')),
    CONSTRAINT ck_eom_terms_acceptances_locale
        CHECK (locale IN ('en', 'es')),
    CONSTRAINT ck_eom_terms_acceptances_recipient
        CHECK (
            recipient_email = lower(btrim(recipient_email))
            AND recipient_email ~ '^[^@[:space:]]+@[^@[:space:]]+\.[^@[:space:]]+$'
        ),
    CONSTRAINT ck_eom_terms_acceptances_signer
        CHECK (length(btrim(signer_name)) BETWEEN 1 AND 256),
    CONSTRAINT ck_eom_terms_acceptances_acknowledgements
        CHECK (terms_accepted AND additional_work_accepted),
    CONSTRAINT ck_eom_terms_acceptances_content_hash
        CHECK (content_hash ~ '^[0-9a-f]{64}$')
);

CREATE TABLE eom_terms_deliveries (
    id UUID CONSTRAINT pk_eom_terms_deliveries PRIMARY KEY,
    kind VARCHAR(24) NOT NULL,
    invitation_id UUID NOT NULL,
    acceptance_id UUID,
    recipient_email VARCHAR(256) NOT NULL,
    subject VARCHAR(512) NOT NULL,
    body TEXT NOT NULL,
    body_hash VARCHAR(64) NOT NULL,
    status VARCHAR(16) NOT NULL DEFAULT 'pending',
    claimed_at TIMESTAMPTZ,
    sent_at TIMESTAMPTZ,
    resend_message_id VARCHAR(256),
    transport_idempotent_replay BOOLEAN,
    confirmed_by_id BIGINT,
    confirmed_by_name VARCHAR(128),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_eom_terms_deliveries_kind_invitation
        UNIQUE (kind, invitation_id),
    CONSTRAINT uq_eom_terms_deliveries_acceptance UNIQUE (acceptance_id),
    CONSTRAINT fk_eom_terms_deliveries_invitation
        FOREIGN KEY (invitation_id) REFERENCES eom_terms_invitations(id)
        ON DELETE RESTRICT,
    CONSTRAINT fk_eom_terms_deliveries_acceptance
        FOREIGN KEY (acceptance_id) REFERENCES eom_terms_acceptances(id)
        ON DELETE RESTRICT,
    CONSTRAINT ck_eom_terms_deliveries_kind
        CHECK (kind IN ('invitation', 'executed_copy')),
    CONSTRAINT ck_eom_terms_deliveries_shape
        CHECK (
            (kind = 'invitation' AND acceptance_id IS NULL)
            OR
            (kind = 'executed_copy' AND acceptance_id IS NOT NULL)
        ),
    CONSTRAINT ck_eom_terms_deliveries_recipient
        CHECK (
            recipient_email = lower(btrim(recipient_email))
            AND recipient_email ~ '^[^@[:space:]]+@[^@[:space:]]+\.[^@[:space:]]+$'
        ),
    CONSTRAINT ck_eom_terms_deliveries_subject
        CHECK (length(btrim(subject)) BETWEEN 1 AND 512),
    CONSTRAINT ck_eom_terms_deliveries_body
        CHECK (length(body) > 0),
    CONSTRAINT ck_eom_terms_deliveries_body_hash
        CHECK (body_hash ~ '^[0-9a-f]{64}$'),
    CONSTRAINT ck_eom_terms_deliveries_status
        CHECK (status IN ('pending', 'sending', 'sent')),
    CONSTRAINT ck_eom_terms_deliveries_state
        CHECK (
            (status = 'pending'
                AND claimed_at IS NULL
                AND sent_at IS NULL
                AND resend_message_id IS NULL
                AND transport_idempotent_replay IS NULL
                AND confirmed_by_id IS NULL
                AND confirmed_by_name IS NULL)
            OR
            (status = 'sending'
                AND claimed_at IS NOT NULL
                AND sent_at IS NULL
                AND resend_message_id IS NULL
                AND transport_idempotent_replay IS NULL
                AND confirmed_by_id IS NULL
                AND confirmed_by_name IS NULL)
            OR
            (status = 'sent'
                AND claimed_at IS NOT NULL
                AND sent_at IS NOT NULL
                AND (
                    (((length(btrim(resend_message_id)) > 0
                            AND transport_idempotent_replay IS NOT NULL)
                        OR
                        (resend_message_id IS NULL
                            AND transport_idempotent_replay IS TRUE))
                        AND confirmed_by_id IS NULL
                        AND confirmed_by_name IS NULL)
                    OR
                    (resend_message_id IS NULL
                        AND transport_idempotent_replay IS NULL
                        AND confirmed_by_id > 0
                        AND length(btrim(confirmed_by_name)) > 0)
                ))
        )
);

CREATE FUNCTION validate_eom_terms_invitation()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY INVOKER
AS $$
DECLARE
    authoritative_time TIMESTAMPTZ;
BEGIN
    PERFORM pg_advisory_xact_lock(
        hashtextextended('eom-terms-current-version', 0)
    );
    IF NEW.revoked_at IS NOT NULL
       OR NEW.revoked_by_id IS NOT NULL
       OR NEW.revoked_by_name IS NOT NULL THEN
        RAISE EXCEPTION 'new EOM Terms invitation cannot start revoked';
    END IF;
    IF NOT EXISTS (
        SELECT 1
          FROM contacts AS contact
          JOIN eom_terms_current_version AS selected ON selected.singleton
          JOIN eom_terms_versions AS version
            ON version.id = selected.version_id
           AND version.status = 'published'
         WHERE contact.id = NEW.contact_id
           AND contact.business_context_id = 'effingham_maids'
           AND contact.contact_type = 'customer'
           AND contact.status = 'active'
           AND contact.customer_type IN ('residential', 'commercial')
           AND contact.customer_type = NEW.audience
           AND btrim(contact.full_name) = NEW.customer_name
           AND lower(btrim(contact.email)) = NEW.recipient_email
           AND version.id = NEW.version_id
    ) THEN
        RAISE EXCEPTION
            'EOM Terms invitation requires the current release and matching active customer';
    END IF;
    authoritative_time := clock_timestamp();
    NEW.business_context_id := 'effingham_maids';
    NEW.issued_at := authoritative_time;
    NEW.expires_at := authoritative_time + INTERVAL '30 days';
    RETURN NEW;
END;
$$;

CREATE FUNCTION protect_eom_terms_invitation()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY INVOKER
AS $$
BEGIN
    IF TG_OP = 'TRUNCATE' OR TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'EOM Terms invitations cannot be removed';
    END IF;
    IF ROW(
        NEW.id, NEW.request_key, NEW.business_context_id, NEW.contact_id,
        NEW.version_id, NEW.audience, NEW.locale, NEW.customer_name,
        NEW.recipient_email, NEW.public_base_url,
        NEW.signing_key_fingerprint, NEW.issued_at, NEW.expires_at,
        NEW.issued_by_id, NEW.issued_by_name
    ) IS DISTINCT FROM ROW(
        OLD.id, OLD.request_key, OLD.business_context_id, OLD.contact_id,
        OLD.version_id, OLD.audience, OLD.locale, OLD.customer_name,
        OLD.recipient_email, OLD.public_base_url,
        OLD.signing_key_fingerprint, OLD.issued_at, OLD.expires_at,
        OLD.issued_by_id, OLD.issued_by_name
    ) THEN
        RAISE EXCEPTION 'EOM Terms invitation identity is immutable';
    END IF;
    IF OLD.revoked_at IS NOT NULL THEN
        IF ROW(NEW.revoked_at, NEW.revoked_by_id, NEW.revoked_by_name)
           IS DISTINCT FROM
           ROW(OLD.revoked_at, OLD.revoked_by_id, OLD.revoked_by_name) THEN
            RAISE EXCEPTION 'EOM Terms invitation revocation is immutable';
        END IF;
        RETURN NEW;
    END IF;
    IF NEW.revoked_at IS NULL
       AND NEW.revoked_by_id IS NULL
       AND NEW.revoked_by_name IS NULL THEN
        RETURN NEW;
    END IF;
    IF NEW.revoked_by_id IS NULL
       OR NEW.revoked_by_id <= 0
       OR length(btrim(NEW.revoked_by_name)) = 0 THEN
        RAISE EXCEPTION 'EOM Terms invitation revocation requires an actor';
    END IF;
    IF EXISTS (
        SELECT 1
          FROM eom_terms_deliveries AS delivery
         WHERE delivery.invitation_id = OLD.id
           AND delivery.kind = 'invitation'
           AND delivery.status = 'sending'
    ) THEN
        RAISE EXCEPTION
            'EOM Terms invitation delivery requires reconciliation';
    END IF;
    IF EXISTS (
        SELECT 1
          FROM eom_terms_acceptances AS acceptance
         WHERE acceptance.invitation_id = OLD.id
    ) THEN
        RAISE EXCEPTION 'accepted EOM Terms invitation cannot be revoked';
    END IF;
    NEW.revoked_at := clock_timestamp();
    RETURN NEW;
END;
$$;

CREATE FUNCTION validate_eom_terms_acceptance()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY INVOKER
AS $$
DECLARE
    invitation_row RECORD;
    authoritative_time TIMESTAMPTZ;
BEGIN
    PERFORM pg_advisory_xact_lock(
        hashtextextended('eom-terms-current-version', 0)
    );
    authoritative_time := clock_timestamp();
    SELECT invitation.contact_id,
           invitation.version_id,
           invitation.audience,
           invitation.locale,
           invitation.recipient_email,
           invitation.revoked_at,
           invitation.expires_at,
           version.content_hash,
           version.publication_order,
           invitation_delivery.status AS delivery_status
      INTO invitation_row
      FROM eom_terms_invitations AS invitation
      JOIN eom_terms_versions AS version
        ON version.id = invitation.version_id
       AND version.status = 'published'
      JOIN eom_terms_deliveries AS invitation_delivery
        ON invitation_delivery.invitation_id = invitation.id
       AND invitation_delivery.kind = 'invitation'
     WHERE invitation.id = NEW.invitation_id
     FOR SHARE OF invitation;
    IF NOT FOUND
       OR invitation_row.revoked_at IS NOT NULL
       OR authoritative_time > invitation_row.expires_at THEN
        RAISE EXCEPTION 'EOM Terms invitation is unavailable';
    END IF;
    IF EXISTS (
        SELECT 1
          FROM eom_terms_versions AS later
         WHERE later.status = 'published'
           AND later.material_change
           AND later.publication_order > invitation_row.publication_order
    ) THEN
        RAISE EXCEPTION
            'EOM Terms invitation was superseded by a material release';
    END IF;
    IF invitation_row.delivery_status = 'sending' THEN
        RAISE EXCEPTION
            'EOM Terms invitation delivery requires reconciliation';
    END IF;
    IF NOT EXISTS (
        SELECT 1
          FROM contacts AS contact
         WHERE contact.id = invitation_row.contact_id
           AND contact.business_context_id = 'effingham_maids'
           AND contact.contact_type = 'customer'
           AND contact.status = 'active'
           AND contact.customer_type = invitation_row.audience
           AND lower(btrim(contact.email)) = invitation_row.recipient_email
    ) THEN
        RAISE EXCEPTION
            'EOM Terms invitation no longer matches an active customer';
    END IF;
    NEW.business_context_id := 'effingham_maids';
    NEW.contact_id := invitation_row.contact_id;
    NEW.version_id := invitation_row.version_id;
    NEW.audience := invitation_row.audience;
    NEW.locale := invitation_row.locale;
    NEW.recipient_email := invitation_row.recipient_email;
    NEW.content_hash := invitation_row.content_hash;
    NEW.accepted_at := authoritative_time;
    RETURN NEW;
END;
$$;

CREATE FUNCTION protect_eom_terms_acceptance()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY INVOKER
AS $$
BEGIN
    RAISE EXCEPTION 'EOM Terms acceptance evidence is append-only';
END;
$$;

CREATE FUNCTION protect_eom_terms_delivery()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY INVOKER
AS $$
BEGIN
    IF TG_OP = 'TRUNCATE' OR TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'EOM Terms delivery evidence cannot be removed';
    END IF;
    IF TG_OP = 'INSERT' THEN
        IF NEW.status <> 'pending'
           OR NEW.claimed_at IS NOT NULL
           OR NEW.sent_at IS NOT NULL
           OR NEW.resend_message_id IS NOT NULL
           OR NEW.transport_idempotent_replay IS NOT NULL
           OR NEW.confirmed_by_id IS NOT NULL
           OR NEW.confirmed_by_name IS NOT NULL THEN
            RAISE EXCEPTION 'new EOM Terms delivery must start pending';
        END IF;
        IF NEW.kind = 'invitation' THEN
            IF NEW.acceptance_id IS NOT NULL OR NOT EXISTS (
                SELECT 1
                  FROM eom_terms_invitations AS invitation
                 WHERE invitation.id = NEW.invitation_id
                   AND invitation.recipient_email = NEW.recipient_email
            ) THEN
                RAISE EXCEPTION
                    'invitation delivery must match its EOM Terms invitation';
            END IF;
        ELSIF NEW.kind = 'executed_copy' THEN
            IF NEW.acceptance_id IS NULL OR NOT EXISTS (
                SELECT 1
                  FROM eom_terms_acceptances AS acceptance
                 WHERE acceptance.id = NEW.acceptance_id
                   AND acceptance.invitation_id = NEW.invitation_id
                   AND acceptance.recipient_email = NEW.recipient_email
            ) THEN
                RAISE EXCEPTION
                    'executed-copy delivery must match its EOM Terms acceptance';
            END IF;
        ELSE
            RAISE EXCEPTION 'EOM Terms delivery kind is invalid';
        END IF;
        RETURN NEW;
    END IF;
    IF ROW(
        NEW.id, NEW.kind, NEW.invitation_id, NEW.acceptance_id,
        NEW.recipient_email, NEW.subject, NEW.body, NEW.body_hash,
        NEW.created_at
    ) IS DISTINCT FROM ROW(
        OLD.id, OLD.kind, OLD.invitation_id, OLD.acceptance_id,
        OLD.recipient_email, OLD.subject, OLD.body, OLD.body_hash,
        OLD.created_at
    ) THEN
        RAISE EXCEPTION 'EOM Terms delivery payload is immutable';
    END IF;
    IF OLD.status = 'pending' AND NEW.status = 'sending' THEN
        IF NEW.sent_at IS NOT NULL
           OR NEW.resend_message_id IS NOT NULL
           OR NEW.transport_idempotent_replay IS NOT NULL
           OR NEW.confirmed_by_id IS NOT NULL
           OR NEW.confirmed_by_name IS NOT NULL THEN
            RAISE EXCEPTION 'EOM Terms delivery claim contains sent evidence';
        END IF;
        IF NEW.kind = 'invitation' AND NOT EXISTS (
            SELECT 1
              FROM eom_terms_invitations AS invitation
              JOIN contacts AS contact ON contact.id = invitation.contact_id
              JOIN eom_terms_versions AS invited
                ON invited.id = invitation.version_id
               AND invited.status = 'published'
             WHERE invitation.id = NEW.invitation_id
               AND invitation.revoked_at IS NULL
               AND clock_timestamp() <= invitation.expires_at
               AND contact.business_context_id = 'effingham_maids'
               AND contact.contact_type = 'customer'
               AND contact.status = 'active'
               AND contact.customer_type = invitation.audience
               AND btrim(contact.full_name) = invitation.customer_name
               AND lower(btrim(contact.email)) = invitation.recipient_email
               AND NOT EXISTS (
                   SELECT 1
                     FROM eom_terms_versions AS later
                    WHERE later.status = 'published'
                      AND later.material_change
                      AND later.publication_order > invited.publication_order
               )
               AND NOT EXISTS (
                   SELECT 1
                     FROM eom_terms_acceptances AS acceptance
                    WHERE acceptance.invitation_id = invitation.id
               )
        ) THEN
            RAISE EXCEPTION 'EOM Terms invitation delivery is no longer valid';
        END IF;
        IF NEW.kind = 'executed_copy' AND NOT EXISTS (
            SELECT 1
              FROM eom_terms_acceptances AS acceptance
              JOIN eom_terms_invitations AS invitation
                ON invitation.id = acceptance.invitation_id
               AND invitation.contact_id = acceptance.contact_id
               AND invitation.audience = acceptance.audience
               AND invitation.recipient_email = acceptance.recipient_email
              JOIN contacts AS contact ON contact.id = acceptance.contact_id
             WHERE acceptance.id = NEW.acceptance_id
               AND acceptance.invitation_id = NEW.invitation_id
               AND acceptance.recipient_email = NEW.recipient_email
               AND contact.business_context_id = 'effingham_maids'
               AND contact.contact_type = 'customer'
               AND contact.status = 'active'
               AND contact.customer_type = acceptance.audience
               AND btrim(contact.full_name) = invitation.customer_name
               AND lower(btrim(contact.email)) = acceptance.recipient_email
        ) THEN
            RAISE EXCEPTION
                'EOM Terms executed-copy delivery is no longer valid';
        END IF;
        NEW.claimed_at := clock_timestamp();
        RETURN NEW;
    END IF;
    IF OLD.status = 'sending' AND NEW.status = 'sent' THEN
        IF NEW.claimed_at IS DISTINCT FROM OLD.claimed_at THEN
            RAISE EXCEPTION 'EOM Terms delivery claim time is immutable';
        END IF;
        IF ((
            (length(btrim(NEW.resend_message_id)) > 0
                AND NEW.transport_idempotent_replay IS NOT NULL)
            OR
            (NEW.resend_message_id IS NULL
                AND NEW.transport_idempotent_replay IS TRUE)
        )
            AND NEW.confirmed_by_id IS NULL
            AND NEW.confirmed_by_name IS NULL
        ) OR (
            NEW.resend_message_id IS NULL
            AND NEW.transport_idempotent_replay IS NULL
            AND NEW.confirmed_by_id > 0
            AND length(btrim(NEW.confirmed_by_name)) > 0
        ) THEN
            NEW.sent_at := clock_timestamp();
            RETURN NEW;
        END IF;
        RAISE EXCEPTION 'EOM Terms sent delivery requires transport or actor evidence';
    END IF;
    RAISE EXCEPTION 'EOM Terms delivery transition is invalid';
END;
$$;

CREATE TRIGGER trg_validate_eom_terms_invitation
    BEFORE INSERT ON eom_terms_invitations
    FOR EACH ROW EXECUTE FUNCTION validate_eom_terms_invitation();

CREATE TRIGGER trg_protect_eom_terms_invitation
    BEFORE UPDATE OR DELETE ON eom_terms_invitations
    FOR EACH ROW EXECUTE FUNCTION protect_eom_terms_invitation();

CREATE TRIGGER trg_protect_eom_terms_invitation_truncate
    BEFORE TRUNCATE ON eom_terms_invitations
    FOR EACH STATEMENT EXECUTE FUNCTION protect_eom_terms_invitation();

CREATE TRIGGER trg_validate_eom_terms_acceptance
    BEFORE INSERT ON eom_terms_acceptances
    FOR EACH ROW EXECUTE FUNCTION validate_eom_terms_acceptance();

CREATE TRIGGER trg_protect_eom_terms_acceptance
    BEFORE UPDATE OR DELETE ON eom_terms_acceptances
    FOR EACH ROW EXECUTE FUNCTION protect_eom_terms_acceptance();

CREATE TRIGGER trg_protect_eom_terms_acceptance_truncate
    BEFORE TRUNCATE ON eom_terms_acceptances
    FOR EACH STATEMENT EXECUTE FUNCTION protect_eom_terms_acceptance();

CREATE TRIGGER trg_protect_eom_terms_delivery
    BEFORE INSERT OR UPDATE OR DELETE ON eom_terms_deliveries
    FOR EACH ROW EXECUTE FUNCTION protect_eom_terms_delivery();

CREATE TRIGGER trg_protect_eom_terms_delivery_truncate
    BEFORE TRUNCATE ON eom_terms_deliveries
    FOR EACH STATEMENT EXECUTE FUNCTION protect_eom_terms_delivery();

CREATE INDEX idx_eom_terms_invitations_contact_issued
    ON eom_terms_invitations (contact_id, issued_at DESC, id DESC);

CREATE INDEX idx_eom_terms_acceptances_contact_accepted
    ON eom_terms_acceptances (contact_id, accepted_at DESC, id DESC);

CREATE INDEX idx_eom_terms_deliveries_status
    ON eom_terms_deliveries (status, created_at, id);

DO $$
DECLARE
    schema_name TEXT := current_schema();
    table_name TEXT;
    function_name TEXT;
    grantee_name TEXT;
BEGIN
    FOREACH function_name IN ARRAY ARRAY[
        'protect_eom_terms_version',
        'validate_eom_terms_invitation',
        'protect_eom_terms_invitation',
        'validate_eom_terms_acceptance',
        'protect_eom_terms_acceptance',
        'protect_eom_terms_delivery'
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
        'eom_terms_invitations',
        'eom_terms_acceptances',
        'eom_terms_deliveries'
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
        'validate_eom_terms_invitation',
        'protect_eom_terms_invitation',
        'validate_eom_terms_acceptance',
        'protect_eom_terms_acceptance',
        'protect_eom_terms_delivery'
    ]
    LOOP
        EXECUTE format(
            'REVOKE ALL PRIVILEGES ON FUNCTION %I.%I() FROM PUBLIC',
            schema_name,
            function_name
        );
        FOR grantee_name IN
            SELECT DISTINCT grantee_role.rolname
              FROM pg_proc AS guarded_function
              JOIN pg_namespace AS namespace
                ON namespace.oid = guarded_function.pronamespace
              CROSS JOIN LATERAL aclexplode(
                  COALESCE(guarded_function.proacl, ARRAY[]::aclitem[])
              ) AS acl
              JOIN pg_roles AS grantee_role ON grantee_role.oid = acl.grantee
             WHERE namespace.nspname = schema_name
               AND guarded_function.proname = function_name
               AND guarded_function.pronargs = 0
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
        'GRANT SELECT ON TABLE %I.eom_terms_invitations TO atlas',
        schema_name
    );
    EXECUTE format(
        'GRANT INSERT (id, request_key, contact_id, version_id, audience, '
        || 'locale, customer_name, recipient_email, public_base_url, '
        || 'signing_key_fingerprint, issued_by_id, issued_by_name) '
        || 'ON TABLE %I.eom_terms_invitations TO atlas',
        schema_name
    );
    EXECUTE format(
        'GRANT UPDATE (revoked_at, revoked_by_id, revoked_by_name) '
        || 'ON TABLE %I.eom_terms_invitations TO atlas',
        schema_name
    );
    EXECUTE format(
        'GRANT SELECT ON TABLE %I.eom_terms_acceptances TO atlas',
        schema_name
    );
    EXECUTE format(
        'GRANT INSERT (id, invitation_id, signer_name, terms_accepted, '
        || 'additional_work_accepted, client_ip) '
        || 'ON TABLE %I.eom_terms_acceptances TO atlas',
        schema_name
    );
    EXECUTE format(
        'GRANT SELECT ON TABLE %I.eom_terms_deliveries TO atlas',
        schema_name
    );
    EXECUTE format(
        'GRANT INSERT (id, kind, invitation_id, acceptance_id, '
        || 'recipient_email, subject, body, body_hash) '
        || 'ON TABLE %I.eom_terms_deliveries TO atlas',
        schema_name
    );
    EXECUTE format(
        'GRANT UPDATE (status, claimed_at, sent_at, resend_message_id, '
        || 'transport_idempotent_replay, confirmed_by_id, confirmed_by_name) '
        || 'ON TABLE %I.eom_terms_deliveries TO atlas',
        schema_name
    );
END;
$$;

COMMENT ON TABLE eom_terms_invitations IS
    'Customer-bound expiring EOM Terms invitation metadata; raw bearer tokens are never stored.';
COMMENT ON TABLE eom_terms_acceptances IS
    'Immutable evidence that one active EOM customer accepted one published Terms version and both required acknowledgements.';
COMMENT ON TABLE eom_terms_deliveries IS
    'Immutable invitation/executed-copy payloads with one-way transport and actor-reconciliation evidence.';
COMMENT ON COLUMN eom_terms_versions.publication_order IS
    'Database-assigned monotonic release order used instead of wall-clock chronology.';
