-- atlas: atomic-bookkeeping
-- Immutable, tenant-scoped first-clean completion evidence for EOM.
--
-- `first_clean_booked` is Calendar booking evidence, not service completion.
-- These rows admit only an authenticated tracker report for an already
-- canonicalized active residential customer. They deliberately create no
-- customer email, token, payment, or Stripe side effect.
--
-- Rollback: stop the completion route and its tracker consumer first, then
-- retain these append-only receipts as audit evidence. Do not delete a receipt
-- merely because an application deployment is rolled back.

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

COMMENT ON TABLE eom_first_clean_completion_operation_receipts IS
    'Globally unique operation-key ownership for EOM first-clean completion reports; retries cannot move a completion to another customer or service.';
COMMENT ON TABLE eom_first_clean_completion_receipts IS
    'Immutable EOM evidence that one active residential canonical customer completed a first service; it is not a booking, email, payment, or Stripe authorization.';
