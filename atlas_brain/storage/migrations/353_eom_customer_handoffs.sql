-- Durable, opaque link between one approved EOM lead and tracker records.
-- Operational service/rate/schedule facts remain in the EOM time tracker.

CREATE TABLE IF NOT EXISTS eom_customer_handoffs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    contact_id UUID NOT NULL UNIQUE REFERENCES contacts(id) ON DELETE RESTRICT,
    approval_key VARCHAR(128) NOT NULL UNIQUE,
    tracker_customer_id BIGINT NOT NULL UNIQUE CHECK (tracker_customer_id > 0),
    tracker_site_id BIGINT NOT NULL UNIQUE CHECK (tracker_site_id > 0),
    approved_by_employee_id BIGINT NOT NULL CHECK (approved_by_employee_id > 0),
    approved_by_name VARCHAR(128) NOT NULL,
    finalized_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_eom_customer_handoffs_finalized
    ON eom_customer_handoffs (finalized_at DESC);

CREATE OR REPLACE FUNCTION require_eom_customer_handoff_finalization()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM contacts AS contact
        JOIN eom_lead_lifecycle_events AS lifecycle
          ON lifecycle.contact_id = contact.id
        WHERE contact.id = NEW.contact_id
          AND contact.business_context_id = 'effingham_maids'
          AND contact.contact_type = 'customer'
          AND contact.lead_stage IS NULL
          AND contact.status = 'active'
          AND lifecycle.event_type = 'customer_approved'
          AND lifecycle.source = 'eom_office'
          AND lifecycle.operation_key = NEW.approval_key
          AND lifecycle.actor = (
              'employee:' || NEW.approved_by_employee_id::text || ':' || NEW.approved_by_name
          )
          AND lifecycle.metadata @> jsonb_build_object(
              'tracker_customer_id', NEW.tracker_customer_id,
              'tracker_site_id', NEW.tracker_site_id,
              'approved_by_employee_id', NEW.approved_by_employee_id
          )
    ) THEN
        RAISE EXCEPTION
            'eom_customer_handoffs requires the matching customer transition and lifecycle evidence';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_require_eom_customer_handoff_finalization
    ON eom_customer_handoffs;
CREATE TRIGGER trg_require_eom_customer_handoff_finalization
    BEFORE INSERT ON eom_customer_handoffs
    FOR EACH ROW
    EXECUTE FUNCTION require_eom_customer_handoff_finalization();

CREATE OR REPLACE FUNCTION prevent_eom_customer_handoff_mutation()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'eom_customer_handoffs is immutable';
END;
$$;

DROP TRIGGER IF EXISTS trg_prevent_eom_customer_handoff_mutation
    ON eom_customer_handoffs;
CREATE TRIGGER trg_prevent_eom_customer_handoff_mutation
    BEFORE UPDATE OR DELETE ON eom_customer_handoffs
    FOR EACH ROW
    EXECUTE FUNCTION prevent_eom_customer_handoff_mutation();

DROP TRIGGER IF EXISTS trg_prevent_eom_customer_handoff_truncate
    ON eom_customer_handoffs;
CREATE TRIGGER trg_prevent_eom_customer_handoff_truncate
    BEFORE TRUNCATE ON eom_customer_handoffs
    FOR EACH STATEMENT
    EXECUTE FUNCTION prevent_eom_customer_handoff_mutation();

COMMENT ON TABLE eom_customer_handoffs IS
    'One immutable tracker Customer/Site link for each Atlas-approved EOM lead.';
