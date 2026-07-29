-- Durable office estimate-booking commands for EOM leads.
--
-- The operation row is the idempotency authority. It owns one deterministic
-- Google event ID and, once projection succeeds, one Atlas appointment.

CREATE TABLE IF NOT EXISTS eom_lead_estimate_booking_operations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    contact_id UUID NOT NULL REFERENCES contacts(id) ON DELETE RESTRICT,
    idempotency_key VARCHAR(128) NOT NULL,
    request_fingerprint VARCHAR(64) NOT NULL,
    actor VARCHAR(128) NOT NULL,
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ NOT NULL,
    service_type VARCHAR(128) NOT NULL,
    location TEXT,
    notes TEXT NOT NULL DEFAULT '',
    contact_snapshot JSONB NOT NULL,
    calendar_id VARCHAR(512) NOT NULL,
    calendar_event_id VARCHAR(256) NOT NULL,
    appointment_id UUID REFERENCES appointments(id) ON DELETE RESTRICT,
    status VARCHAR(32) NOT NULL DEFAULT 'pending',
    projection_started_at TIMESTAMPTZ,
    projection_token UUID,
    reclaimed_projection BOOLEAN NOT NULL DEFAULT FALSE,
    last_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_eom_lead_estimate_booking_operation_status
        CHECK (status IN ('pending', 'projecting', 'calendar_failed', 'calendar_rejected', 'completed')),
    CONSTRAINT uq_eom_lead_estimate_booking_contact_key
        UNIQUE (contact_id, idempotency_key),
    CONSTRAINT uq_eom_lead_estimate_booking_calendar_event
        UNIQUE (calendar_id, calendar_event_id)
);

ALTER TABLE eom_lead_estimate_booking_operations
    DROP CONSTRAINT IF EXISTS uq_eom_lead_estimate_booking_contact;

ALTER TABLE eom_lead_estimate_booking_operations
    DROP CONSTRAINT IF EXISTS chk_eom_lead_estimate_booking_operation_status;

ALTER TABLE eom_lead_estimate_booking_operations
    ADD COLUMN IF NOT EXISTS projection_token UUID;

ALTER TABLE eom_lead_estimate_booking_operations
    ADD COLUMN IF NOT EXISTS reclaimed_projection BOOLEAN NOT NULL DEFAULT FALSE;

ALTER TABLE eom_lead_estimate_booking_operations
    ADD CONSTRAINT chk_eom_lead_estimate_booking_operation_status
        CHECK (status IN ('pending', 'projecting', 'calendar_failed', 'calendar_rejected', 'completed'));

CREATE UNIQUE INDEX IF NOT EXISTS uq_eom_lead_estimate_booking_contact_active
    ON eom_lead_estimate_booking_operations (contact_id)
    WHERE status <> 'calendar_rejected';

CREATE INDEX IF NOT EXISTS idx_eom_lead_estimate_booking_contact_created
    ON eom_lead_estimate_booking_operations (contact_id, created_at DESC);

COMMENT ON TABLE eom_lead_estimate_booking_operations IS
    'Idempotent office estimate-booking commands for active EOM lead/new contacts.';

CREATE OR REPLACE FUNCTION prevent_eom_pending_estimate_booking_contact_state_mutation()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $$
DECLARE
    pending_operation_id UUID;
BEGIN
    IF OLD.business_context_id IS DISTINCT FROM 'effingham_maids'
       OR OLD.contact_type IS DISTINCT FROM 'lead' THEN
        IF TG_OP = 'DELETE' THEN
            RETURN OLD;
        END IF;
        RETURN NEW;
    END IF;

    EXECUTE format(
        'SELECT operation.id
           FROM %I.eom_lead_estimate_booking_operations AS operation
          WHERE operation.contact_id = $1
            AND operation.appointment_id IS NULL
            AND operation.status IN (''pending'', ''projecting'', ''calendar_failed'')
          LIMIT 1',
        TG_TABLE_SCHEMA
    )
    INTO pending_operation_id
    USING OLD.id;

    IF pending_operation_id IS NULL THEN
        IF TG_OP = 'DELETE' THEN
            RETURN OLD;
        END IF;
        RETURN NEW;
    END IF;

    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION
            'EOM estimate booking contact state is locked while booking operation % is pending',
            pending_operation_id
            USING ERRCODE = 'check_violation';
    END IF;

    IF NEW.business_context_id IS NOT DISTINCT FROM OLD.business_context_id
       AND NEW.contact_type IS NOT DISTINCT FROM OLD.contact_type
       AND NEW.status IS NOT DISTINCT FROM OLD.status
       AND OLD.lead_stage IS NOT DISTINCT FROM 'new'
       AND NEW.lead_stage IS NOT DISTINCT FROM 'estimate_booked' THEN
        RETURN NEW;
    END IF;

    IF NEW.business_context_id IS DISTINCT FROM OLD.business_context_id
       OR NEW.contact_type IS DISTINCT FROM OLD.contact_type
       OR NEW.lead_stage IS DISTINCT FROM OLD.lead_stage
       OR NEW.status IS DISTINCT FROM OLD.status THEN
        RAISE EXCEPTION
            'EOM estimate booking contact state is locked while booking operation % is pending',
            pending_operation_id
            USING ERRCODE = 'check_violation';
    END IF;

    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_prevent_eom_pending_estimate_booking_contact_state_mutation
    ON contacts;
CREATE TRIGGER trg_prevent_eom_pending_estimate_booking_contact_state_mutation
    BEFORE UPDATE OF business_context_id, contact_type, lead_stage, status
        OR DELETE ON contacts
    FOR EACH ROW
    EXECUTE FUNCTION prevent_eom_pending_estimate_booking_contact_state_mutation();

-- Migration 354 gives the browser-facing NocoDB role ordinary CRM-table write
-- capability on appointments. This operation link is not ordinary appointment
-- data: it is the Atlas-owned proof that one durable booking operation created
-- one appointment. Keep NocoDB's existing appointment edit surface by replacing
-- table-level INSERT/UPDATE with column-level grants that omit only this link.
DO $$
DECLARE
    schema_name TEXT := current_schema();
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'atlas_nocodb') THEN
        EXECUTE format(
            'REVOKE INSERT, UPDATE ON TABLE %I.appointments FROM atlas_nocodb',
            schema_name
        );
        EXECUTE format(
            'GRANT INSERT (id, start_time, end_time, duration_minutes, '
            || 'service_type, notes, customer_name, customer_phone, '
            || 'customer_email, customer_address, calendar_event_id, '
            || 'business_context_id, call_id, contact_id, status, created_at, '
            || 'updated_at, cancelled_at, cancellation_reason, confirmation_sent, '
            || 'confirmation_sent_at, reminder_sent, reminder_sent_at, '
            || 'metadata, recurrence_interval, recurrence_unit, assigned_cleaner, '
            || 'per_visit_price) ON TABLE %I.appointments TO atlas_nocodb',
            schema_name
        );
        EXECUTE format(
            'GRANT UPDATE (start_time, end_time, duration_minutes, '
            || 'service_type, notes, customer_name, customer_phone, '
            || 'customer_email, customer_address, calendar_event_id, '
            || 'business_context_id, call_id, contact_id, status, created_at, '
            || 'updated_at, cancelled_at, cancellation_reason, confirmation_sent, '
            || 'confirmation_sent_at, reminder_sent, reminder_sent_at, '
            || 'metadata, recurrence_interval, recurrence_unit, assigned_cleaner, '
            || 'per_visit_price) ON TABLE %I.appointments TO atlas_nocodb',
            schema_name
        );
    END IF;
END;
$$;
