-- EOM estimate-booking appointment-link column, FK, and uniqueness.
--
-- Companion to migration 356, which creates the durable operation table. Add
-- the nullable Atlas-owned operation link to the established appointments table
-- in this non-transactional, statement-by-statement migration so table locks do
-- not remain held through the larger operation-table setup batch. The FK is
-- added NOT VALID first and then validated with PostgreSQL's low-lock
-- validation path. Build this unique partial index concurrently so deploys do
-- not take a regular index-build lock on live appointment writes.
--
-- Why drop first: a canceled/failed CREATE INDEX CONCURRENTLY can leave this
-- relation present but invalid. Without the drop, a retry would satisfy
-- IF NOT EXISTS, skip the rebuild, record the migration, and leave startup
-- readiness permanently failing on the invalid/not-ready index.

ALTER TABLE appointments
    ADD COLUMN IF NOT EXISTS eom_estimate_booking_operation_id UUID;

COMMENT ON COLUMN appointments.eom_estimate_booking_operation_id IS
    'EOM lead estimate-booking command that created this appointment; NULL for all legacy appointments.';

ALTER TABLE appointments
    DROP CONSTRAINT IF EXISTS appointments_eom_estimate_booking_operation_id_fkey;

ALTER TABLE appointments
    ADD CONSTRAINT appointments_eom_estimate_booking_operation_id_fkey
        FOREIGN KEY (eom_estimate_booking_operation_id)
        REFERENCES eom_lead_estimate_booking_operations(id)
        ON DELETE SET NULL
        NOT VALID;

ALTER TABLE appointments
    VALIDATE CONSTRAINT appointments_eom_estimate_booking_operation_id_fkey;

CREATE OR REPLACE FUNCTION prevent_eom_estimate_booking_appointment_mutation()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        IF OLD.eom_estimate_booking_operation_id IS NOT NULL THEN
            RAISE EXCEPTION
                'EOM estimate booking appointment % is managed by booking operation %',
                OLD.id,
                OLD.eom_estimate_booking_operation_id
                USING ERRCODE = 'check_violation';
        END IF;
        RETURN OLD;
    END IF;

    IF OLD.eom_estimate_booking_operation_id IS NOT NULL AND (
        OLD.eom_estimate_booking_operation_id IS DISTINCT FROM NEW.eom_estimate_booking_operation_id
        OR OLD.start_time IS DISTINCT FROM NEW.start_time
        OR OLD.end_time IS DISTINCT FROM NEW.end_time
        OR OLD.duration_minutes IS DISTINCT FROM NEW.duration_minutes
        OR OLD.service_type IS DISTINCT FROM NEW.service_type
        OR OLD.notes IS DISTINCT FROM NEW.notes
        OR OLD.customer_name IS DISTINCT FROM NEW.customer_name
        OR OLD.customer_phone IS DISTINCT FROM NEW.customer_phone
        OR OLD.customer_email IS DISTINCT FROM NEW.customer_email
        OR OLD.customer_address IS DISTINCT FROM NEW.customer_address
        OR OLD.calendar_event_id IS DISTINCT FROM NEW.calendar_event_id
        OR OLD.business_context_id IS DISTINCT FROM NEW.business_context_id
        OR OLD.contact_id IS DISTINCT FROM NEW.contact_id
        OR OLD.status IS DISTINCT FROM NEW.status
        OR OLD.cancelled_at IS DISTINCT FROM NEW.cancelled_at
        OR OLD.cancellation_reason IS DISTINCT FROM NEW.cancellation_reason
    ) THEN
        RAISE EXCEPTION
            'EOM estimate booking appointment % is managed by booking operation %',
            OLD.id,
            OLD.eom_estimate_booking_operation_id
            USING ERRCODE = 'check_violation';
    END IF;

    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_prevent_eom_estimate_booking_appointment_mutation
    ON appointments;

CREATE TRIGGER trg_prevent_eom_estimate_booking_appointment_mutation
    BEFORE UPDATE OF eom_estimate_booking_operation_id, start_time, end_time,
        duration_minutes, service_type, notes, customer_name, customer_phone,
        customer_email, customer_address, calendar_event_id, business_context_id,
        contact_id, status, cancelled_at, cancellation_reason
        OR DELETE ON appointments
    FOR EACH ROW
    EXECUTE FUNCTION prevent_eom_estimate_booking_appointment_mutation();

DROP INDEX CONCURRENTLY IF EXISTS uq_appointments_eom_estimate_booking_operation;

CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_appointments_eom_estimate_booking_operation
    ON appointments (eom_estimate_booking_operation_id)
    WHERE eom_estimate_booking_operation_id IS NOT NULL;
