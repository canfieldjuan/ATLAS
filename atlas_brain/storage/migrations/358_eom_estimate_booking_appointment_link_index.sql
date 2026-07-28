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

DROP INDEX CONCURRENTLY IF EXISTS uq_appointments_eom_estimate_booking_operation;

CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_appointments_eom_estimate_booking_operation
    ON appointments (eom_estimate_booking_operation_id)
    WHERE eom_estimate_booking_operation_id IS NOT NULL;
