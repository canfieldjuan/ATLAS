-- Structured operating facts for an existing customer appointment.
-- This records the schedule/owner/price; it does not generate future visits.

ALTER TABLE appointments
    ADD COLUMN IF NOT EXISTS recurrence_interval SMALLINT,
    ADD COLUMN IF NOT EXISTS recurrence_unit VARCHAR(16),
    ADD COLUMN IF NOT EXISTS assigned_cleaner VARCHAR(128),
    ADD COLUMN IF NOT EXISTS per_visit_price NUMERIC(12,2);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'chk_appointments_recurrence_pair'
          AND conrelid = 'appointments'::regclass
    ) THEN
        ALTER TABLE appointments
            ADD CONSTRAINT chk_appointments_recurrence_pair
            CHECK (
                (recurrence_interval IS NULL)
                = (recurrence_unit IS NULL)
            );
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'chk_appointments_recurrence_value'
          AND conrelid = 'appointments'::regclass
    ) THEN
        ALTER TABLE appointments
            ADD CONSTRAINT chk_appointments_recurrence_value
            CHECK (
                recurrence_interval IS NULL
                OR (
                    recurrence_interval BETWEEN 1 AND 365
                    AND recurrence_unit IN ('day', 'week', 'month')
                )
            );
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'chk_appointments_assigned_cleaner'
          AND conrelid = 'appointments'::regclass
    ) THEN
        ALTER TABLE appointments
            ADD CONSTRAINT chk_appointments_assigned_cleaner
            CHECK (
                assigned_cleaner IS NULL
                OR btrim(assigned_cleaner) <> ''
            );
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'chk_appointments_per_visit_price'
          AND conrelid = 'appointments'::regclass
    ) THEN
        ALTER TABLE appointments
            ADD CONSTRAINT chk_appointments_per_visit_price
            CHECK (per_visit_price IS NULL OR per_visit_price >= 0);
    END IF;
END
$$;

COMMENT ON COLUMN appointments.recurrence_interval IS
    'Positive every-N interval paired with recurrence_unit; NULL for one-off or unknown';
COMMENT ON COLUMN appointments.recurrence_unit IS
    'day, week, or month; paired with recurrence_interval';
COMMENT ON COLUMN appointments.assigned_cleaner IS
    'Operator-facing cleaner assignment label';
COMMENT ON COLUMN appointments.per_visit_price IS
    'Exact price snapshot for this visit, independent of later service-rate changes';
