ALTER TABLE b2b_report_subscription_delivery_log
    ADD COLUMN IF NOT EXISTS delivery_mode TEXT;

UPDATE b2b_report_subscription_delivery_log
SET delivery_mode = CASE
    WHEN status = 'dry_run' THEN 'dry_run'
    ELSE 'live'
END
WHERE delivery_mode IS NULL
   OR delivery_mode NOT IN ('live', 'dry_run');

ALTER TABLE b2b_report_subscription_delivery_log
    ALTER COLUMN delivery_mode SET DEFAULT 'live';

ALTER TABLE b2b_report_subscription_delivery_log
    ALTER COLUMN delivery_mode SET NOT NULL;

ALTER TABLE b2b_report_subscription_delivery_log
    DROP CONSTRAINT IF EXISTS b2b_report_subscription_delivery_log_delivery_mode_check;

ALTER TABLE b2b_report_subscription_delivery_log
    ADD CONSTRAINT b2b_report_subscription_delivery_log_delivery_mode_check
    CHECK (delivery_mode IN ('live', 'dry_run'));

DO $$
DECLARE
    legacy_constraint_name TEXT;
BEGIN
    SELECT con.conname
    INTO legacy_constraint_name
    FROM pg_constraint con
    JOIN pg_class rel ON rel.oid = con.conrelid
    JOIN pg_namespace nsp ON nsp.oid = rel.relnamespace
    WHERE nsp.nspname = 'public'
      AND rel.relname = 'b2b_report_subscription_delivery_log'
      AND con.contype = 'u'
      AND pg_get_constraintdef(con.oid) = 'UNIQUE (subscription_id, scheduled_for)';

    IF legacy_constraint_name IS NOT NULL THEN
        EXECUTE format(
            'ALTER TABLE public.b2b_report_subscription_delivery_log DROP CONSTRAINT %I',
            legacy_constraint_name
        );
    END IF;
END
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint con
        JOIN pg_class rel ON rel.oid = con.conrelid
        JOIN pg_namespace nsp ON nsp.oid = rel.relnamespace
        WHERE nsp.nspname = 'public'
          AND rel.relname = 'b2b_report_subscription_delivery_log'
          AND con.conname = 'uq_b2b_report_subscription_delivery_log_schedule_mode'
    ) THEN
        ALTER TABLE public.b2b_report_subscription_delivery_log
            ADD CONSTRAINT uq_b2b_report_subscription_delivery_log_schedule_mode
            UNIQUE (subscription_id, scheduled_for, delivery_mode);
    END IF;
END
$$;
