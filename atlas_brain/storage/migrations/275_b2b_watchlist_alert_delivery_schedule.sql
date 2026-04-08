-- Migration 275: Persist watchlist alert email schedules and durable delivery claims.

ALTER TABLE b2b_watchlist_views
    ADD COLUMN IF NOT EXISTS alert_email_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS alert_delivery_frequency TEXT,
    ADD COLUMN IF NOT EXISTS next_alert_delivery_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS last_alert_delivery_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS last_alert_delivery_status TEXT,
    ADD COLUMN IF NOT EXISTS last_alert_delivery_summary TEXT;

UPDATE b2b_watchlist_views
SET alert_delivery_frequency = COALESCE(NULLIF(BTRIM(alert_delivery_frequency), ''), 'daily')
WHERE alert_delivery_frequency IS NULL
   OR BTRIM(alert_delivery_frequency) = '';

ALTER TABLE b2b_watchlist_views
    ALTER COLUMN alert_delivery_frequency SET DEFAULT 'daily';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'chk_b2b_watchlist_views_alert_delivery_frequency'
    ) THEN
        ALTER TABLE b2b_watchlist_views
            ADD CONSTRAINT chk_b2b_watchlist_views_alert_delivery_frequency
            CHECK (
                alert_delivery_frequency IS NULL
                OR alert_delivery_frequency IN ('daily', 'weekly')
            );
    END IF;
END
$$;

ALTER TABLE b2b_watchlist_alert_email_log
    ADD COLUMN IF NOT EXISTS scheduled_for TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS delivery_frequency TEXT,
    ADD COLUMN IF NOT EXISTS delivery_mode TEXT NOT NULL DEFAULT 'live';

UPDATE b2b_watchlist_alert_email_log
SET delivery_mode = COALESCE(NULLIF(BTRIM(delivery_mode), ''), 'live')
WHERE delivery_mode IS NULL
   OR BTRIM(delivery_mode) = '';

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'chk_b2b_watchlist_alert_email_log_status'
    ) THEN
        ALTER TABLE b2b_watchlist_alert_email_log
            DROP CONSTRAINT chk_b2b_watchlist_alert_email_log_status;
    END IF;
END
$$;

ALTER TABLE b2b_watchlist_alert_email_log
    ADD CONSTRAINT chk_b2b_watchlist_alert_email_log_status
        CHECK (status IN ('processing', 'sent', 'partial', 'failed', 'no_events', 'skipped'));

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'chk_b2b_watchlist_alert_email_log_delivery_frequency'
    ) THEN
        ALTER TABLE b2b_watchlist_alert_email_log
            ADD CONSTRAINT chk_b2b_watchlist_alert_email_log_delivery_frequency
            CHECK (
                delivery_frequency IS NULL
                OR delivery_frequency IN ('daily', 'weekly')
            );
    END IF;
END
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'chk_b2b_watchlist_alert_email_log_delivery_mode'
    ) THEN
        ALTER TABLE b2b_watchlist_alert_email_log
            ADD CONSTRAINT chk_b2b_watchlist_alert_email_log_delivery_mode
            CHECK (delivery_mode IN ('live', 'scheduled'));
    END IF;
END
$$;

CREATE UNIQUE INDEX IF NOT EXISTS idx_b2b_watchlist_alert_email_log_schedule_claim
    ON b2b_watchlist_alert_email_log (watchlist_view_id, scheduled_for, delivery_mode)
    WHERE scheduled_for IS NOT NULL;
