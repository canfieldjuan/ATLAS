-- Migration 271: Persist watchlist alert thresholds with saved views

ALTER TABLE b2b_watchlist_views
    ADD COLUMN IF NOT EXISTS vendor_alert_threshold NUMERIC(4, 2),
    ADD COLUMN IF NOT EXISTS account_alert_threshold NUMERIC(4, 2),
    ADD COLUMN IF NOT EXISTS stale_days_threshold INTEGER;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'chk_b2b_watchlist_views_vendor_alert_threshold'
    ) THEN
        ALTER TABLE b2b_watchlist_views
            ADD CONSTRAINT chk_b2b_watchlist_views_vendor_alert_threshold
            CHECK (
                vendor_alert_threshold IS NULL
                OR (vendor_alert_threshold >= 0 AND vendor_alert_threshold <= 10)
            );
    END IF;
END
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'chk_b2b_watchlist_views_account_alert_threshold'
    ) THEN
        ALTER TABLE b2b_watchlist_views
            ADD CONSTRAINT chk_b2b_watchlist_views_account_alert_threshold
            CHECK (
                account_alert_threshold IS NULL
                OR (account_alert_threshold >= 0 AND account_alert_threshold <= 10)
            );
    END IF;
END
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'chk_b2b_watchlist_views_stale_days_threshold'
    ) THEN
        ALTER TABLE b2b_watchlist_views
            ADD CONSTRAINT chk_b2b_watchlist_views_stale_days_threshold
            CHECK (
                stale_days_threshold IS NULL
                OR (stale_days_threshold >= 0 AND stale_days_threshold <= 365)
            );
    END IF;
END
$$;
