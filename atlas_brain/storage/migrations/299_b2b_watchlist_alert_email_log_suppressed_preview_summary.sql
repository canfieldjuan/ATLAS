ALTER TABLE b2b_watchlist_alert_email_log
    ADD COLUMN IF NOT EXISTS suppressed_preview_summary JSONB;
