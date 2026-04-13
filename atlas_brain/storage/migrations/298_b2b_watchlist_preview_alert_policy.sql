-- Persist per-watchlist preview-backed account alert policy overrides.

ALTER TABLE b2b_watchlist_views
    ADD COLUMN IF NOT EXISTS preview_alerts_enabled BOOLEAN,
    ADD COLUMN IF NOT EXISTS preview_alert_min_confidence NUMERIC(4, 2),
    ADD COLUMN IF NOT EXISTS preview_alert_require_budget_authority BOOLEAN;

UPDATE b2b_watchlist_views
SET preview_alerts_enabled = TRUE
WHERE preview_alerts_enabled IS NULL;

ALTER TABLE b2b_watchlist_views
    ALTER COLUMN preview_alerts_enabled SET DEFAULT TRUE;

ALTER TABLE b2b_watchlist_views
    ALTER COLUMN preview_alerts_enabled SET NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'chk_b2b_watchlist_views_preview_alert_min_confidence'
    ) THEN
        ALTER TABLE b2b_watchlist_views
            ADD CONSTRAINT chk_b2b_watchlist_views_preview_alert_min_confidence
            CHECK (
                preview_alert_min_confidence IS NULL
                OR (
                    preview_alert_min_confidence >= 0
                    AND preview_alert_min_confidence <= 1
                )
            );
    END IF;
END
$$;
