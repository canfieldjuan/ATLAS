-- Track how many times an alert event has been re-opened after being resolved.
-- Helps users distinguish recurring vs persistent signals.

ALTER TABLE b2b_watchlist_alert_events
    ADD COLUMN IF NOT EXISTS reopen_count INT NOT NULL DEFAULT 0;
