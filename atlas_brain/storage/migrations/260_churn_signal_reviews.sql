-- Add signal_reviews column to b2b_churn_signals.
-- signal_reviews counts only reviews with urgency > 0 OR intent_to_leave
-- OR competitors_mentioned, excluding zero-signal enrichments that dilute
-- churn density metrics.
ALTER TABLE b2b_churn_signals
    ADD COLUMN IF NOT EXISTS signal_reviews integer;
