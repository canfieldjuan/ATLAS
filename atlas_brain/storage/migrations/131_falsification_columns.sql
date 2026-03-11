-- Add columns needed by the falsification watcher (WS0F).
-- direction: trend direction for change events (rising/falling/stable)
-- overall_sentiment: sentiment label for enriched reviews (positive/negative/mixed/neutral)

ALTER TABLE b2b_change_events
    ADD COLUMN IF NOT EXISTS direction TEXT;

ALTER TABLE b2b_reviews
    ADD COLUMN IF NOT EXISTS overall_sentiment TEXT;
