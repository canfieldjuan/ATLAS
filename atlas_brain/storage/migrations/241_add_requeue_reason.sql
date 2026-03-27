-- Migration 241: Add requeue_reason to b2b_reviews
-- Allows tracking WHY a review was re-queued (parser upgrade, model drift, etc)

ALTER TABLE b2b_reviews 
    ADD COLUMN IF NOT EXISTS requeue_reason TEXT;

COMMIT;
