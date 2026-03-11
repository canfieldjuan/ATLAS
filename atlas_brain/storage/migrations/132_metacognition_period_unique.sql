-- Fix: metacognition flush uses ON CONFLICT (period_start) but no unique
-- constraint existed. Add one so upsert works correctly.

CREATE UNIQUE INDEX IF NOT EXISTS idx_rm_period_unique
    ON reasoning_metacognition(period_start);
