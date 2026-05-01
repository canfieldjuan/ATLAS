-- Displacement velocity: rate of change in competitive flow intensity.
-- Migration 147
--
-- velocity_7d / velocity_30d = change in mention_count vs the same pair's
-- value N days ago.  Positive = accelerating, negative = decelerating.
-- Computed at write time by the churn intelligence task.

ALTER TABLE b2b_displacement_edges
    ADD COLUMN IF NOT EXISTS velocity_7d  INT,
    ADD COLUMN IF NOT EXISTS velocity_30d INT;
