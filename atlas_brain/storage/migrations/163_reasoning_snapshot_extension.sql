-- Migration 163: Reasoning Snapshot Extension
-- Adds firmographic and advanced sentiment metrics to vendor snapshots
-- for deeper reasoning archetypes (scale_up_stumble, pivot_abandonment).

ALTER TABLE b2b_vendor_snapshots
ADD COLUMN IF NOT EXISTS employee_growth_rate NUMERIC(5,4), -- e.g. 0.25 = 25%
ADD COLUMN IF NOT EXISTS support_sentiment NUMERIC(3,2),    -- -1.0 to 1.0
ADD COLUMN IF NOT EXISTS new_feature_velocity NUMERIC(5,2), -- arbitrary scale
ADD COLUMN IF NOT EXISTS legacy_support_score NUMERIC(3,2); -- -1.0 to 1.0
