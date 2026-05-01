-- Persist opportunity score component breakdown for campaign diagnostics.
-- Migration 146
--
-- When a campaign underperforms, the breakdown lets you diagnose which
-- scoring component (urgency, role, stage, seats, context) was wrong.

ALTER TABLE b2b_campaigns
    ADD COLUMN IF NOT EXISTS score_components JSONB;
