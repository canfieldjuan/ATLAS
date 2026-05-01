-- 073: Fix b2b_campaigns schema for sequence-generated campaigns
--
-- 1. Add metadata JSONB column (stores cta, angle_reasoning from progression LLM)
-- 2. Make vendor_name nullable (sequence follow-ups don't have a vendor)

ALTER TABLE b2b_campaigns
    ADD COLUMN IF NOT EXISTS metadata jsonb DEFAULT '{}'::jsonb;

ALTER TABLE b2b_campaigns
    ALTER COLUMN vendor_name DROP NOT NULL;
