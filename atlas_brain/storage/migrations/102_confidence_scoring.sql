-- Migration 102: Add confidence_score to use cases, integrations, and buyer profiles
-- Mirrors the confidence_score column already on b2b_vendor_pain_points (migration 100)

ALTER TABLE b2b_vendor_use_cases ADD COLUMN IF NOT EXISTS confidence_score NUMERIC(3,2) DEFAULT 0;
ALTER TABLE b2b_vendor_integrations ADD COLUMN IF NOT EXISTS confidence_score NUMERIC(3,2) DEFAULT 0;
ALTER TABLE b2b_vendor_buyer_profiles ADD COLUMN IF NOT EXISTS confidence_score NUMERIC(3,2) DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_bvuc_confidence ON b2b_vendor_use_cases (confidence_score DESC);
CREATE INDEX IF NOT EXISTS idx_bvi_confidence ON b2b_vendor_integrations (confidence_score DESC);
CREATE INDEX IF NOT EXISTS idx_bvbp_confidence ON b2b_vendor_buyer_profiles (confidence_score DESC);
