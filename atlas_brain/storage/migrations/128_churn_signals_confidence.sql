-- Add confidence_score column to b2b_churn_signals
-- This was referenced by the upsert logic but never added to the table,
-- causing all churn signal upserts to silently fail.
ALTER TABLE b2b_churn_signals ADD COLUMN IF NOT EXISTS confidence_score FLOAT DEFAULT 0;
