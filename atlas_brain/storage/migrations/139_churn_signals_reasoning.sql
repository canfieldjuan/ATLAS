-- Add stratified reasoning fields to churn signals
ALTER TABLE b2b_churn_signals ADD COLUMN IF NOT EXISTS archetype TEXT;
ALTER TABLE b2b_churn_signals ADD COLUMN IF NOT EXISTS archetype_confidence FLOAT;
ALTER TABLE b2b_churn_signals ADD COLUMN IF NOT EXISTS reasoning_mode TEXT;
ALTER TABLE b2b_churn_signals ADD COLUMN IF NOT EXISTS falsification_conditions JSONB;
