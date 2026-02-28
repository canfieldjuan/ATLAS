-- 062: Expand b2b_churn_signals with aggregated columns for new enrichment fields.
-- Additive only -- no existing columns modified.

ALTER TABLE b2b_churn_signals
    ADD COLUMN IF NOT EXISTS top_use_cases           JSONB NOT NULL DEFAULT '[]',
    ADD COLUMN IF NOT EXISTS top_integration_stacks  JSONB NOT NULL DEFAULT '[]',
    ADD COLUMN IF NOT EXISTS budget_signal_summary   JSONB NOT NULL DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS sentiment_distribution  JSONB NOT NULL DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS buyer_authority_summary  JSONB NOT NULL DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS timeline_summary         JSONB NOT NULL DEFAULT '[]';
