-- Extract intent confidence from action_plan JSONB into indexed column.
-- Migration 149
--
-- LLM returns 0.0-1.0 confidence with each intent classification.
-- Currently buried in action_plan->>'confidence'. Surfacing as a column
-- enables classifier performance analysis and threshold tuning over time.

ALTER TABLE processed_emails
    ADD COLUMN IF NOT EXISTS intent_confidence REAL;

CREATE INDEX IF NOT EXISTS idx_processed_emails_intent_confidence
    ON processed_emails (intent_confidence)
    WHERE intent_confidence IS NOT NULL;
