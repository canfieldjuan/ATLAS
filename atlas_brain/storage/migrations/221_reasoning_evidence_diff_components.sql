-- Persist explainable drift component scores for reconstitute vs full_reason.

ALTER TABLE reasoning_evidence_diffs
    ADD COLUMN IF NOT EXISTS component_scores JSONB NOT NULL DEFAULT '{}'::jsonb;
