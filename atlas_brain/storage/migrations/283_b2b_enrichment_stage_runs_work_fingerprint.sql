ALTER TABLE b2b_enrichment_stage_runs
    ADD COLUMN IF NOT EXISTS work_fingerprint TEXT;

UPDATE b2b_enrichment_stage_runs
SET work_fingerprint = request_fingerprint
WHERE work_fingerprint IS NULL;

ALTER TABLE b2b_enrichment_stage_runs
    ALTER COLUMN work_fingerprint SET NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_b2b_enrichment_stage_runs_review_stage_work
    ON b2b_enrichment_stage_runs (review_id, stage_id, work_fingerprint);
