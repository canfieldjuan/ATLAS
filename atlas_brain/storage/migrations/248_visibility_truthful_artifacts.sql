-- Truthful pipeline visibility follow-up:
--   * primary artifact rows carry real lifecycle state
--   * synthesis validation results are queryable per rule
--   * dedup decisions are first-class rows
--   * operator review actions are immutable audit rows

ALTER TABLE blog_posts
    ADD COLUMN IF NOT EXISTS latest_run_id TEXT,
    ADD COLUMN IF NOT EXISTS latest_attempt_no INT,
    ADD COLUMN IF NOT EXISTS latest_failure_step TEXT,
    ADD COLUMN IF NOT EXISTS latest_error_code TEXT,
    ADD COLUMN IF NOT EXISTS latest_error_summary TEXT,
    ADD COLUMN IF NOT EXISTS quality_score INT,
    ADD COLUMN IF NOT EXISTS quality_threshold INT,
    ADD COLUMN IF NOT EXISTS blocker_count INT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS warning_count INT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS rejected_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS rejection_reason TEXT;

ALTER TABLE b2b_intelligence
    ADD COLUMN IF NOT EXISTS latest_run_id TEXT,
    ADD COLUMN IF NOT EXISTS latest_attempt_no INT,
    ADD COLUMN IF NOT EXISTS latest_failure_step TEXT,
    ADD COLUMN IF NOT EXISTS latest_error_code TEXT,
    ADD COLUMN IF NOT EXISTS latest_error_summary TEXT,
    ADD COLUMN IF NOT EXISTS blocker_count INT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS warning_count INT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS quality_score INT,
    ADD COLUMN IF NOT EXISTS quality_threshold INT;

CREATE TABLE IF NOT EXISTS synthesis_validation_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vendor_name TEXT NOT NULL,
    as_of_date DATE NOT NULL,
    analysis_window_days INT NOT NULL,
    schema_version TEXT NOT NULL,
    run_id TEXT,
    attempt_no INT NOT NULL DEFAULT 1,
    rule_code TEXT NOT NULL,
    severity TEXT NOT NULL
        CHECK (severity IN ('info', 'warning', 'error', 'critical')),
    passed BOOLEAN NOT NULL DEFAULT FALSE,
    summary TEXT NOT NULL,
    field_path TEXT,
    detail JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_synthesis_validation_vendor
    ON synthesis_validation_results (vendor_name, as_of_date DESC, attempt_no DESC);
CREATE INDEX IF NOT EXISTS idx_synthesis_validation_rule
    ON synthesis_validation_results (rule_code, severity, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_synthesis_validation_run
    ON synthesis_validation_results (run_id) WHERE run_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS dedup_decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id TEXT,
    stage TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    survivor_entity_id TEXT,
    discarded_entity_id TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    comparison_metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    actor_type TEXT NOT NULL DEFAULT 'system',
    actor_id TEXT,
    decided_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_dedup_decisions_stage
    ON dedup_decisions (stage, decided_at DESC);
CREATE INDEX IF NOT EXISTS idx_dedup_decisions_discarded
    ON dedup_decisions (entity_type, discarded_entity_id);
CREATE INDEX IF NOT EXISTS idx_dedup_decisions_survivor
    ON dedup_decisions (entity_type, survivor_entity_id)
    WHERE survivor_entity_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS pipeline_review_actions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    review_id UUID NOT NULL REFERENCES pipeline_visibility_reviews(id) ON DELETE CASCADE,
    fingerprint TEXT NOT NULL,
    target_entity_type TEXT NOT NULL,
    target_entity_id TEXT NOT NULL,
    action TEXT NOT NULL,
    note TEXT,
    actor_id TEXT,
    actor_type TEXT NOT NULL DEFAULT 'human',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pipeline_review_actions_review
    ON pipeline_review_actions (review_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_pipeline_review_actions_target
    ON pipeline_review_actions (target_entity_type, target_entity_id, created_at DESC);
