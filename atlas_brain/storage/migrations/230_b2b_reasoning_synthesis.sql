-- Reasoning synthesis: per-vendor structured reasoning output that feeds
-- deterministic battle card builders, challenger briefs, and all downstream
-- products. Produced by a dedicated LLM reasoning call that consumes all 6
-- pool layers and outputs 5 structured sections (causal_narrative,
-- segment_playbook, timing_intelligence, competitive_reframes, migration_proof).
-- Cached by evidence hash — only re-runs when pool data changes.

CREATE TABLE IF NOT EXISTS b2b_reasoning_synthesis (
    vendor_name          text        NOT NULL,
    as_of_date           date        NOT NULL,
    analysis_window_days int         NOT NULL,
    schema_version       text        NOT NULL,
    evidence_hash        text        NOT NULL,
    synthesis            jsonb       NOT NULL,
    tokens_used          int         NOT NULL DEFAULT 0,
    llm_model            text,
    created_at           timestamptz NOT NULL DEFAULT NOW(),
    PRIMARY KEY (vendor_name, as_of_date, analysis_window_days, schema_version)
);

CREATE INDEX IF NOT EXISTS idx_b2b_reasoning_synthesis_vendor
    ON b2b_reasoning_synthesis (vendor_name);
CREATE INDEX IF NOT EXISTS idx_b2b_reasoning_synthesis_date
    ON b2b_reasoning_synthesis (as_of_date DESC);
CREATE INDEX IF NOT EXISTS idx_b2b_reasoning_synthesis_hash
    ON b2b_reasoning_synthesis (evidence_hash);
