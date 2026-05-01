-- Stratified reasoning engine: semantic cache + metacognition tables.
-- Phase 1 of the stratified reasoning architecture (WS0).

CREATE TABLE IF NOT EXISTS reasoning_semantic_cache (
    id                       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pattern_sig              TEXT NOT NULL,
    pattern_class            TEXT NOT NULL,
    vendor_name              TEXT,
    product_category         TEXT,
    conclusion               JSONB NOT NULL,
    confidence               FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    reasoning_steps          JSONB NOT NULL DEFAULT '[]',
    boundary_conditions      JSONB NOT NULL DEFAULT '{}',
    falsification_conditions JSONB DEFAULT '[]',
    uncertainty_sources      TEXT[] DEFAULT '{}',
    decay_half_life_days     INT DEFAULT 90,
    conclusion_type          TEXT,
    evidence_hash            TEXT,
    created_at               TIMESTAMPTZ DEFAULT NOW(),
    last_validated_at        TIMESTAMPTZ DEFAULT NOW(),
    validation_count         INT DEFAULT 1,
    invalidated_at           TIMESTAMPTZ,
    UNIQUE(pattern_sig)
);

CREATE INDEX IF NOT EXISTS idx_rsc_pattern
    ON reasoning_semantic_cache(pattern_sig) WHERE invalidated_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_rsc_class
    ON reasoning_semantic_cache(pattern_class) WHERE invalidated_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_rsc_vendor
    ON reasoning_semantic_cache(vendor_name) WHERE invalidated_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_rsc_confidence
    ON reasoning_semantic_cache(confidence DESC) WHERE invalidated_at IS NULL;

CREATE TABLE IF NOT EXISTS reasoning_metacognition (
    id                           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    period_start                 TIMESTAMPTZ NOT NULL,
    period_end                   TIMESTAMPTZ NOT NULL,
    total_queries                INT DEFAULT 0,
    recall_hits                  INT DEFAULT 0,
    reconstitute_hits            INT DEFAULT 0,
    full_reasons                 INT DEFAULT 0,
    surprise_escalations         INT DEFAULT 0,
    exploration_samples          INT DEFAULT 0,
    total_tokens_saved           INT DEFAULT 0,
    total_tokens_spent           INT DEFAULT 0,
    conclusion_type_distribution JSONB DEFAULT '{}',
    cache_quality_score          FLOAT,
    created_at                   TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rm_period
    ON reasoning_metacognition(period_start DESC);
