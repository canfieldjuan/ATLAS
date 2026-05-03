-- Migration 259: llm_cache_savings (PR-A3, owned by extraction).
--
-- One row per cache hit. Records the input/output tokens and cost the
-- underlying LLM call would have incurred without the cache so the
-- product can answer "$ saved by cache last month".
--
-- Owned by extracted_llm_infrastructure -- NOT synced from atlas_brain.
-- This table is new in the extracted package and is intentionally not
-- back-ported to atlas_brain (Atlas tracks cache-hit counters in memory
-- inside enrichment_row_runner; the extracted package adds persistence
-- as a Phase 1 owned addition).
--
-- Attribution is free-form JSONB so consumers can add dimensions
-- without schema migrations. Index dimensions you care about with
-- PostgreSQL generated columns or partial expression indexes.

CREATE TABLE IF NOT EXISTS llm_cache_savings (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cache_key           TEXT NOT NULL,
    namespace           TEXT NOT NULL,
    provider            TEXT NOT NULL,
    model               TEXT NOT NULL,
    saved_input_tokens  INT NOT NULL DEFAULT 0,
    saved_output_tokens INT NOT NULL DEFAULT 0,
    saved_cost_usd      NUMERIC(12, 6) NOT NULL DEFAULT 0,
    attribution         JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    hit_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Primary query pattern: rollups by date range
CREATE INDEX IF NOT EXISTS idx_llm_cache_savings_hit_at
    ON llm_cache_savings (hit_at DESC);

-- Secondary: per-namespace rollups within a date range
CREATE INDEX IF NOT EXISTS idx_llm_cache_savings_namespace_hit_at
    ON llm_cache_savings (namespace, hit_at DESC);
