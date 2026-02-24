-- news_articles: enrichment columns for full-text + SORAM classification
ALTER TABLE news_articles
    ADD COLUMN IF NOT EXISTS content TEXT,
    ADD COLUMN IF NOT EXISTS soram_channels JSONB,
    ADD COLUMN IF NOT EXISTS linguistic_indicators JSONB,
    ADD COLUMN IF NOT EXISTS entities_detected TEXT[] NOT NULL DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS enrichment_status TEXT NOT NULL DEFAULT 'pending',
    ADD COLUMN IF NOT EXISTS enrichment_attempts INT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS enriched_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_news_articles_enrichment
    ON news_articles(enrichment_status) WHERE enrichment_status = 'pending';

-- Per-entity pressure tracking (one row per entity, upserted daily)
CREATE TABLE IF NOT EXISTS entity_pressure_baselines (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_name         TEXT NOT NULL,
    entity_type         TEXT NOT NULL DEFAULT 'company',
    pressure_score      NUMERIC(4,2) NOT NULL DEFAULT 0.0,
    sentiment_drift     NUMERIC(6,4) NOT NULL DEFAULT 0.0,
    narrative_frequency INT NOT NULL DEFAULT 0,
    soram_breakdown     JSONB NOT NULL DEFAULT '{}',
    linguistic_signals  JSONB NOT NULL DEFAULT '{}',
    watchlist_id        UUID REFERENCES data_watchlist(id),
    last_computed_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (entity_name, entity_type)
);
CREATE INDEX IF NOT EXISTS idx_pressure_baselines_entity
    ON entity_pressure_baselines(entity_name, entity_type);

-- reasoning_journal: add pressure readings column
ALTER TABLE reasoning_journal
    ADD COLUMN IF NOT EXISTS pressure_readings JSONB NOT NULL DEFAULT '[]';
