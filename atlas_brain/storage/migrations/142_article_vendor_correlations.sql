-- Article-to-vendor archetype correlations.
-- Links news_articles to B2B vendors when temporal or entity-based
-- alignment with archetype shifts is detected.

CREATE TABLE IF NOT EXISTS b2b_article_correlations (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id          UUID NOT NULL REFERENCES news_articles(id) ON DELETE CASCADE,
    vendor_name         TEXT NOT NULL,
    correlation_type    TEXT NOT NULL,          -- entity_match, temporal_shift, soram_alignment
    archetype           TEXT,                   -- vendor archetype at time of correlation
    archetype_confidence FLOAT,
    change_event_id     UUID REFERENCES b2b_change_events(id) ON DELETE SET NULL,
    soram_alignment     JSONB,                  -- which SORAM channels align with archetype
    relevance_score     FLOAT DEFAULT 0,        -- 0-1 composite score
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (article_id, vendor_name)
);

CREATE INDEX IF NOT EXISTS idx_bac_vendor
    ON b2b_article_correlations (vendor_name, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_bac_article
    ON b2b_article_correlations (article_id);
CREATE INDEX IF NOT EXISTS idx_bac_archetype
    ON b2b_article_correlations (archetype, created_at DESC)
    WHERE archetype IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_bac_change_event
    ON b2b_article_correlations (change_event_id)
    WHERE change_event_id IS NOT NULL;
