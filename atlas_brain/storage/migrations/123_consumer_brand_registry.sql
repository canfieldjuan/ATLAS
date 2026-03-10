-- Migration 123: Consumer brand registry + provenance fields
--
-- Ports B2B vendor registry pattern (migration 095) to consumer brands.
-- Enables canonical brand name resolution, alias matching, and fuzzy search.
-- pg_trgm extension already enabled by migration 114.

-- Canonical brand registry
CREATE TABLE IF NOT EXISTS consumer_brand_registry (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    canonical_name  TEXT NOT NULL UNIQUE,
    aliases         JSONB NOT NULL DEFAULT '[]',
    metadata        JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- GIN index on aliases for containment queries
CREATE INDEX IF NOT EXISTS idx_consumer_brand_aliases
    ON consumer_brand_registry USING GIN (aliases);

-- Case-insensitive exact lookup
CREATE INDEX IF NOT EXISTS idx_consumer_brand_canonical_lower
    ON consumer_brand_registry (LOWER(canonical_name));

-- Trigram fuzzy search (pg_trgm already enabled)
CREATE INDEX IF NOT EXISTS idx_consumer_brand_canonical_trgm
    ON consumer_brand_registry USING GIST (canonical_name gist_trgm_ops);

-- Provenance columns on brand_intelligence
ALTER TABLE brand_intelligence
    ADD COLUMN IF NOT EXISTS source_review_count INTEGER,
    ADD COLUMN IF NOT EXISTS source_distribution JSONB DEFAULT '{}'::jsonb;

-- Seed registry from existing product_metadata brands
INSERT INTO consumer_brand_registry (canonical_name)
SELECT DISTINCT brand
FROM product_metadata
WHERE brand IS NOT NULL AND brand != ''
ON CONFLICT (canonical_name) DO NOTHING;
