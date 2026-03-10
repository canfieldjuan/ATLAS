-- Migration 124: Consumer canonical entities & confidence scoring
--
-- Ports B2B displacement edges (migration 099) and confidence scoring
-- (migration 102) to the consumer product review pipeline.
-- Creates product_displacement_edges table for brand-to-brand competitive
-- flows and adds confidence_score to brand_intelligence and product_pain_points.

-- product_displacement_edges (append-only time-series)
CREATE TABLE IF NOT EXISTS product_displacement_edges (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    from_brand            TEXT NOT NULL,
    to_brand              TEXT NOT NULL,
    direction             TEXT NOT NULL DEFAULT 'compared',
    mention_count         INT NOT NULL DEFAULT 0,
    signal_strength       TEXT CHECK (signal_strength IN ('strong', 'moderate', 'emerging')),
    avg_rating            NUMERIC(3,2),
    category_distribution JSONB DEFAULT '{}'::jsonb,
    sample_review_ids     UUID[] DEFAULT '{}',
    confidence_score      NUMERIC(3,2) DEFAULT 0.00,
    computed_date         DATE NOT NULL DEFAULT CURRENT_DATE,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (from_brand, to_brand, direction, computed_date)
);

CREATE INDEX IF NOT EXISTS idx_pde_from_brand
    ON product_displacement_edges (from_brand);
CREATE INDEX IF NOT EXISTS idx_pde_to_brand
    ON product_displacement_edges (to_brand);
CREATE INDEX IF NOT EXISTS idx_pde_computed_date
    ON product_displacement_edges (computed_date DESC);
CREATE INDEX IF NOT EXISTS idx_pde_confidence
    ON product_displacement_edges (confidence_score DESC);

-- Add confidence_score to brand_intelligence
ALTER TABLE brand_intelligence
    ADD COLUMN IF NOT EXISTS confidence_score NUMERIC(3,2) DEFAULT 0.00;

-- Add confidence_score to product_pain_points
ALTER TABLE product_pain_points
    ADD COLUMN IF NOT EXISTS confidence_score NUMERIC(3,2) DEFAULT 0.00;
