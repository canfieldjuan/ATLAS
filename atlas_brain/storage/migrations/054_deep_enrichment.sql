-- Deep enrichment pipeline: second-pass extraction of 10+ rich fields per review
-- Runs independently from basic enrichment via separate state machine

ALTER TABLE product_reviews
    ADD COLUMN IF NOT EXISTS deep_extraction          JSONB,
    ADD COLUMN IF NOT EXISTS deep_enrichment_status   TEXT NOT NULL DEFAULT 'pending',
    ADD COLUMN IF NOT EXISTS deep_enrichment_attempts INT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS deep_enriched_at         TIMESTAMPTZ;

-- Exclude non-enriched rows from the deep pipeline
UPDATE product_reviews
SET deep_enrichment_status = 'not_applicable'
WHERE enrichment_status != 'enriched';

CREATE INDEX IF NOT EXISTS idx_product_reviews_deep_pending
    ON product_reviews(deep_enrichment_status)
    WHERE deep_enrichment_status = 'pending';

CREATE INDEX IF NOT EXISTS idx_product_reviews_deep_extraction
    ON product_reviews USING GIN (deep_extraction)
    WHERE deep_extraction IS NOT NULL;
