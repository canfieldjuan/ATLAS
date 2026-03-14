-- Partial indexes for autonomous task query performance
-- b2b_product_profiles: 8 fetchers all filter on enrichment_status='enriched' + enriched_at window
-- competitive_intelligence: 9 fetchers filter on deep_enrichment_status='enriched'

-- b2b_reviews: cover the enriched + time-window pattern used by all product profile fetchers
CREATE INDEX IF NOT EXISTS idx_b2b_reviews_enriched_vendor
    ON b2b_reviews (vendor_name, enriched_at)
    WHERE enrichment_status = 'enriched';

-- product_reviews: cover JOIN to product_metadata for deep-enriched reviews (competitive_intelligence)
CREATE INDEX IF NOT EXISTS idx_product_reviews_deep_enriched_asin
    ON product_reviews (asin)
    WHERE deep_enrichment_status = 'enriched';
