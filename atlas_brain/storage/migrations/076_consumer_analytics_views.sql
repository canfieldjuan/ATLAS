-- 076: Materialized views for consumer analytics dashboard
-- Replaces on-the-fly JSONB scans with pre-aggregated rollups.
-- Refreshed every 6 hours by consumer_analytics_refresh task.

-- Brand-level metrics from deep-extracted reviews
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_brand_summary AS
WITH brand_products AS (
    SELECT m.brand,
           AVG(m.average_rating) AS pm_avg_rating,
           SUM(m.rating_number) AS total_ratings
    FROM product_metadata m
    WHERE m.brand IS NOT NULL AND m.brand != ''
    GROUP BY m.brand
)
SELECT
    pm.brand,
    COUNT(DISTINCT pm.asin) AS product_count,
    COUNT(pr.id) AS review_count,
    bp.pm_avg_rating,
    bp.total_ratings,
    AVG(pr.pain_score) FILTER (WHERE pr.rating <= 3) AS avg_complaint_score,
    AVG(pr.pain_score) FILTER (WHERE pr.rating > 3) AS avg_praise_score,
    COUNT(*) FILTER (WHERE pr.rating <= 3) AS complaint_count,
    COUNT(*) FILTER (WHERE pr.rating > 3) AS praise_count,
    COUNT(*) FILTER (WHERE pr.deep_extraction IS NOT NULL AND pr.deep_extraction != '{}'::jsonb) AS deep_count,
    COUNT(*) FILTER (WHERE (pr.deep_extraction->>'would_repurchase')::boolean IS TRUE) AS repurchase_yes,
    COUNT(*) FILTER (WHERE pr.deep_extraction->>'would_repurchase' IN ('true','false')) AS repurchase_total,
    COUNT(*) FILTER (WHERE (pr.deep_extraction->'safety_flag'->>'flagged')::boolean IS TRUE) AS safety_count,
    COUNT(*) FILTER (WHERE pr.deep_extraction->>'sentiment_trajectory' IN ('always_positive','improved','mixed_then_positive')) AS trajectory_positive,
    COUNT(*) FILTER (WHERE pr.deep_extraction->>'sentiment_trajectory' IN ('always_negative','degraded','mixed_then_negative')) AS trajectory_negative,
    COUNT(*) FILTER (WHERE pr.deep_extraction->>'replacement_behavior' IN ('kept_using','repurchased','replaced_same')) AS retention_positive,
    COUNT(*) FILTER (WHERE pr.deep_extraction->>'replacement_behavior' IN ('switched_to','switched_brand','returned','avoided')) AS retention_negative
FROM product_reviews pr
JOIN product_metadata pm ON pm.asin = pr.asin
JOIN brand_products bp ON bp.brand = pm.brand
WHERE pr.enrichment_status = 'enriched'
  AND pm.brand IS NOT NULL AND pm.brand != ''
GROUP BY pm.brand, bp.pm_avg_rating, bp.total_ratings;

CREATE UNIQUE INDEX IF NOT EXISTS mv_brand_summary_brand_idx ON mv_brand_summary (brand);


-- Category-level rollup (excludes NULL categories for unique index safety)
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_category_summary AS
SELECT
    pr.source_category AS category,
    COUNT(*) AS total_reviews,
    COUNT(*) FILTER (WHERE pr.enrichment_status = 'enriched') AS enriched_count,
    COUNT(*) FILTER (WHERE pr.deep_enrichment_status = 'enriched') AS deep_enriched_count,
    AVG(pr.rating) AS avg_rating,
    AVG(pr.pain_score) FILTER (WHERE pr.rating <= 3) AS avg_pain_score,
    COUNT(*) FILTER (WHERE pr.severity = 'critical') AS critical_count,
    COUNT(*) FILTER (WHERE pr.severity = 'major') AS major_count,
    COUNT(*) FILTER (WHERE pr.severity = 'minor') AS minor_count,
    COUNT(*) FILTER (WHERE (pr.deep_extraction->'safety_flag'->>'flagged')::boolean IS TRUE) AS safety_count
FROM product_reviews pr
WHERE pr.source_category IS NOT NULL
GROUP BY pr.source_category;

CREATE UNIQUE INDEX IF NOT EXISTS mv_category_summary_category_idx ON mv_category_summary (category);


-- Per-ASIN rollup for product-level analytics
-- GROUP BY asin + metadata columns (1:1 via LEFT JOIN on PK); source_category
-- is aggregated via mode() to avoid duplicate rows when an ASIN has reviews in
-- multiple categories.
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_asin_summary AS
SELECT
    pr.asin,
    pm.brand,
    pm.title,
    mode() WITHIN GROUP (ORDER BY pr.source_category) AS category,
    COUNT(*) AS total_reviews,
    AVG(pr.rating) AS avg_rating,
    AVG(pr.pain_score) AS avg_pain_score,
    COUNT(*) FILTER (WHERE pr.rating <= 3) AS complaint_count,
    COUNT(*) FILTER (WHERE pr.rating > 3) AS praise_count,
    COUNT(*) FILTER (WHERE pr.severity = 'critical') AS critical_count,
    COUNT(*) FILTER (WHERE (pr.deep_extraction->>'would_repurchase')::boolean IS TRUE) AS repurchase_yes,
    COUNT(*) FILTER (WHERE (pr.deep_extraction->>'would_repurchase')::boolean IS FALSE) AS repurchase_no,
    COUNT(*) FILTER (WHERE (pr.deep_extraction->'safety_flag'->>'flagged')::boolean IS TRUE) AS safety_count,
    mode() WITHIN GROUP (ORDER BY pr.root_cause) AS top_root_cause
FROM product_reviews pr
LEFT JOIN product_metadata pm ON pm.asin = pr.asin
WHERE pr.enrichment_status = 'enriched'
GROUP BY pr.asin, pm.brand, pm.title;

CREATE UNIQUE INDEX IF NOT EXISTS mv_asin_summary_asin_idx ON mv_asin_summary (asin);
