-- Recreate consumer materialized views to use promoted columns instead of JSONB.
-- Migration 154
--
-- mv_brand_summary and mv_asin_summary previously extracted would_repurchase,
-- sentiment_trajectory, and replacement_behavior from deep_extraction JSONB.
-- Now reference the indexed columns from migration 153.

DROP MATERIALIZED VIEW IF EXISTS mv_brand_summary CASCADE;
DROP MATERIALIZED VIEW IF EXISTS mv_asin_summary CASCADE;

-- Rebuild mv_brand_summary with column references
-- brand_products is a CTE aggregating product_metadata per brand
CREATE MATERIALIZED VIEW mv_brand_summary AS
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
    bp.pm_avg_rating,
    bp.total_ratings,
    AVG(pr.pain_score) FILTER (WHERE pr.rating <= 3) AS avg_complaint_score,
    AVG(pr.pain_score) FILTER (WHERE pr.rating > 3) AS avg_praise_score,
    COUNT(*) FILTER (WHERE pr.rating <= 3) AS complaint_count,
    COUNT(*) FILTER (WHERE pr.rating > 3) AS praise_count,
    COUNT(*) FILTER (WHERE pr.deep_extraction IS NOT NULL AND pr.deep_extraction != '{}'::jsonb) AS deep_count,
    COUNT(*) FILTER (WHERE pr.would_repurchase IS TRUE) AS repurchase_yes,
    COUNT(*) FILTER (WHERE pr.would_repurchase IS NOT NULL) AS repurchase_total,
    COUNT(*) FILTER (WHERE (pr.deep_extraction->'safety_flag'->>'flagged')::boolean IS TRUE) AS safety_count,
    COUNT(*) FILTER (WHERE pr.sentiment_trajectory IN ('always_positive','improved','mixed_then_positive')) AS trajectory_positive,
    COUNT(*) FILTER (WHERE pr.sentiment_trajectory IN ('always_negative','degraded','mixed_then_negative')) AS trajectory_negative,
    COUNT(*) FILTER (WHERE pr.replacement_behavior IN ('kept_using','repurchased','replaced_same')) AS retention_positive,
    COUNT(*) FILTER (WHERE pr.replacement_behavior IN ('switched_to','switched_brand','returned','avoided')) AS retention_negative
FROM product_reviews pr
JOIN product_metadata pm ON pm.asin = pr.asin
JOIN brand_products bp ON bp.brand = pm.brand
WHERE pr.enrichment_status = 'enriched'
  AND pm.brand IS NOT NULL AND pm.brand != ''
GROUP BY pm.brand, bp.pm_avg_rating, bp.total_ratings;

CREATE UNIQUE INDEX mv_brand_summary_brand_idx ON mv_brand_summary (brand);


-- Rebuild mv_asin_summary with column references
CREATE MATERIALIZED VIEW mv_asin_summary AS
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
    COUNT(*) FILTER (WHERE pr.would_repurchase IS TRUE) AS repurchase_yes,
    COUNT(*) FILTER (WHERE pr.would_repurchase IS FALSE) AS repurchase_no,
    COUNT(*) FILTER (WHERE (pr.deep_extraction->'safety_flag'->>'flagged')::boolean IS TRUE) AS safety_count,
    mode() WITHIN GROUP (ORDER BY pr.root_cause) AS top_root_cause
FROM product_reviews pr
LEFT JOIN product_metadata pm ON pm.asin = pr.asin
WHERE pr.enrichment_status = 'enriched'
GROUP BY pr.asin, pm.brand, pm.title;

CREATE UNIQUE INDEX mv_asin_summary_asin_idx ON mv_asin_summary (asin);
