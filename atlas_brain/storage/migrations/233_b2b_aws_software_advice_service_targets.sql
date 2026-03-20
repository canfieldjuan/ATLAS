WITH defaults AS (
    SELECT
        COALESCE(
            (
                SELECT priority
                FROM b2b_scrape_targets
                WHERE source = 'software_advice'
                  AND enabled = true
                GROUP BY priority
                ORDER BY count(*) DESC, priority DESC
                LIMIT 1
            ),
            15
        ) AS priority,
        COALESCE(
            (
                SELECT max_pages
                FROM b2b_scrape_targets
                WHERE source = 'software_advice'
                  AND enabled = true
                GROUP BY max_pages
                ORDER BY count(*) DESC, max_pages DESC
                LIMIT 1
            ),
            15
        ) AS max_pages,
        COALESCE(
            (
                SELECT scrape_interval_hours
                FROM b2b_scrape_targets
                WHERE source = 'software_advice'
                  AND enabled = true
                GROUP BY scrape_interval_hours
                ORDER BY count(*) DESC, scrape_interval_hours DESC
                LIMIT 1
            ),
            168
        ) AS scrape_interval_hours
),
targets AS (
    SELECT *
    FROM (
        VALUES
            (
                'software_advice',
                'Amazon Web Services',
                'Amazon CloudWatch',
                'log-analysis/amazon-cloudwatch-profile',
                'Cloud Infrastructure',
                '{"curated_service_target": true, "curated_service_family": "aws"}'::jsonb
            ),
            (
                'software_advice',
                'Amazon Web Services',
                'Amazon Lightsail',
                'virtual-machine/amazon-lightsail-profile',
                'Cloud Infrastructure',
                '{"curated_service_target": true, "curated_service_family": "aws"}'::jsonb
            ),
            (
                'software_advice',
                'Amazon Web Services',
                'Amazon API Gateway',
                'api-management/amazon-api-gateway-profile',
                'Cloud Infrastructure',
                '{"curated_service_target": true, "curated_service_family": "aws"}'::jsonb
            ),
            (
                'software_advice',
                'Amazon Web Services',
                'Amazon RDS',
                'database-management-systems/amazon-rds-profile',
                'Cloud Infrastructure',
                '{"curated_service_target": true, "curated_service_family": "aws"}'::jsonb
            ),
            (
                'software_advice',
                'Amazon Web Services',
                'Amazon Simple Notification Service (SNS)',
                'push-notifications/amazon-simple-notification-service-sns-profile',
                'Cloud Infrastructure',
                '{"curated_service_target": true, "curated_service_family": "aws"}'::jsonb
            )
    ) AS t(
        source,
        vendor_name,
        product_name,
        product_slug,
        product_category,
        metadata
    )
)
INSERT INTO b2b_scrape_targets (
    source,
    vendor_name,
    product_name,
    product_slug,
    product_category,
    max_pages,
    enabled,
    priority,
    scrape_interval_hours,
    scrape_mode,
    metadata
)
SELECT
    t.source,
    t.vendor_name,
    t.product_name,
    t.product_slug,
    t.product_category,
    d.max_pages,
    true,
    d.priority,
    d.scrape_interval_hours,
    'incremental',
    t.metadata
FROM targets AS t
CROSS JOIN defaults AS d
ON CONFLICT (source, product_slug, scrape_mode) DO NOTHING;
