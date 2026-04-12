-- Backfill canonical activity references for historical webhook delivery attempts
WITH parsed AS (
    SELECT
        id,
        CASE
            WHEN event_type = 'report_generated'
             AND COALESCE(payload->'data'->>'report_id', '') ~* '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
            THEN (payload->'data'->>'report_id')::uuid
            WHEN COALESCE(payload->'data'->>'company_signal_id', payload->'data'->>'signal_id', '') ~* '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
            THEN COALESCE(payload->'data'->>'company_signal_id', payload->'data'->>'signal_id')::uuid
            ELSE NULL
        END AS parsed_signal_id,
        CASE
            WHEN COALESCE(payload->'data'->>'review_id', '') ~* '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
            THEN (payload->'data'->>'review_id')::uuid
            ELSE NULL
        END AS parsed_review_id,
        CASE
            WHEN COALESCE(payload->'data'->>'report_id', '') ~* '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
            THEN (payload->'data'->>'report_id')::uuid
            ELSE NULL
        END AS parsed_report_id,
        NULLIF(BTRIM(COALESCE(payload->>'vendor', payload->'data'->>'vendor_name', '')), '') AS parsed_vendor_name,
        NULLIF(BTRIM(COALESCE(payload->'data'->>'company_name', payload->'data'->>'company', '')), '') AS parsed_company_name
    FROM b2b_webhook_delivery_log
    WHERE signal_id IS NULL
       OR review_id IS NULL
       OR report_id IS NULL
       OR vendor_name IS NULL
       OR company_name IS NULL
)
UPDATE b2b_webhook_delivery_log AS log
SET signal_id = COALESCE(log.signal_id, parsed.parsed_signal_id),
    review_id = COALESCE(log.review_id, parsed.parsed_review_id),
    report_id = COALESCE(log.report_id, parsed.parsed_report_id),
    vendor_name = COALESCE(log.vendor_name, parsed.parsed_vendor_name),
    company_name = COALESCE(log.company_name, parsed.parsed_company_name)
FROM parsed
WHERE log.id = parsed.id;
