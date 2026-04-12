-- Backfill exact CRM activity references from canonical signal and report rows
WITH signal_matches AS (
    SELECT
        log.id,
        cs.review_id AS canonical_review_id,
        NULLIF(BTRIM(cs.vendor_name), '') AS canonical_vendor_name,
        NULLIF(BTRIM(cs.company_name), '') AS canonical_company_name
    FROM b2b_crm_push_log AS log
    JOIN b2b_company_signals AS cs
      ON cs.id = log.signal_id
    WHERE log.signal_type IN ('company_signal', 'high_intent_push', 'change_event')
      AND (
          log.review_id IS NULL
          OR NULLIF(BTRIM(log.vendor_name), '') IS NULL
          OR NULLIF(BTRIM(log.company_name), '') IS NULL
      )
)
UPDATE b2b_crm_push_log AS log
SET review_id = COALESCE(log.review_id, signal_matches.canonical_review_id),
    vendor_name = COALESCE(
        NULLIF(BTRIM(log.vendor_name), ''),
        signal_matches.canonical_vendor_name,
        log.vendor_name
    ),
    company_name = COALESCE(
        NULLIF(BTRIM(log.company_name), ''),
        signal_matches.canonical_company_name,
        log.company_name
    )
FROM signal_matches
WHERE log.id = signal_matches.id;

WITH report_matches AS (
    SELECT
        log.id,
        NULLIF(BTRIM(report.vendor_filter), '') AS canonical_vendor_name
    FROM b2b_crm_push_log AS log
    JOIN b2b_intelligence AS report
      ON report.id = log.signal_id
    WHERE log.signal_type = 'report_generated'
      AND NULLIF(BTRIM(log.vendor_name), '') IS NULL
)
UPDATE b2b_crm_push_log AS log
SET vendor_name = COALESCE(
    NULLIF(BTRIM(log.vendor_name), ''),
    report_matches.canonical_vendor_name,
    log.vendor_name
)
FROM report_matches
WHERE log.id = report_matches.id;
