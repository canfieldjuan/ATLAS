-- Prevent duplicate reports for the same (date, type, vendor, category, account).
-- COALESCE handles NULLs, LOWER makes matching case-insensitive.
DROP INDEX IF EXISTS idx_b2b_intelligence_dedup;
CREATE UNIQUE INDEX idx_b2b_intelligence_dedup
ON b2b_intelligence (
    report_date,
    report_type,
    LOWER(COALESCE(vendor_filter, '')),
    LOWER(COALESCE(category_filter, '')),
    COALESCE(account_id, '00000000-0000-0000-0000-000000000000'::uuid)
);
