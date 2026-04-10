ALTER TABLE b2b_report_subscriptions
    ADD COLUMN IF NOT EXISTS filter_payload JSONB NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE b2b_report_subscriptions
    DROP CONSTRAINT IF EXISTS b2b_report_subscriptions_scope_match;

ALTER TABLE b2b_report_subscriptions
    DROP CONSTRAINT IF EXISTS b2b_report_subscriptions_scope_type_check;

ALTER TABLE b2b_report_subscriptions
    ADD CONSTRAINT b2b_report_subscriptions_scope_type_check
    CHECK (scope_type IN ('library', 'library_view', 'report'));

ALTER TABLE b2b_report_subscriptions
    ADD CONSTRAINT b2b_report_subscriptions_scope_match
    CHECK (
        (scope_type = 'library' AND report_id IS NULL AND scope_key = 'library' AND filter_payload = '{}'::jsonb)
        OR (scope_type = 'library_view' AND report_id IS NULL AND scope_key <> 'library' AND filter_payload <> '{}'::jsonb)
        OR (scope_type = 'report' AND report_id IS NOT NULL AND filter_payload = '{}'::jsonb)
    );

CREATE INDEX IF NOT EXISTS idx_b2b_report_subscriptions_filter_payload
    ON b2b_report_subscriptions USING GIN (filter_payload);
