ALTER TABLE b2b_report_subscription_delivery_log
    DROP CONSTRAINT IF EXISTS b2b_report_subscription_delivery_log_freshness_state_check;

ALTER TABLE b2b_report_subscription_delivery_log
    ADD CONSTRAINT b2b_report_subscription_delivery_log_freshness_state_check
    CHECK (
        freshness_state IN (
            'fresh',
            'monitor',
            'stale',
            'unknown',
            'mixed',
            'none',
            'blocked'
        )
    );
