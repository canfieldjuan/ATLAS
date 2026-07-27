-- Durable, opaque link between one approved EOM lead and tracker records.
-- Operational service/rate/schedule facts remain in the EOM time tracker.

CREATE TABLE IF NOT EXISTS eom_customer_handoffs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    contact_id UUID NOT NULL UNIQUE REFERENCES contacts(id) ON DELETE RESTRICT,
    approval_key VARCHAR(128) NOT NULL UNIQUE,
    tracker_customer_id BIGINT NOT NULL CHECK (tracker_customer_id > 0),
    tracker_site_id BIGINT NOT NULL CHECK (tracker_site_id > 0),
    approved_by_employee_id BIGINT NOT NULL CHECK (approved_by_employee_id > 0),
    approved_by_name VARCHAR(128) NOT NULL,
    finalized_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_eom_customer_handoffs_finalized
    ON eom_customer_handoffs (finalized_at DESC);

COMMENT ON TABLE eom_customer_handoffs IS
    'One immutable tracker Customer/Site link for each Atlas-approved EOM lead.';
