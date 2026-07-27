-- Dedicated tenant-scoped customer-service tickets for CRM complaints.
-- Consumer-review complaint tables and Content Ops support tickets are separate
-- domains and intentionally remain unchanged.

CREATE TABLE IF NOT EXISTS customer_service_tickets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    contact_id UUID NOT NULL REFERENCES contacts(id) ON DELETE RESTRICT,
    business_context_id VARCHAR(64) NOT NULL,
    summary VARCHAR(500) NOT NULL,
    details TEXT,
    status VARCHAR(16) NOT NULL DEFAULT 'open',
    priority VARCHAR(64),
    assignee VARCHAR(128),
    resolution TEXT,
    closed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT customer_service_tickets_tenant_check
        CHECK (btrim(business_context_id) <> ''),
    CONSTRAINT customer_service_tickets_summary_check
        CHECK (btrim(summary) <> ''),
    CONSTRAINT customer_service_tickets_status_check
        CHECK (status IN ('open', 'closed')),
    CONSTRAINT customer_service_tickets_close_fields_check
        CHECK (
            (status = 'open' AND resolution IS NULL AND closed_at IS NULL)
            OR
            (
                status = 'closed'
                AND NULLIF(btrim(resolution), '') IS NOT NULL
                AND closed_at IS NOT NULL
            )
        )
);

CREATE INDEX IF NOT EXISTS idx_customer_service_tickets_open_queue
    ON customer_service_tickets (
        business_context_id,
        created_at DESC,
        id DESC
    )
    WHERE status = 'open';

CREATE INDEX IF NOT EXISTS idx_customer_service_tickets_contact
    ON customer_service_tickets (
        business_context_id,
        contact_id,
        created_at DESC
    );

COMMENT ON TABLE customer_service_tickets IS
    'Tenant-scoped CRM customer-service complaints linked to canonical contacts';
COMMENT ON COLUMN customer_service_tickets.priority IS
    'Caller-defined priority label; no global taxonomy is implied';
COMMENT ON COLUMN customer_service_tickets.assignee IS
    'Operator label responsible for the ticket; not a user-table foreign key';
