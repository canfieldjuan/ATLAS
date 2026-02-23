-- Customer service agreements for recurring invoicing.
-- Each row links a CRM contact to a service definition with calendar matching
-- for automated monthly invoice generation.

CREATE TABLE IF NOT EXISTS customer_services (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    contact_id          UUID NOT NULL REFERENCES contacts(id) ON DELETE CASCADE,

    -- Service definition
    service_name        VARCHAR(256) NOT NULL,
    service_description TEXT,

    -- Pricing
    rate                NUMERIC(12,2) NOT NULL,
    rate_label          VARCHAR(64) DEFAULT 'Per Visit',
    tax_rate            NUMERIC(5,4) DEFAULT 0,

    -- Calendar matching
    calendar_keyword    VARCHAR(256) NOT NULL,
    calendar_id         VARCHAR(256),

    -- Auto-invoicing control
    auto_invoice        BOOLEAN NOT NULL DEFAULT TRUE,
    last_invoiced_at    DATE,
    next_invoice_date   DATE,

    -- Lifecycle
    status              VARCHAR(32) NOT NULL DEFAULT 'active',
    start_date          DATE NOT NULL DEFAULT CURRENT_DATE,
    end_date            DATE,

    -- Linking
    business_context_id VARCHAR(64),
    notes               TEXT,
    metadata            JSONB DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_customer_services_contact ON customer_services(contact_id);
CREATE INDEX IF NOT EXISTS idx_customer_services_status ON customer_services(status);
CREATE INDEX IF NOT EXISTS idx_customer_services_auto ON customer_services(auto_invoice) WHERE auto_invoice = TRUE AND status = 'active';
