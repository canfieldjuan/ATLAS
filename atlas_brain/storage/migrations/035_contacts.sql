-- CRM: Contacts table + interaction log
-- Migration: 035_contacts.sql
--
-- contacts becomes the single source of truth for customer data.
-- Managed by Directus; also directly queryable via asyncpg (DatabaseCRMProvider).
-- Directus will auto-discover this table when connected to the same Postgres instance.

CREATE TABLE IF NOT EXISTS contacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Core identity
    full_name   VARCHAR(256) NOT NULL,
    first_name  VARCHAR(128),
    last_name   VARCHAR(128),

    -- Contact channels
    email       VARCHAR(256),
    phone       VARCHAR(32),

    -- Address
    address     TEXT,
    city        VARCHAR(128),
    state       VARCHAR(64),
    zip         VARCHAR(16),

    -- Business context
    business_context_id VARCHAR(64),
    contact_type        VARCHAR(32) NOT NULL DEFAULT 'customer',   -- customer, lead, prospect, vendor
    status              VARCHAR(32) NOT NULL DEFAULT 'active',      -- active, inactive, archived

    -- Notes / tagging
    tags        TEXT[]  DEFAULT '{}',
    notes       TEXT,

    -- Origin tracking
    source      VARCHAR(64)  DEFAULT 'manual',  -- phone_call, email, manual, appointment_import, web
    source_ref  VARCHAR(256),                   -- original appointment UUID if imported

    -- Timestamps
    created_at  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Extensible
    metadata    JSONB DEFAULT '{}'::jsonb
);

-- Full-text search on name
CREATE INDEX IF NOT EXISTS idx_contacts_name_fts
    ON contacts USING GIN(to_tsvector('english', full_name));

-- Exact lookups by phone / email
CREATE INDEX IF NOT EXISTS idx_contacts_phone
    ON contacts(phone);
CREATE INDEX IF NOT EXISTS idx_contacts_email
    ON contacts(email);

-- Scoped list queries
CREATE INDEX IF NOT EXISTS idx_contacts_business_context
    ON contacts(business_context_id, status, contact_type);
CREATE INDEX IF NOT EXISTS idx_contacts_status
    ON contacts(status, updated_at DESC);

-- -----------------------------------------------------------------------
-- Interaction log: one row per customer touch-point
-- -----------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS contact_interactions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    contact_id      UUID NOT NULL REFERENCES contacts(id) ON DELETE CASCADE,
    interaction_type VARCHAR(32) NOT NULL, -- call, email, appointment, sms, note, meeting
    summary         TEXT,
    intent          VARCHAR(256),          -- e.g. booking, inquiry, complaint, follow_up
    occurred_at     TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata        JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_contact_interactions_contact
    ON contact_interactions(contact_id, occurred_at DESC);

-- -----------------------------------------------------------------------
-- Link appointments to contacts (opt-in; legacy rows keep contact_id NULL)
-- -----------------------------------------------------------------------

ALTER TABLE appointments
    ADD COLUMN IF NOT EXISTS contact_id UUID REFERENCES contacts(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_appointments_contact_id
    ON appointments(contact_id)
    WHERE contact_id IS NOT NULL;

-- -----------------------------------------------------------------------
-- Comments
-- -----------------------------------------------------------------------

COMMENT ON TABLE contacts IS
    'CRM: single source of truth for customer/contact data (Directus-managed)';
COMMENT ON TABLE contact_interactions IS
    'Log of every customer interaction: calls, emails, appointments, notes';
COMMENT ON COLUMN contacts.source IS
    'Origin: phone_call | email | manual | appointment_import | web';
COMMENT ON COLUMN contacts.source_ref IS
    'Reference ID from originating system (e.g. appointments.id when source=appointment_import)';
COMMENT ON COLUMN appointments.contact_id IS
    'CRM contact this appointment belongs to; NULL for legacy rows not yet linked';


