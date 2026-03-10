-- Migration 108: CRM Event Ingestion
--
-- Stores inbound CRM events (deal stage changes, activity logs, contact updates)
-- from external CRM systems (HubSpot, Salesforce, Pipedrive, generic webhook).
-- An autonomous task processes pending events and auto-records campaign outcomes.

CREATE TABLE IF NOT EXISTS b2b_crm_events (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Source identification
    crm_provider    TEXT NOT NULL,      -- hubspot, salesforce, pipedrive, generic
    crm_event_id    TEXT,               -- Provider's native event ID (for dedup)
    event_type      TEXT NOT NULL,       -- deal_stage_change, deal_won, deal_lost, meeting_booked, activity_logged, contact_updated
    -- Entity references
    company_name    TEXT,
    contact_email   TEXT,
    contact_name    TEXT,
    deal_id         TEXT,               -- Provider's deal/opportunity ID
    deal_name       TEXT,
    -- Event data
    deal_stage      TEXT,               -- Current stage (e.g. "closed_won", "demo_scheduled")
    deal_amount     NUMERIC(12,2),      -- Deal value if available
    event_data      JSONB NOT NULL DEFAULT '{}',  -- Full provider payload for audit
    -- Processing state
    status          TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'processed', 'matched', 'unmatched', 'skipped', 'error')),
    matched_sequence_id UUID REFERENCES campaign_sequences(id) ON DELETE SET NULL,
    outcome_recorded TEXT,              -- The outcome that was auto-recorded (if any)
    processing_notes TEXT,
    -- Tenant scoping
    account_id      UUID REFERENCES saas_accounts(id) ON DELETE SET NULL,
    -- Timestamps
    event_timestamp TIMESTAMPTZ,        -- When the event happened in the CRM
    received_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at    TIMESTAMPTZ
);

-- Dedup: same provider event should not be ingested twice
CREATE UNIQUE INDEX IF NOT EXISTS idx_crm_events_dedup
    ON b2b_crm_events (crm_provider, crm_event_id)
    WHERE crm_event_id IS NOT NULL;

-- Processing queue: pending events ordered by receipt
CREATE INDEX IF NOT EXISTS idx_crm_events_pending
    ON b2b_crm_events (received_at ASC)
    WHERE status = 'pending';

-- Lookup by company for matching
CREATE INDEX IF NOT EXISTS idx_crm_events_company
    ON b2b_crm_events (LOWER(company_name))
    WHERE company_name IS NOT NULL;

-- Lookup by matched sequence
CREATE INDEX IF NOT EXISTS idx_crm_events_sequence
    ON b2b_crm_events (matched_sequence_id)
    WHERE matched_sequence_id IS NOT NULL;

-- Tenant scoping
CREATE INDEX IF NOT EXISTS idx_crm_events_account
    ON b2b_crm_events (account_id)
    WHERE account_id IS NOT NULL;
