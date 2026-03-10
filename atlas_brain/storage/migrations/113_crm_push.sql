-- Phase 5 Sprint 4: CRM outbound push
--
-- Extends webhook subscriptions to support direct CRM API push.
-- New channel types format payloads for HubSpot, Salesforce, and Pipedrive
-- native APIs.  auth_header stores the CRM API credential.

-- Expand channel CHECK to include CRM channels
ALTER TABLE b2b_webhook_subscriptions DROP CONSTRAINT IF EXISTS b2b_webhook_subscriptions_channel_check;
ALTER TABLE b2b_webhook_subscriptions
    ADD CONSTRAINT b2b_webhook_subscriptions_channel_check
        CHECK (channel IN (
            'generic', 'slack', 'teams',
            'crm_hubspot', 'crm_salesforce', 'crm_pipedrive'
        ));

-- Auth header for CRM API calls (e.g., "Bearer pat-xxx" for HubSpot)
ALTER TABLE b2b_webhook_subscriptions
    ADD COLUMN IF NOT EXISTS auth_header TEXT;

-- Push tracking: which signals have been synced to avoid duplicates
CREATE TABLE IF NOT EXISTS b2b_crm_push_log (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subscription_id UUID NOT NULL REFERENCES b2b_webhook_subscriptions(id) ON DELETE CASCADE,
    signal_type     TEXT NOT NULL,       -- churn_signal, company_signal, change_event
    signal_id       UUID,                -- Reference to source signal row
    vendor_name     TEXT NOT NULL,
    company_name    TEXT,                -- For company_signal pushes
    crm_record_id   TEXT,               -- ID returned by CRM after create/update
    crm_record_type TEXT,               -- deal, contact, company, note
    status          TEXT NOT NULL DEFAULT 'pushed',
    error           TEXT,
    pushed_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_crm_push_log_sub
    ON b2b_crm_push_log (subscription_id, pushed_at DESC);

CREATE INDEX IF NOT EXISTS idx_crm_push_log_signal
    ON b2b_crm_push_log (signal_type, signal_id)
    WHERE signal_id IS NOT NULL;
