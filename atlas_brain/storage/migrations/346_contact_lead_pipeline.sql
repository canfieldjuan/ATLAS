-- Optional lead-pipeline state on the canonical CRM contact.
-- Existing contacts remain unchanged; new website leads initialize lead_stage
-- through the intake path after this migration is deployed.

ALTER TABLE contacts
    ADD COLUMN IF NOT EXISTS lead_stage VARCHAR(64),
    ADD COLUMN IF NOT EXISTS lead_owner VARCHAR(128),
    ADD COLUMN IF NOT EXISTS next_follow_up_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_contacts_lead_follow_up
    ON contacts (business_context_id, next_follow_up_at, updated_at)
    WHERE contact_type = 'lead'
      AND status = 'active'
      AND next_follow_up_at IS NOT NULL;

COMMENT ON COLUMN contacts.lead_stage IS
    'Caller-defined pipeline stage for contact_type=lead';
COMMENT ON COLUMN contacts.lead_owner IS
    'Operator responsible for the lead; not a user-table foreign key';
COMMENT ON COLUMN contacts.next_follow_up_at IS
    'Next operator follow-up time for the lead';
