-- 365: make business_contexts the ENFORCED tenant registry (Slice 0 / ATLAS #2318).
--
-- Closes the tenant-EXISTENCE axis that D1 (#2317) deliberately left open: the
-- create_contact guard there validates tenant PRESENCE only, because
-- business_contexts was empty and there was no FK, so validating existence would
-- have rejected every real tenant. This migration seeds the registry and adds the
-- FK, after which existence can be enforced (the DB is the durable enforcement;
-- the create_contact guard adds the clean typed refusal).
--
-- Ordering is load-bearing: SEED BEFORE the FK. In prod `contacts` already carries
-- ~709 rows stamped effingham_maids/churnsignals, so those ids must exist in
-- business_contexts before the constraint validates.

-- (1) Named registry rows for the two real tenants. These are REGISTRY seeds, not
--     voice-assistant configuration: business_contexts is also the voice product's
--     config table (greeting/persona/hours/...), but it is empty in prod and those
--     columns stay NULL here. `phone_numbers` is NOT NULL, so an empty array
--     satisfies it. The voice product can populate config later via upsert.
--     Idempotent: ON CONFLICT preserves any row the voice product already owns.
INSERT INTO business_contexts (id, name, phone_numbers, enabled)
VALUES
    ('effingham_maids', 'Effingham Office Maids', '{}', TRUE),
    ('churnsignals',    'ChurnSignals',           '{}', TRUE)
ON CONFLICT (id) DO NOTHING;

-- (2) Backstop: seed any OTHER tenant already stamped on contacts (id doubles as
--     the display name -- rename later) so the FK can never fail on an unexpected
--     tenant at deploy time. In a fresh DB `contacts` is empty and this is a no-op.
INSERT INTO business_contexts (id, name, phone_numbers, enabled)
SELECT DISTINCT business_context_id, business_context_id, '{}', TRUE
FROM contacts
WHERE business_context_id IS NOT NULL
ON CONFLICT (id) DO NOTHING;

-- (3) The FK -- the durable existence enforcement. A NULL business_context_id is
--     allowed by the FK (the D1 create-time guard forbids NULL on the agent path);
--     this closes the "non-blank but unknown tenant" hole D1 left open. Added
--     idempotently so a re-run is safe.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'contacts_business_context_id_fkey'
    ) THEN
        ALTER TABLE contacts
            ADD CONSTRAINT contacts_business_context_id_fkey
            FOREIGN KEY (business_context_id)
            REFERENCES business_contexts (id);
    END IF;
END $$;
