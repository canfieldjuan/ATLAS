-- 365: make business_contexts the ENFORCED tenant registry (Slice 0 / ATLAS #2318).
--
-- Closes the tenant-EXISTENCE axis that D1 (#2317) left open: create_contact's
-- guard validates tenant PRESENCE only, so a non-blank-but-unknown
-- business_context_id reaches the provider and creates an orphan. This seeds the
-- registry and adds the FK, after which existence is enforced (the DB is the
-- durable enforcement; the create_contact guard adds the clean typed refusal).
--
-- Ordering is load-bearing: SEED BEFORE the FK. In prod `contacts` already carries
-- ~709 rows stamped effingham_maids/churnsignals, so those ids must exist in
-- business_contexts before the constraint validates.

-- (1) Seed the tenant registry with voice-product config EXPLICITLY neutralized.
--     `d` is the two named tenants plus any OTHER tenant already stamped on
--     contacts (id doubles as the display name -- rename later; a fresh DB has no
--     contacts so that arm is a no-op). Every non-registry column is written NULL
--     (or FALSE for the enable flags) DIRECTLY in the INSERT rather than left to
--     migration 040's defaults -- business_contexts is also the voice product's
--     config table, and its defaults would otherwise manufacture an active 'Atlas'
--     voice, 09:00-17:00 hours, America/Chicago, and scheduling / SMS /
--     message-taking ENABLED for a pure registry row. (An UPDATE-after-INSERT in a
--     CTE cannot do this: a data-modifying CTE's inserts are not visible to a
--     sibling UPDATE of the same table in one statement.) ON CONFLICT DO NOTHING
--     preserves any row the voice product already owns. Idempotent.
INSERT INTO business_contexts (
    id, name, phone_numbers, enabled,
    description, greeting, voice_name, persona, business_type, services,
    service_area, pricing_info,
    monday_open, monday_close, tuesday_open, tuesday_close,
    wednesday_open, wednesday_close, thursday_open, thursday_close,
    friday_open, friday_close, saturday_open, saturday_close,
    sunday_open, sunday_close, timezone, after_hours_message,
    scheduling_enabled, scheduling_calendar_id, scheduling_min_notice_hours,
    scheduling_max_advance_days, scheduling_default_duration, scheduling_buffer_minutes,
    transfer_number, take_messages, max_call_duration_minutes,
    sms_enabled, sms_auto_reply
)
SELECT
    d.id, d.name, '{}'::text[], TRUE,
    NULL, NULL, NULL, NULL, NULL, NULL,
    NULL, NULL,
    NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL,
    FALSE, NULL, NULL,
    NULL, NULL, NULL,
    NULL, FALSE, NULL,
    FALSE, FALSE
FROM (
    VALUES
        ('effingham_maids', 'Effingham Office Maids'),
        ('churnsignals',    'ChurnSignals')
    UNION
    SELECT DISTINCT c.business_context_id, c.business_context_id
    FROM contacts c
    WHERE c.business_context_id IS NOT NULL
      AND c.business_context_id NOT IN ('effingham_maids', 'churnsignals')
) AS d(id, name)
ON CONFLICT (id) DO NOTHING;

-- (2) The FK -- the durable existence enforcement. Guarded idempotently AND scoped
--     to the `contacts` relation: constraint names are schema-local, so checking
--     the name alone could skip the ALTER if another schema (e.g. a disposable
--     test schema) already has a same-named constraint. A NULL business_context_id
--     is allowed by the FK (the D1 create-time guard forbids NULL on the agent
--     path); this closes the "non-blank but unknown tenant" hole D1 left open.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'contacts_business_context_id_fkey'
          AND conrelid = 'contacts'::regclass
    ) THEN
        ALTER TABLE contacts
            ADD CONSTRAINT contacts_business_context_id_fkey
            FOREIGN KEY (business_context_id)
            REFERENCES business_contexts (id);
    END IF;
END $$;
