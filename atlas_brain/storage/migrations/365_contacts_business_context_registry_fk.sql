-- 365: make business_contexts the ENFORCED tenant registry (Slice 0 / ATLAS #2318).
--
-- Closes the tenant-EXISTENCE axis that D1 (#2317) left open: create_contact's
-- guard validates tenant PRESENCE only, so a non-blank-but-unknown
-- business_context_id reaches the provider and creates an orphan. This seeds the
-- registry and adds the FK contacts.business_context_id -> business_contexts.id.
--
-- The FK governs EVERY contact writer, not just the MCP create_contact path, so:
--  * seed the three legitimate contexts every writer can produce -- effingham_maids
--    and churnsignals (EOM / B2B) plus `personal`, the atlas_comms ContextRouter
--    fallback that unmatched inbound SMS resolves to -- so those writes do not hit
--    an FK violation and silently drop the contact/interaction;
--  * plus any OTHER tenant already stamped on contacts (dynamic backstop).
--
-- Seed + FK run in ONE DO block on purpose. The migration runner is AUTOCOMMIT
-- (it splits statements so CREATE INDEX CONCURRENTLY can run), so a plain
-- multi-statement seed-then-ALTER would leave a window where another (old / pre-
-- migration) app instance commits a contact with a new tenant AFTER the seed
-- snapshot but BEFORE the ALTER's lock -- that tenant would be missing from
-- business_contexts and fail FK validation, aborting the migration. A DO block is
-- its own transaction, so `LOCK TABLE contacts IN SHARE MODE` (which blocks writes)
-- is held across BOTH the seed and the ALTER, making the pair atomic w.r.t. writes.
DO $$
BEGIN
    LOCK TABLE contacts IN SHARE MODE;

    -- Registry seed. enabled = FALSE keeps these rows OUT of the voice loader
    -- (list_enabled filters on enabled = TRUE); their NULL voice/scheduling/SMS
    -- config is written explicitly (not left to migration 040 defaults, which
    -- would manufacture an active 'Atlas' voice / 9-5 hours / SMS enabled).
    -- ON CONFLICT DO NOTHING preserves any row the voice product already owns.
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
        d.id, d.name, '{}'::text[], FALSE,
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
            ('churnsignals',    'ChurnSignals'),
            ('personal',        'Personal')
        UNION
        SELECT DISTINCT c.business_context_id, c.business_context_id
        FROM contacts c
        WHERE c.business_context_id IS NOT NULL
          AND c.business_context_id NOT IN ('effingham_maids', 'churnsignals', 'personal')
    ) AS d(id, name)
    ON CONFLICT (id) DO NOTHING;

    -- The FK -- durable existence enforcement. Scoped to the contacts relation
    -- (constraint names are schema-local) and idempotent. NULL business_context_id
    -- is allowed by the FK (the D1 create-time guard forbids NULL on the agent path).
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
