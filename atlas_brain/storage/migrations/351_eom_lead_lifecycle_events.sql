-- Immutable lifecycle evidence for the EOM lead funnel.
--
-- Initial contact creation is the first lifecycle transition.  Recording it
-- in an EOM-only trigger keeps the contact and its audit row in the same
-- database transaction. Later funnel endpoints will insert actor-authenticated
-- booking and conversion events through this same ledger.
-- Lifecycle evidence intentionally prevents hard deletion of its contact;
-- runtime contact removal is a soft archive until a retention policy provides
-- a sanctioned purge path.

CREATE TABLE IF NOT EXISTS eom_lead_lifecycle_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    contact_id UUID NOT NULL REFERENCES contacts(id) ON DELETE RESTRICT,
    event_type VARCHAR(64) NOT NULL,
    from_stage VARCHAR(64),
    to_stage VARCHAR(64),
    actor VARCHAR(128) NOT NULL DEFAULT 'system',
    source VARCHAR(64) NOT NULL,
    operation_key VARCHAR(256),
    reason TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_eom_lead_lifecycle_operation
    ON eom_lead_lifecycle_events (contact_id, event_type, operation_key)
    WHERE operation_key IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_eom_lead_lifecycle_contact
    ON eom_lead_lifecycle_events (contact_id, occurred_at DESC);

CREATE OR REPLACE FUNCTION prevent_eom_lead_lifecycle_event_mutation()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'eom_lead_lifecycle_events is append-only';
END;
$$;

DROP TRIGGER IF EXISTS trg_prevent_eom_lead_lifecycle_event_mutation
    ON eom_lead_lifecycle_events;
CREATE TRIGGER trg_prevent_eom_lead_lifecycle_event_mutation
    BEFORE UPDATE OR DELETE ON eom_lead_lifecycle_events
    FOR EACH ROW
    EXECUTE FUNCTION prevent_eom_lead_lifecycle_event_mutation();

DROP TRIGGER IF EXISTS trg_prevent_eom_lead_lifecycle_event_truncate
    ON eom_lead_lifecycle_events;
CREATE TRIGGER trg_prevent_eom_lead_lifecycle_event_truncate
    BEFORE TRUNCATE ON eom_lead_lifecycle_events
    FOR EACH STATEMENT
    EXECUTE FUNCTION prevent_eom_lead_lifecycle_event_mutation();

CREATE OR REPLACE FUNCTION record_eom_lead_created()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.business_context_id = 'effingham_maids'
       AND NEW.contact_type = 'lead'
       AND NEW.lead_stage = 'new' THEN
        INSERT INTO eom_lead_lifecycle_events (
            contact_id,
            event_type,
            from_stage,
            to_stage,
            actor,
            source,
            operation_key,
            metadata
        ) VALUES (
            NEW.id,
            'lead_created',
            NULL,
            'new',
            'system',
            COALESCE(NULLIF(NEW.source, ''), 'unknown'),
            COALESCE(NULLIF(NEW.source_ref, ''), 'contact:' || NEW.id::text),
            jsonb_build_object('contact_source', NEW.source)
        )
        ON CONFLICT (contact_id, event_type, operation_key)
            WHERE operation_key IS NOT NULL
            DO NOTHING;
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_record_eom_lead_created ON contacts;
CREATE TRIGGER trg_record_eom_lead_created
    AFTER INSERT ON contacts
    FOR EACH ROW
    EXECUTE FUNCTION record_eom_lead_created();

COMMENT ON TABLE eom_lead_lifecycle_events IS
    'Immutable EOM funnel lifecycle events; contact insert trigger records lead_created; evidence blocks hard contact deletion.';
