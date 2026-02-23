-- SMS messages: persistent storage for inbound and outbound SMS
-- Stores message content, CRM link, intent classification, and delivery status

CREATE TABLE IF NOT EXISTS sms_messages (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_sid         VARCHAR(128) NOT NULL,
    from_number         VARCHAR(32) NOT NULL,
    to_number           VARCHAR(32) NOT NULL,
    direction           VARCHAR(10) NOT NULL DEFAULT 'inbound',
    body                TEXT NOT NULL DEFAULT '',
    media_urls          JSONB DEFAULT '[]'::jsonb,
    business_context_id VARCHAR(64),
    conversation_id     UUID,
    intent              VARCHAR(32),
    extracted_data      JSONB DEFAULT '{}'::jsonb,
    summary             TEXT,
    contact_id          UUID REFERENCES contacts(id) ON DELETE SET NULL,
    status              VARCHAR(32) NOT NULL DEFAULT 'received',
    error_message       TEXT,
    notified            BOOLEAN DEFAULT FALSE,
    source              VARCHAR(64),
    source_ref          VARCHAR(256),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    delivered_at        TIMESTAMPTZ,
    processed_at        TIMESTAMPTZ
);

-- Lookup by provider message SID (unique prevents webhook retry duplicates)
CREATE UNIQUE INDEX IF NOT EXISTS idx_sms_messages_message_sid ON sms_messages(message_sid);

-- Recent messages by direction
CREATE INDEX IF NOT EXISTS idx_sms_messages_direction_created ON sms_messages(direction, created_at DESC);

-- Conversation threading by phone pair
CREATE INDEX IF NOT EXISTS idx_sms_messages_from_created ON sms_messages(from_number, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sms_messages_to_created ON sms_messages(to_number, created_at DESC);

-- CRM contact link
CREATE INDEX IF NOT EXISTS idx_sms_messages_contact_id ON sms_messages(contact_id);

-- Business context queries
CREATE INDEX IF NOT EXISTS idx_sms_messages_context_created ON sms_messages(business_context_id, created_at DESC);

-- Conversation grouping
CREATE INDEX IF NOT EXISTS idx_sms_messages_conversation_id ON sms_messages(conversation_id);

-- Status filtering
CREATE INDEX IF NOT EXISTS idx_sms_messages_status ON sms_messages(status);

-- Intent filtering
CREATE INDEX IF NOT EXISTS idx_sms_messages_intent ON sms_messages(intent);
