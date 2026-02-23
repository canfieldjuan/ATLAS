-- Migration 040: Business contexts table
-- Stores business identities for multi-tenant call/SMS handling.
-- Each context maps phone numbers to a persona, schedule, and services.

CREATE TABLE IF NOT EXISTS business_contexts (
    id              VARCHAR(64) PRIMARY KEY,
    name            VARCHAR(256) NOT NULL,
    description     TEXT DEFAULT '',

    -- Phone numbers (E.164 format)
    phone_numbers   TEXT[] NOT NULL DEFAULT '{}',

    -- Voice persona
    greeting        TEXT DEFAULT 'Hello, how can I help you today?',
    voice_name      VARCHAR(128) DEFAULT 'Atlas',
    persona         TEXT DEFAULT '',

    -- Business info (fed to LLM for call intelligence)
    business_type   VARCHAR(128) DEFAULT '',
    services        TEXT[] DEFAULT '{}',
    service_area    TEXT DEFAULT '',
    pricing_info    TEXT DEFAULT '',

    -- Operating hours (24-hour format, NULL = closed)
    monday_open     VARCHAR(5) DEFAULT '09:00',
    monday_close    VARCHAR(5) DEFAULT '17:00',
    tuesday_open    VARCHAR(5) DEFAULT '09:00',
    tuesday_close   VARCHAR(5) DEFAULT '17:00',
    wednesday_open  VARCHAR(5) DEFAULT '09:00',
    wednesday_close VARCHAR(5) DEFAULT '17:00',
    thursday_open   VARCHAR(5) DEFAULT '09:00',
    thursday_close  VARCHAR(5) DEFAULT '17:00',
    friday_open     VARCHAR(5) DEFAULT '09:00',
    friday_close    VARCHAR(5) DEFAULT '17:00',
    saturday_open   VARCHAR(5),
    saturday_close  VARCHAR(5),
    sunday_open     VARCHAR(5),
    sunday_close    VARCHAR(5),
    timezone        VARCHAR(64) DEFAULT 'America/Chicago',
    after_hours_message TEXT DEFAULT 'Thank you for calling. We are currently closed. Please call back during business hours.',

    -- Scheduling
    scheduling_enabled          BOOLEAN DEFAULT TRUE,
    scheduling_calendar_id      VARCHAR(256),
    scheduling_min_notice_hours INTEGER DEFAULT 24,
    scheduling_max_advance_days INTEGER DEFAULT 30,
    scheduling_default_duration INTEGER DEFAULT 60,
    scheduling_buffer_minutes   INTEGER DEFAULT 15,

    -- Call handling
    transfer_number             VARCHAR(32),
    take_messages               BOOLEAN DEFAULT TRUE,
    max_call_duration_minutes   INTEGER DEFAULT 10,

    -- SMS
    sms_enabled     BOOLEAN DEFAULT TRUE,
    sms_auto_reply  BOOLEAN DEFAULT TRUE,

    -- Lifecycle
    enabled         BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_business_contexts_enabled
    ON business_contexts(enabled) WHERE enabled = TRUE;

CREATE INDEX IF NOT EXISTS idx_business_contexts_phone
    ON business_contexts USING GIN(phone_numbers);
