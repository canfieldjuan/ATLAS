-- Appointments table for scheduled customer appointments
-- Migration: 012_appointments.sql

CREATE TABLE IF NOT EXISTS appointments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Timing
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE NOT NULL,
    duration_minutes INTEGER NOT NULL DEFAULT 60,

    -- Service details
    service_type VARCHAR(128) NOT NULL,
    notes TEXT DEFAULT '',

    -- Customer info
    customer_name VARCHAR(256) NOT NULL,
    customer_phone VARCHAR(32) NOT NULL,
    customer_email VARCHAR(256),
    customer_address TEXT,

    -- Booking metadata
    calendar_event_id VARCHAR(256),  -- Google Calendar event ID
    business_context_id VARCHAR(64) NOT NULL,
    call_id UUID,  -- Link to the call that created this (if any)

    -- Status tracking
    status VARCHAR(32) NOT NULL DEFAULT 'confirmed',  -- confirmed, cancelled, completed, no_show
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    cancelled_at TIMESTAMP WITH TIME ZONE,
    cancellation_reason TEXT,

    -- Confirmation tracking
    confirmation_sent BOOLEAN DEFAULT FALSE,
    confirmation_sent_at TIMESTAMP WITH TIME ZONE,
    reminder_sent BOOLEAN DEFAULT FALSE,
    reminder_sent_at TIMESTAMP WITH TIME ZONE,

    -- Extensible metadata
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Index for finding appointments by date range (most common query)
CREATE INDEX IF NOT EXISTS idx_appointments_start_time
    ON appointments(start_time ASC)
    WHERE status = 'confirmed';

-- Index for finding appointments by customer phone (for reschedule/cancel)
CREATE INDEX IF NOT EXISTS idx_appointments_customer_phone
    ON appointments(customer_phone, start_time DESC)
    WHERE status = 'confirmed';

-- Index for business context queries
CREATE INDEX IF NOT EXISTS idx_appointments_business_context
    ON appointments(business_context_id, start_time ASC)
    WHERE status = 'confirmed';

-- Index for calendar event lookups
CREATE INDEX IF NOT EXISTS idx_appointments_calendar_event
    ON appointments(calendar_event_id)
    WHERE calendar_event_id IS NOT NULL;

-- Messages table for voicemail/callback requests
CREATE TABLE IF NOT EXISTS appointment_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Caller info
    caller_phone VARCHAR(32) NOT NULL,
    caller_name VARCHAR(256),

    -- Message content
    message_text TEXT NOT NULL,

    -- Context
    business_context_id VARCHAR(64) NOT NULL,
    call_id UUID,

    -- Status
    read BOOLEAN DEFAULT FALSE,
    read_at TIMESTAMP WITH TIME ZONE,
    responded BOOLEAN DEFAULT FALSE,
    responded_at TIMESTAMP WITH TIME ZONE,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    metadata JSONB DEFAULT '{}'::jsonb
);

-- Index for unread messages
CREATE INDEX IF NOT EXISTS idx_appointment_messages_unread
    ON appointment_messages(created_at DESC)
    WHERE read = FALSE;

COMMENT ON TABLE appointments IS 'Customer appointments booked via phone/chat';
COMMENT ON COLUMN appointments.status IS 'confirmed, cancelled, completed, no_show';
COMMENT ON COLUMN appointments.calendar_event_id IS 'Google Calendar event ID for sync';
COMMENT ON TABLE appointment_messages IS 'Voicemail and callback request messages';
