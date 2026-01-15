-- Reminders table for user-created time-based alerts
-- Migration: 010_reminders.sql

CREATE TABLE IF NOT EXISTS reminders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message TEXT NOT NULL,
    due_at TIMESTAMP WITH TIME ZONE NOT NULL,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed BOOLEAN DEFAULT FALSE,
    completed_at TIMESTAMP WITH TIME ZONE,
    delivered BOOLEAN DEFAULT FALSE,
    delivered_at TIMESTAMP WITH TIME ZONE,
    repeat_pattern VARCHAR(16),  -- "daily", "weekly", "monthly", or NULL
    source VARCHAR(32) DEFAULT 'voice',  -- "voice", "api", "scheduled"
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Index for finding due reminders efficiently (most common query)
CREATE INDEX IF NOT EXISTS idx_reminders_due_pending
    ON reminders(due_at ASC)
    WHERE completed = FALSE AND delivered = FALSE;

-- Index for user's reminders
CREATE INDEX IF NOT EXISTS idx_reminders_user_active
    ON reminders(user_id, due_at ASC)
    WHERE completed = FALSE;

-- Index for cleanup of old completed reminders
CREATE INDEX IF NOT EXISTS idx_reminders_completed_at
    ON reminders(completed_at DESC)
    WHERE completed = TRUE;

COMMENT ON TABLE reminders IS 'User reminders with optional recurrence';
COMMENT ON COLUMN reminders.due_at IS 'When the reminder should trigger (timezone-aware)';
COMMENT ON COLUMN reminders.repeat_pattern IS 'Recurrence: daily, weekly, monthly, or NULL for one-time';
COMMENT ON COLUMN reminders.delivered IS 'Whether the reminder notification was sent';
COMMENT ON COLUMN reminders.source IS 'How the reminder was created: voice, api, or scheduled';
