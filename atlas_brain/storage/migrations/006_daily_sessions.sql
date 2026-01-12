-- Migration: 006_daily_sessions.sql
-- Add turn_type for separating commands from conversations
-- Add session_date for daily session management

-- Add turn_type to conversation_turns
-- Values: 'conversation' (default), 'command'
ALTER TABLE conversation_turns
ADD COLUMN IF NOT EXISTS turn_type VARCHAR(20) DEFAULT 'conversation';

-- Add index for filtering by turn_type
CREATE INDEX IF NOT EXISTS idx_conversation_turns_type
    ON conversation_turns(session_id, turn_type);

-- Add session_date to sessions for daily session lookup
ALTER TABLE sessions
ADD COLUMN IF NOT EXISTS session_date DATE DEFAULT CURRENT_DATE;

-- Add index for daily session lookup
CREATE INDEX IF NOT EXISTS idx_sessions_user_date
    ON sessions(user_id, session_date, is_active);

-- Update existing sessions to have correct session_date
UPDATE sessions
SET session_date = DATE(started_at)
WHERE session_date IS NULL;
