-- Add speaker_uuid column to conversation_turns for proper UUID linkage.
-- speaker_id continues to store the display name; speaker_uuid stores the users.id FK.

ALTER TABLE conversation_turns ADD COLUMN IF NOT EXISTS speaker_uuid UUID;

CREATE INDEX IF NOT EXISTS idx_turns_speaker_uuid
    ON conversation_turns (speaker_uuid) WHERE speaker_uuid IS NOT NULL;
