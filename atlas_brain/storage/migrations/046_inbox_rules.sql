-- User-defined inbox automation rules (Gmail-filter-style).
-- Evaluated between CRM cross-reference (step 6) and LLM intent classification
-- (step 7) in the email intake pipeline.  First matching rule (by position) wins.

CREATE TABLE IF NOT EXISTS email_inbox_rules (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL,
    enabled     BOOLEAN NOT NULL DEFAULT true,
    position    INTEGER NOT NULL DEFAULT 0,

    -- Conditions (all nullable -- NULL = don't check)
    sender_domain      TEXT,
    sender_contains    TEXT,
    subject_contains   TEXT,
    category           TEXT,
    has_unsubscribe    BOOLEAN,
    priority           TEXT,
    replyable          BOOLEAN,
    is_known_contact   BOOLEAN,

    -- Actions
    set_priority       TEXT,
    set_category       TEXT,
    set_replyable      BOOLEAN,
    label              TEXT,
    skip_llm           BOOLEAN NOT NULL DEFAULT false,
    skip_notify        BOOLEAN NOT NULL DEFAULT false,
    archive            BOOLEAN NOT NULL DEFAULT false,

    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_inbox_rules_enabled_position
    ON email_inbox_rules (enabled, position);

-- Track which rule matched on processed_emails
ALTER TABLE processed_emails
    ADD COLUMN IF NOT EXISTS inbox_rule_id UUID,
    ADD COLUMN IF NOT EXISTS inbox_rule_label TEXT;
