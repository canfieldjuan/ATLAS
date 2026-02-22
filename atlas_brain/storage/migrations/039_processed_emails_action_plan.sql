ALTER TABLE processed_emails
    ADD COLUMN IF NOT EXISTS action_plan JSONB;

ALTER TABLE processed_emails
    ADD COLUMN IF NOT EXISTS customer_context_summary TEXT;

COMMENT ON COLUMN processed_emails.action_plan IS
    'LLM-generated action plan JSON for CRM-matched emails; NULL for unmatched';

COMMENT ON COLUMN processed_emails.customer_context_summary IS
    'One-line customer context string used in ntfy notification';
