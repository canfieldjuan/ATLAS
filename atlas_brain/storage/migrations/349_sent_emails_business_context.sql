-- Make sent-email history tenant-addressable without guessing legacy ownership.

ALTER TABLE sent_emails
    ADD COLUMN IF NOT EXISTS business_context_id VARCHAR(64);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'chk_sent_emails_business_context_nonblank'
          AND conrelid = 'sent_emails'::regclass
    ) THEN
        ALTER TABLE sent_emails
            ADD CONSTRAINT chk_sent_emails_business_context_nonblank
            CHECK (
                business_context_id IS NULL
                OR btrim(business_context_id) <> ''
            );
    END IF;
END
$$;

CREATE INDEX IF NOT EXISTS idx_sent_emails_context_sent_at
    ON sent_emails (business_context_id, sent_at DESC)
    WHERE business_context_id IS NOT NULL;
