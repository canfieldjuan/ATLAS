-- Allow a pending idempotency reservation before the external Zendesk macro id
-- exists. This closes the create-then-persist retry window: a retry that sees a
-- pending row must reconcile instead of creating a duplicate macro.

ALTER TABLE ticket_faq_macro_writebacks
    ADD COLUMN IF NOT EXISTS publish_status TEXT NOT NULL DEFAULT 'published';

ALTER TABLE ticket_faq_macro_writebacks
    DROP CONSTRAINT IF EXISTS chk_ticket_faq_macro_writebacks_external_id;

ALTER TABLE ticket_faq_macro_writebacks
    ADD CONSTRAINT chk_ticket_faq_macro_writebacks_publish_status
    CHECK (publish_status IN ('pending', 'published'));

ALTER TABLE ticket_faq_macro_writebacks
    ADD CONSTRAINT chk_ticket_faq_macro_writebacks_external_id_when_published
    CHECK (publish_status <> 'published' OR btrim(external_id) <> '');

ALTER TABLE ticket_faq_macro_writebacks
    DROP CONSTRAINT IF EXISTS ticket_faq_macro_writebacks_account_id_platform_external_id_key;

CREATE UNIQUE INDEX IF NOT EXISTS idx_ticket_faq_macro_writebacks_external_id_unique
    ON ticket_faq_macro_writebacks (account_id, platform, external_id)
    WHERE btrim(external_id) <> '';
