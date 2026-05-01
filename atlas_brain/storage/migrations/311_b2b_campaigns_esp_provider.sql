-- Provider-scope esp_message_id so multi-ESP webhooks cannot
-- cross-match.
--
-- The webhook handler in atlas_brain/api/campaign_webhooks.py looks up
-- campaigns by esp_message_id alone. Once a second ESP plugin (SES,
-- SendGrid, Postmark, Mailgun) goes live, two providers can legitimately
-- emit the same identifier value/format; a Resend webhook delivering an
-- event for "abc123" could update an SES-sent campaign that also has
-- esp_message_id='abc123' (and vice versa), corrupting opens/clicks/
-- bounces and triggering the wrong suppression cascade.
--
-- This migration adds an esp_provider column populated by campaign_send
-- when it records esp_message_id, backfills existing rows to 'resend'
-- (the only provider in production today), and replaces the single-
-- column lookup index with a composite (esp_provider, esp_message_id)
-- index that satisfies the new provider-scoped lookup. The webhook
-- handler matches on (esp_message_id, esp_provider) so a webhook from
-- one provider can never accidentally update a campaign sent through
-- another.

ALTER TABLE b2b_campaigns
    ADD COLUMN IF NOT EXISTS esp_provider TEXT;

UPDATE b2b_campaigns
SET esp_provider = 'resend'
WHERE esp_message_id IS NOT NULL
  AND esp_provider IS NULL;

DROP INDEX IF EXISTS idx_b2b_campaigns_esp_msg;

CREATE INDEX IF NOT EXISTS idx_b2b_campaigns_provider_esp_msg
    ON b2b_campaigns (esp_provider, esp_message_id)
    WHERE esp_message_id IS NOT NULL;
