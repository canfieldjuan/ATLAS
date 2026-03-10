-- Phase 5 Sprint 3: Slack + Teams notification channels
--
-- Adds channel column to webhook subscriptions, enabling channel-specific
-- payload formatting (Slack Block Kit, Teams Adaptive Cards) alongside
-- the existing generic JSON webhooks.

ALTER TABLE b2b_webhook_subscriptions
    ADD COLUMN IF NOT EXISTS channel TEXT NOT NULL DEFAULT 'generic'
        CHECK (channel IN ('generic', 'slack', 'teams'));
