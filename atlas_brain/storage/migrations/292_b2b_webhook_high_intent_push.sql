-- Make high_intent_push a managed CRM webhook event without changing existing CRM delivery defaults.
UPDATE b2b_webhook_subscriptions
SET event_types = event_types || ARRAY['high_intent_push']::text[],
    updated_at = NOW()
WHERE COALESCE(channel, 'generic') LIKE 'crm_%'
  AND NOT ('high_intent_push' = ANY(event_types));
