-- Keep the EOM lead review queue indexed after the onboarding email is sent.
--
-- A lead in onboarding_sent has received Juan-approved onboarding email but is
-- still not a Customer/Site handoff. The review queue and later onboarding
-- completion slice must be able to see it.
--
-- Rollback evidence:
--   If application code is rolled back before it knows about onboarding_sent,
--   restore the prior predicate by recreating this index with:
--     lead_stage IN ('new', 'estimate_booked', 'won')
--   Do not rewrite persisted onboarding_sent contacts during a database-only
--   rollback; instead roll the application forward to a build that admits and
--   reconciles onboarding_sent, or manually review those leads before applying
--   the prior predicate so they are not hidden from the office queue.

DROP INDEX CONCURRENTLY IF EXISTS idx_contacts_eom_lead_review_queue;

CREATE INDEX CONCURRENTLY idx_contacts_eom_lead_review_queue
    ON contacts (created_at DESC, id DESC)
    WHERE business_context_id = 'effingham_maids'
      AND status = 'active'
      AND contact_type = 'lead'
      AND lead_stage IN ('new', 'estimate_booked', 'won', 'onboarding_sent');
