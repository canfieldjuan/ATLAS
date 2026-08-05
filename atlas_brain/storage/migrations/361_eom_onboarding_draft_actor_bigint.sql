-- atlas: atomic-bookkeeping
-- Widen the onboarding-draft approver id to the funnel actor boundary.
--
-- The funnel actor guard admits X-EOM-Actor-ID up to signed 64-bit and the
-- customer-handoff table already stores its approver as BIGINT (migration
-- 353). Migration 360 declared approved_by_employee_id as INT, so a valid
-- authenticated approval with an actor id above 2147483647 would pass the
-- HTTP boundary and then fail inside Postgres during the claim instead of
-- taking the promised 201/200 path. Widening to BIGINT aligns the column
-- with both the boundary and the handoff precedent.
--
-- Rollback evidence:
--   ALTER TABLE eom_onboarding_email_drafts
--       ALTER COLUMN approved_by_employee_id TYPE INTEGER;
--   (Safe while every stored value fits int4; the column is freshly
--   deployed and empty in practice.)
--
-- Roll-forward safety:
--   Widening INT -> BIGINT is value-preserving; older application code
--   reads and writes the column unchanged.

ALTER TABLE eom_onboarding_email_drafts
    ALTER COLUMN approved_by_employee_id TYPE BIGINT;
