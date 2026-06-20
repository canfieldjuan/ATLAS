-- Effective account/platform admin flag used by SaaS auth dependencies.
--
-- `atlas_brain.auth.dependencies.require_auth` reads this column for every
-- authenticated request. Older databases may already have it from manual or
-- out-of-band setup; fresh repo-managed databases need the migration so auth
-- does not fail with a missing-column SQL error.

ALTER TABLE saas_users
    ADD COLUMN IF NOT EXISTS is_admin BOOLEAN NOT NULL DEFAULT FALSE;
