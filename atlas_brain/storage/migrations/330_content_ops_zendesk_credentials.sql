-- Tenant-scoped Zendesk credentials for Content Ops FAQ macro writeback.
--
-- API tokens are encrypted at rest with the existing BYOK Fernet KEK
-- (SaaSAuthConfig.byok_encryption_kek). Store only a token prefix for
-- display; never expose encrypted_api_token through list/display DTOs.

CREATE TABLE IF NOT EXISTS content_ops_zendesk_credentials (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id          UUID NOT NULL REFERENCES saas_accounts(id) ON DELETE CASCADE,
    email               TEXT NOT NULL,
    encrypted_api_token BYTEA NOT NULL,
    encryption_kid      VARCHAR(64) NOT NULL,
    api_token_prefix    VARCHAR(16) NOT NULL,
    subdomain           TEXT NOT NULL DEFAULT '',
    base_url            TEXT NOT NULL DEFAULT '',
    label               VARCHAR(128) NOT NULL DEFAULT '',
    added_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_used_at        TIMESTAMPTZ,
    revoked_at          TIMESTAMPTZ,
    CONSTRAINT chk_content_ops_zendesk_credentials_email
        CHECK (btrim(email) <> ''),
    CONSTRAINT chk_content_ops_zendesk_credentials_token_prefix
        CHECK (btrim(api_token_prefix) <> ''),
    CONSTRAINT chk_content_ops_zendesk_credentials_endpoint
        CHECK (btrim(subdomain) <> '' OR btrim(base_url) <> '')
);

CREATE INDEX IF NOT EXISTS idx_content_ops_zendesk_credentials_account_active
    ON content_ops_zendesk_credentials (account_id, added_at DESC)
    WHERE revoked_at IS NULL;

CREATE UNIQUE INDEX IF NOT EXISTS uq_content_ops_zendesk_credentials_one_active
    ON content_ops_zendesk_credentials (account_id)
    WHERE revoked_at IS NULL;
