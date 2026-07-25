-- Exact-context OAuth credentials for scoped CRM mailbox reads.
--
-- The encrypted payload is an authenticated Fernet token containing the Gmail
-- client id, client secret, and refresh token.  It deliberately lives outside
-- business_contexts so broad context reads can never project credential data.

CREATE TABLE IF NOT EXISTS scoped_mailbox_credentials (
    business_context_id   VARCHAR(64) NOT NULL,
    provider              VARCHAR(16) NOT NULL,
    encrypted_credentials BYTEA NOT NULL,
    encryption_kid        VARCHAR(64) NOT NULL,
    generation            BIGINT NOT NULL DEFAULT 1,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    revoked_at            TIMESTAMPTZ,
    PRIMARY KEY (business_context_id, provider),
    CONSTRAINT chk_scoped_mailbox_credentials_context_nonblank
        CHECK (btrim(business_context_id) <> ''),
    CONSTRAINT chk_scoped_mailbox_credentials_provider
        CHECK (provider = 'gmail'),
    CONSTRAINT chk_scoped_mailbox_credentials_kid_nonblank
        CHECK (btrim(encryption_kid) <> ''),
    CONSTRAINT chk_scoped_mailbox_credentials_generation_positive
        CHECK (generation > 0)
);

CREATE INDEX IF NOT EXISTS idx_scoped_mailbox_credentials_active
    ON scoped_mailbox_credentials (business_context_id, provider)
    WHERE revoked_at IS NULL;
