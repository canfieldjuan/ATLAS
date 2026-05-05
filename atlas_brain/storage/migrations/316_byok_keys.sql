-- BYOK (Bring Your Own Keys) provider key storage (PR-D5).
--
-- Customers configure their own Anthropic / OpenRouter / Together /
-- Groq keys in the dashboard; the LLM Gateway router (PR-D4) reads
-- this table per request to proxy with the customer's credentials.
--
-- Encryption-at-rest model: Fernet (AES-128-CBC + HMAC-SHA256) using
-- a server-wide KEK from SaaSAuthConfig.byok_encryption_kek. The
-- ``encryption_kid`` column tags each row with the KEK ID at write
-- time so we can rotate KEKs without re-encrypting every row at once.
-- ``MultiFernet`` accepts a list of keys and decrypts with whichever
-- matches the kid, while writing fresh rows under the latest kid.
--
-- ``key_prefix`` is the first 8 chars of the raw provider key for
-- display ("sk-ant-x..." in the customer dashboard); never use it
-- for auth.
--
-- Soft-delete via revoked_at; never DELETE rows so we keep audit trail
-- (last_used_at) for billing reconciliation.

CREATE TABLE IF NOT EXISTS byok_keys (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id      UUID NOT NULL REFERENCES saas_accounts(id) ON DELETE CASCADE,
    provider        VARCHAR(32) NOT NULL,
    encrypted_key   BYTEA NOT NULL,
    encryption_kid  VARCHAR(64) NOT NULL,
    key_prefix      VARCHAR(16) NOT NULL,
    label           VARCHAR(128) NOT NULL DEFAULT '',
    added_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_used_at    TIMESTAMPTZ,
    revoked_at      TIMESTAMPTZ
);

-- Lookup path: PR-D4's gateway resolver queries by (account_id, provider)
-- with revoked_at IS NULL. The partial index covers exactly that filter.
CREATE INDEX IF NOT EXISTS idx_byok_keys_account_provider_active
    ON byok_keys (account_id, provider)
    WHERE revoked_at IS NULL;

-- One active key per (account, provider). Customers can revoke and
-- re-add; the unique constraint only counts rows where revoked_at IS NULL.
CREATE UNIQUE INDEX IF NOT EXISTS uq_byok_keys_one_active_per_provider
    ON byok_keys (account_id, provider)
    WHERE revoked_at IS NULL;
