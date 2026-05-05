-- LLM Gateway API keys (PR-D1).
--
-- Long-lived bearer tokens that customer scripts use to call the
-- /api/v1/llm/* endpoints. JWT-from-/auth/login is for the dashboard;
-- API keys are for production scripts. Both auth methods resolve to
-- the same AuthUser shape so downstream endpoints are auth-method
-- agnostic.
--
-- Hashing model: HMAC-SHA256(server_pepper, raw_key) -> hex.
-- raw_key is 32 random base32 characters (~160 bits entropy), so a
-- KDF (bcrypt/PBKDF2) is unnecessary. The pepper is a server-wide
-- secret in SaaSAuthConfig.api_key_pepper, never per-key salt.
--
-- key_prefix stores the first 8 chars of the raw key for fast lookup
-- on the verify path (narrow candidate set, then HMAC-compare).
--
-- Soft-delete via revoked_at; never DELETE rows so audit trail
-- (last_used_at, last_used_ip) survives revocation.

CREATE TABLE IF NOT EXISTS api_keys (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id      UUID NOT NULL REFERENCES saas_accounts(id) ON DELETE CASCADE,
    user_id         UUID REFERENCES saas_users(id) ON DELETE SET NULL,
    name            VARCHAR(128) NOT NULL,
    key_prefix      VARCHAR(24) NOT NULL,
    key_hash        VARCHAR(128) NOT NULL,
    scopes          TEXT[] NOT NULL DEFAULT ARRAY['llm:*'],
    last_used_at    TIMESTAMPTZ,
    last_used_ip    INET,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    revoked_at      TIMESTAMPTZ
);

-- Lookup path: fetch active keys by account, then by prefix on verify.
CREATE INDEX IF NOT EXISTS idx_api_keys_account_active
    ON api_keys (account_id)
    WHERE revoked_at IS NULL;

-- Verify path narrows by prefix (~16 distinct prefixes/account in practice)
-- then HMAC-compares full key against candidates.
CREATE INDEX IF NOT EXISTS idx_api_keys_prefix_active
    ON api_keys (key_prefix)
    WHERE revoked_at IS NULL;
