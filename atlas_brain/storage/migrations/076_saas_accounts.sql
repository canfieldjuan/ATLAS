-- SaaS multi-tenant accounts and users for consumer intelligence dashboard
CREATE TABLE IF NOT EXISTS saas_accounts (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name                    VARCHAR(256) NOT NULL,
    plan                    VARCHAR(32) NOT NULL DEFAULT 'trial',
    plan_status             VARCHAR(32) NOT NULL DEFAULT 'trialing',
    stripe_customer_id      VARCHAR(128) UNIQUE,
    stripe_subscription_id  VARCHAR(128) UNIQUE,
    trial_ends_at           TIMESTAMPTZ,
    asin_limit              INT NOT NULL DEFAULT 5,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS saas_users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id      UUID NOT NULL REFERENCES saas_accounts(id) ON DELETE CASCADE,
    email           VARCHAR(256) NOT NULL UNIQUE,
    password_hash   VARCHAR(256) NOT NULL,
    full_name       VARCHAR(256),
    role            VARCHAR(32) NOT NULL DEFAULT 'member',
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    last_login_at   TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_saas_users_account ON saas_users(account_id);
CREATE INDEX IF NOT EXISTS idx_saas_users_email ON saas_users(email);
CREATE INDEX IF NOT EXISTS idx_saas_accounts_stripe_customer ON saas_accounts(stripe_customer_id);
