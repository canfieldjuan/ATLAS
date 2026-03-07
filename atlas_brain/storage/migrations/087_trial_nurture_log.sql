CREATE TABLE IF NOT EXISTS trial_nurture_log (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id  UUID NOT NULL REFERENCES saas_accounts(id) ON DELETE CASCADE,
    milestone   VARCHAR(30) NOT NULL,
    sent_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(account_id, milestone)
);

CREATE INDEX IF NOT EXISTS idx_trial_nurture_account ON trial_nurture_log(account_id);
