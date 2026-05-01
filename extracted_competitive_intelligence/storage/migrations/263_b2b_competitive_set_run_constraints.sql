DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_b2b_competitive_set_runs_account'
    ) THEN
        ALTER TABLE b2b_competitive_set_runs
        ADD CONSTRAINT fk_b2b_competitive_set_runs_account
        FOREIGN KEY (account_id) REFERENCES saas_accounts(id) ON DELETE CASCADE;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'chk_b2b_competitive_set_runs_trigger'
    ) THEN
        ALTER TABLE b2b_competitive_set_runs
        ADD CONSTRAINT chk_b2b_competitive_set_runs_trigger
        CHECK (trigger IN ('manual', 'scheduled'));
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'chk_b2b_competitive_set_runs_status'
    ) THEN
        ALTER TABLE b2b_competitive_set_runs
        ADD CONSTRAINT chk_b2b_competitive_set_runs_status
        CHECK (status IN ('running', 'succeeded', 'partial', 'failed'));
    END IF;
END $$;
