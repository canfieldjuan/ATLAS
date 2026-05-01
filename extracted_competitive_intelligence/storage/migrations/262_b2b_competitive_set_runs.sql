CREATE TABLE IF NOT EXISTS b2b_competitive_set_runs (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    competitive_set_id  UUID NOT NULL REFERENCES b2b_competitive_sets(id) ON DELETE CASCADE,
    account_id          UUID NOT NULL,
    run_id              TEXT NOT NULL,
    trigger             TEXT NOT NULL DEFAULT 'manual',
    status              TEXT NOT NULL DEFAULT 'running',
    execution_id        TEXT,
    summary             JSONB NOT NULL DEFAULT '{}'::jsonb,
    started_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at        TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_b2b_competitive_set_runs_scope_run
    ON b2b_competitive_set_runs (competitive_set_id, run_id);

CREATE INDEX IF NOT EXISTS idx_b2b_competitive_set_runs_scope_started
    ON b2b_competitive_set_runs (competitive_set_id, started_at DESC);

CREATE INDEX IF NOT EXISTS idx_b2b_competitive_set_runs_account_started
    ON b2b_competitive_set_runs (account_id, started_at DESC);
