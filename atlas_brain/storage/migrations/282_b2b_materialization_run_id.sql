-- Stamp vendor-level churn materializations with the producing task run.
-- This lets downstream readers and audits verify that scorecard and
-- evidence-vault rows came from the same intelligence execution.

ALTER TABLE b2b_evidence_vault
    ADD COLUMN IF NOT EXISTS materialization_run_id TEXT;

CREATE INDEX IF NOT EXISTS idx_b2b_evidence_vault_materialization_run_id
    ON b2b_evidence_vault (materialization_run_id)
    WHERE materialization_run_id IS NOT NULL;

ALTER TABLE b2b_churn_signals
    ADD COLUMN IF NOT EXISTS materialization_run_id TEXT;

CREATE INDEX IF NOT EXISTS idx_b2b_churn_signals_materialization_run_id
    ON b2b_churn_signals (materialization_run_id)
    WHERE materialization_run_id IS NOT NULL;
