-- Migration 235: Add additive account ownership to vendor_targets

ALTER TABLE vendor_targets
    ADD COLUMN IF NOT EXISTS account_id UUID REFERENCES saas_accounts(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_vendor_targets_account
    ON vendor_targets(account_id);

CREATE INDEX IF NOT EXISTS idx_vendor_targets_account_mode_status
    ON vendor_targets(account_id, target_mode, status);
