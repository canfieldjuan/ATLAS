-- Migration 234: Track the source of each tracked vendor entry

CREATE TABLE IF NOT EXISTS tracked_vendor_sources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id UUID NOT NULL REFERENCES saas_accounts(id) ON DELETE CASCADE,
    vendor_name TEXT NOT NULL,
    source_type VARCHAR(32) NOT NULL,
    source_key TEXT NOT NULL,
    track_mode VARCHAR(32) NOT NULL DEFAULT 'competitor',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(account_id, vendor_name, source_type, source_key)
);

CREATE INDEX IF NOT EXISTS idx_tracked_vendor_sources_account
    ON tracked_vendor_sources(account_id);

CREATE INDEX IF NOT EXISTS idx_tracked_vendor_sources_vendor
    ON tracked_vendor_sources(vendor_name);

CREATE INDEX IF NOT EXISTS idx_tracked_vendor_sources_lookup
    ON tracked_vendor_sources(account_id, source_type, source_key);

INSERT INTO tracked_vendor_sources (
    account_id,
    vendor_name,
    source_type,
    source_key,
    track_mode,
    created_at,
    updated_at
)
SELECT
    tv.account_id,
    tv.vendor_name,
    'manual',
    'legacy_import',
    tv.track_mode,
    COALESCE(tv.added_at, NOW()),
    NOW()
FROM tracked_vendors tv
ON CONFLICT (account_id, vendor_name, source_type, source_key) DO NOTHING;
