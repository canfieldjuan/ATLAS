-- Vendor firmographics mapped from org enrichment for reasoning features.

CREATE TABLE IF NOT EXISTS b2b_vendor_firmographics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vendor_name TEXT NOT NULL,
    vendor_name_norm TEXT NOT NULL,
    company_name_raw TEXT,
    company_name_norm TEXT,
    org_cache_id UUID REFERENCES prospect_org_cache(id) ON DELETE SET NULL,
    domain TEXT,
    industry TEXT,
    employee_count INT,
    annual_revenue_range TEXT,
    source TEXT NOT NULL DEFAULT 'prospect_org_cache',
    match_confidence NUMERIC(3,2) NOT NULL DEFAULT 1.0,
    last_synced_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT b2b_vendor_firmographics_vendor_name_norm_key UNIQUE (vendor_name_norm)
);

CREATE INDEX IF NOT EXISTS idx_b2b_vendor_firmographics_company_name_norm
    ON b2b_vendor_firmographics (company_name_norm);

CREATE INDEX IF NOT EXISTS idx_b2b_vendor_firmographics_org_cache_id
    ON b2b_vendor_firmographics (org_cache_id);

CREATE TABLE IF NOT EXISTS b2b_vendor_firmographic_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vendor_name TEXT NOT NULL,
    vendor_name_norm TEXT NOT NULL,
    snapshot_date DATE NOT NULL DEFAULT CURRENT_DATE,
    employee_count INT,
    annual_revenue_range TEXT,
    industry TEXT,
    source TEXT NOT NULL DEFAULT 'prospect_org_cache',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT b2b_vendor_firmographic_snapshots_vendor_name_norm_snapshot_date_key
        UNIQUE (vendor_name_norm, snapshot_date)
);

CREATE INDEX IF NOT EXISTS idx_b2b_vendor_firmographic_snapshots_vendor_date
    ON b2b_vendor_firmographic_snapshots (vendor_name_norm, snapshot_date DESC);
