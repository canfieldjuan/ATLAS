-- DB-backed Apollo company override registry.
-- Migration 161

CREATE TABLE IF NOT EXISTS apollo_company_overrides (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_name_raw  TEXT NOT NULL,
    company_name_norm TEXT NOT NULL,
    search_names      JSONB NOT NULL DEFAULT '[]'::jsonb,
    domains           JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT apollo_company_overrides_company_name_norm_key UNIQUE (company_name_norm),
    CONSTRAINT apollo_company_overrides_search_names_array CHECK (jsonb_typeof(search_names) = 'array'),
    CONSTRAINT apollo_company_overrides_domains_array CHECK (jsonb_typeof(domains) = 'array')
);

CREATE INDEX IF NOT EXISTS idx_apollo_company_overrides_norm
    ON apollo_company_overrides (company_name_norm);
