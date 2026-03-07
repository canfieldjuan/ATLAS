-- Apollo.io prospect pipeline tables.
-- prospect_org_cache: one row per company, caches Apollo org enrichment (1 credit each).
-- prospects: one row per decision-maker with verified/probabilistic email.

CREATE TABLE IF NOT EXISTS prospect_org_cache (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_name_raw    TEXT NOT NULL,
    company_name_norm   TEXT NOT NULL,
    apollo_org_id       TEXT,
    domain              TEXT,
    industry            TEXT,
    employee_count      INT,
    annual_revenue_range TEXT,
    tech_stack          JSONB DEFAULT '[]',
    status              TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'enriched', 'not_found', 'error')),
    error_detail        TEXT,
    enriched_at         TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_prospect_org_cache_norm
    ON prospect_org_cache (company_name_norm);
CREATE INDEX IF NOT EXISTS idx_prospect_org_cache_status
    ON prospect_org_cache (status);

-- -------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS prospects (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    apollo_person_id    TEXT,
    first_name          TEXT,
    last_name           TEXT,
    email               TEXT,
    email_status        TEXT,
    title               TEXT,
    seniority           TEXT,
    department          TEXT,
    linkedin_url        TEXT,
    city                TEXT,
    state               TEXT,
    country             TEXT,
    company_name        TEXT,
    company_domain      TEXT,
    company_name_norm   TEXT,
    org_cache_id        UUID REFERENCES prospect_org_cache(id) ON DELETE SET NULL,
    status              TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'contacted', 'opted_out', 'bounced', 'suppressed')),
    raw_person_response JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_prospects_apollo_person
    ON prospects (apollo_person_id) WHERE apollo_person_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_prospects_email_unique
    ON prospects (LOWER(email)) WHERE email IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_prospects_company_norm
    ON prospects (company_name_norm);
CREATE INDEX IF NOT EXISTS idx_prospects_status
    ON prospects (status);
