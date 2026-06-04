-- Tenant-scoped saved brand voice profiles for Content Ops generation.
--
-- The executable request-time brand voice contract lives in the extracted
-- package. This host table stores account-owned profiles so the UI can select
-- a profile id and the API can resolve the full bounded prompt guidance before
-- preview, plan, or execute.

CREATE TABLE IF NOT EXISTS content_ops_brand_voice_profiles (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id     UUID NOT NULL REFERENCES saas_accounts(id) ON DELETE CASCADE,
    name           TEXT NOT NULL,
    descriptors    JSONB NOT NULL DEFAULT '[]'::jsonb,
    exemplars      JSONB NOT NULL DEFAULT '[]'::jsonb,
    banned_terms   JSONB NOT NULL DEFAULT '[]'::jsonb,
    preferred_pov  TEXT,
    reading_level  TEXT,
    metadata       JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    archived_at    TIMESTAMPTZ,
    CONSTRAINT chk_content_ops_brand_voice_profiles_name
        CHECK (btrim(name) <> ''),
    CONSTRAINT chk_content_ops_brand_voice_profiles_guidance
        CHECK (
            jsonb_array_length(descriptors) > 0
            OR jsonb_array_length(exemplars) > 0
            OR jsonb_array_length(banned_terms) > 0
            OR preferred_pov IS NOT NULL
            OR reading_level IS NOT NULL
        )
);

CREATE INDEX IF NOT EXISTS idx_content_ops_brand_voice_profiles_account_active
    ON content_ops_brand_voice_profiles (account_id, updated_at DESC)
    WHERE archived_at IS NULL;

CREATE UNIQUE INDEX IF NOT EXISTS uq_content_ops_brand_voice_profiles_account_name_active
    ON content_ops_brand_voice_profiles (account_id, lower(name))
    WHERE archived_at IS NULL;
