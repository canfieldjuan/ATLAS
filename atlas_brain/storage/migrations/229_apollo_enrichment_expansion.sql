-- Expand prospect_org_cache and prospects tables to capture all valuable
-- Apollo fields that already exist in raw_person_response JSONB.

-- Org-level: phone, geography, funding, headcount growth, financials
ALTER TABLE prospect_org_cache
    ADD COLUMN IF NOT EXISTS phone text,
    ADD COLUMN IF NOT EXISTS sanitized_phone text,
    ADD COLUMN IF NOT EXISTS city text,
    ADD COLUMN IF NOT EXISTS state text,
    ADD COLUMN IF NOT EXISTS country text,
    ADD COLUMN IF NOT EXISTS founded_year int,
    ADD COLUMN IF NOT EXISTS total_funding text,
    ADD COLUMN IF NOT EXISTS latest_funding_stage text,
    ADD COLUMN IF NOT EXISTS annual_revenue numeric,
    ADD COLUMN IF NOT EXISTS market_cap text,
    ADD COLUMN IF NOT EXISTS publicly_traded_exchange text,
    ADD COLUMN IF NOT EXISTS publicly_traded_symbol text,
    ADD COLUMN IF NOT EXISTS headcount_growth_6m numeric,
    ADD COLUMN IF NOT EXISTS headcount_growth_12m numeric,
    ADD COLUMN IF NOT EXISTS headcount_growth_24m numeric,
    ADD COLUMN IF NOT EXISTS linkedin_url text,
    ADD COLUMN IF NOT EXISTS website_url text,
    ADD COLUMN IF NOT EXISTS short_description text,
    ADD COLUMN IF NOT EXISTS naics_codes jsonb,
    ADD COLUMN IF NOT EXISTS sic_codes jsonb;

-- Person-level: headline, departments, timezone, social
ALTER TABLE prospects
    ADD COLUMN IF NOT EXISTS headline text,
    ADD COLUMN IF NOT EXISTS departments jsonb,
    ADD COLUMN IF NOT EXISTS time_zone text,
    ADD COLUMN IF NOT EXISTS twitter_url text,
    ADD COLUMN IF NOT EXISTS photo_url text,
    ADD COLUMN IF NOT EXISTS facebook_url text;
