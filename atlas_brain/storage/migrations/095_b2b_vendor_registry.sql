-- 095: Canonical vendor registry
-- Normalizes vendor_name TEXT across all B2B tables via a registry of
-- canonical names and aliases.  No FK migration -- just consistent TEXT values.

-- 1. Create registry table
CREATE TABLE IF NOT EXISTS b2b_vendors (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    canonical_name  TEXT NOT NULL UNIQUE,
    aliases         JSONB NOT NULL DEFAULT '[]',
    metadata        JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_b2b_vendors_aliases ON b2b_vendors USING GIN (aliases);
CREATE INDEX IF NOT EXISTS idx_b2b_vendors_canonical_lower ON b2b_vendors (LOWER(canonical_name));

-- 2. Seed from existing data (distinct vendor_name across reviews + targets)
INSERT INTO b2b_vendors (canonical_name)
SELECT DISTINCT vendor_name
FROM (
    SELECT vendor_name FROM b2b_scrape_targets WHERE vendor_name IS NOT NULL AND vendor_name <> ''
    UNION
    SELECT vendor_name FROM b2b_reviews WHERE vendor_name IS NOT NULL AND vendor_name <> ''
) AS src
ON CONFLICT (canonical_name) DO NOTHING;

-- 3. Seed well-known aliases from the hardcoded _B2B_COMPETITOR_ALIASES dict.
--    Each canonical vendor gets its aliases merged in.
INSERT INTO b2b_vendors (canonical_name, aliases) VALUES
    ('Google Cloud Platform', '["gcp","google cloud"]'),
    ('Amazon Web Services',   '["aws","amazon web services"]'),
    ('Microsoft Teams',       '["ms teams"]'),
    ('Microsoft 365',         '["ms 365","office 365","o365"]'),
    ('Salesforce',            '["sf","sfdc"]'),
    ('HubSpot',               '["hubspot crm"]'),
    ('Google Workspace',      '["g suite","google workspace","gsuite"]')
ON CONFLICT (canonical_name) DO UPDATE SET
    aliases    = EXCLUDED.aliases,
    updated_at = NOW();

-- 4. Normalize existing vendor_name values across the 9 writable tables.
--    Two passes per table: (a) exact alias match, (b) case-insensitive canonical match.

-- Helper: temp mapping of lowered alias -> canonical_name
CREATE TEMP TABLE _vendor_alias_map AS
SELECT
    v.canonical_name,
    LOWER(alias.val) AS alias_lower
FROM b2b_vendors v,
     jsonb_array_elements_text(v.aliases) AS alias(val);

CREATE INDEX ON _vendor_alias_map (alias_lower);

-- 4a. Update via alias match (e.g. "gcp" -> "Google Cloud Platform")
UPDATE b2b_reviews r SET vendor_name = m.canonical_name
FROM _vendor_alias_map m WHERE LOWER(r.vendor_name) = m.alias_lower AND r.vendor_name <> m.canonical_name;

UPDATE b2b_scrape_targets r SET vendor_name = m.canonical_name
FROM _vendor_alias_map m WHERE LOWER(r.vendor_name) = m.alias_lower AND r.vendor_name <> m.canonical_name;

UPDATE b2b_churn_signals r SET vendor_name = m.canonical_name
FROM _vendor_alias_map m WHERE LOWER(r.vendor_name) = m.alias_lower AND r.vendor_name <> m.canonical_name;

UPDATE b2b_product_profiles r SET vendor_name = m.canonical_name
FROM _vendor_alias_map m WHERE LOWER(r.vendor_name) = m.alias_lower AND r.vendor_name <> m.canonical_name;

UPDATE b2b_campaigns r SET vendor_name = m.canonical_name
FROM _vendor_alias_map m WHERE LOWER(r.vendor_name) = m.alias_lower AND r.vendor_name <> m.canonical_name;

UPDATE b2b_alert_baselines r SET vendor_name = m.canonical_name
FROM _vendor_alias_map m WHERE LOWER(r.vendor_name) = m.alias_lower AND r.vendor_name <> m.canonical_name;

UPDATE b2b_keyword_signals r SET vendor_name = m.canonical_name
FROM _vendor_alias_map m WHERE LOWER(r.vendor_name) = m.alias_lower AND r.vendor_name <> m.canonical_name;

UPDATE b2b_vendor_briefings r SET vendor_name = m.canonical_name
FROM _vendor_alias_map m WHERE LOWER(r.vendor_name) = m.alias_lower AND r.vendor_name <> m.canonical_name;

UPDATE tracked_vendors r SET vendor_name = m.canonical_name
FROM _vendor_alias_map m WHERE LOWER(r.vendor_name) = m.alias_lower AND r.vendor_name <> m.canonical_name;

-- 4b. Update case-insensitive canonical match (e.g. "salesforce" -> "Salesforce")
UPDATE b2b_reviews r SET vendor_name = v.canonical_name
FROM b2b_vendors v WHERE LOWER(r.vendor_name) = LOWER(v.canonical_name) AND r.vendor_name <> v.canonical_name;

UPDATE b2b_scrape_targets r SET vendor_name = v.canonical_name
FROM b2b_vendors v WHERE LOWER(r.vendor_name) = LOWER(v.canonical_name) AND r.vendor_name <> v.canonical_name;

UPDATE b2b_churn_signals r SET vendor_name = v.canonical_name
FROM b2b_vendors v WHERE LOWER(r.vendor_name) = LOWER(v.canonical_name) AND r.vendor_name <> v.canonical_name;

UPDATE b2b_product_profiles r SET vendor_name = v.canonical_name
FROM b2b_vendors v WHERE LOWER(r.vendor_name) = LOWER(v.canonical_name) AND r.vendor_name <> v.canonical_name;

UPDATE b2b_campaigns r SET vendor_name = v.canonical_name
FROM b2b_vendors v WHERE LOWER(r.vendor_name) = LOWER(v.canonical_name) AND r.vendor_name <> v.canonical_name;

UPDATE b2b_alert_baselines r SET vendor_name = v.canonical_name
FROM b2b_vendors v WHERE LOWER(r.vendor_name) = LOWER(v.canonical_name) AND r.vendor_name <> v.canonical_name;

UPDATE b2b_keyword_signals r SET vendor_name = v.canonical_name
FROM b2b_vendors v WHERE LOWER(r.vendor_name) = LOWER(v.canonical_name) AND r.vendor_name <> v.canonical_name;

UPDATE b2b_vendor_briefings r SET vendor_name = v.canonical_name
FROM b2b_vendors v WHERE LOWER(r.vendor_name) = LOWER(v.canonical_name) AND r.vendor_name <> v.canonical_name;

UPDATE tracked_vendors r SET vendor_name = v.canonical_name
FROM b2b_vendors v WHERE LOWER(r.vendor_name) = LOWER(v.canonical_name) AND r.vendor_name <> v.canonical_name;

DROP TABLE _vendor_alias_map;

-- 5. Refresh materialized view that depends on vendor_name
REFRESH MATERIALIZED VIEW IF EXISTS campaign_funnel_stats;
