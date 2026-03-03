-- 083: B2B product profiles — pre-computed vendor knowledge cards
-- Aggregated from enriched b2b_reviews, one row per vendor.

CREATE TABLE IF NOT EXISTS b2b_product_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vendor_name TEXT NOT NULL,
    product_category TEXT,

    -- What this product is good at (aggregated from positive reviews)
    -- Format: [{"area": "integration_ease", "score": 8.2, "evidence_count": 45}, ...]
    strengths JSONB NOT NULL DEFAULT '[]',

    -- What this product is bad at (aggregated from negative reviews + pain categories)
    -- Format: [{"area": "pricing_transparency", "score": 3.1, "evidence_count": 30}, ...]
    weaknesses JSONB NOT NULL DEFAULT '[]',

    -- Pain categories this product ADDRESSES well (inverted from competitors' pain)
    -- Format: {"integration_complexity": 0.85, "poor_support": 0.72, ...}
    -- Score 0-1: how well this vendor solves that pain
    pain_addressed JSONB NOT NULL DEFAULT '{}',

    -- Aggregate metrics
    total_reviews_analyzed INT NOT NULL DEFAULT 0,
    avg_rating NUMERIC(3,2),
    recommend_rate NUMERIC(3,2),
    avg_urgency NUMERIC(3,1),

    -- Use case fit
    -- Format: [{"use_case": "marketing_automation", "fit_score": 0.9}, ...]
    primary_use_cases JSONB NOT NULL DEFAULT '[]',

    -- Company fit signals
    -- Format: {"1-50": 0.35, "51-200": 0.40, "201-1000": 0.20, "1000+": 0.05}
    typical_company_size JSONB NOT NULL DEFAULT '{}',

    -- Format: [{"industry": "SaaS", "pct": 0.30}, ...]
    typical_industries JSONB NOT NULL DEFAULT '[]',

    -- Integration ecosystem
    -- Format: ["Salesforce", "Zapier", "HubSpot"]
    top_integrations JSONB NOT NULL DEFAULT '[]',

    -- Competitive positioning
    -- Format: [{"vendor": "HubSpot", "win_rate": 0.6, "mentions": 25}, ...]
    commonly_compared_to JSONB NOT NULL DEFAULT '[]',

    -- Format: [{"vendor": "Salesforce", "count": 15, "top_reason": "pricing"}, ...]
    commonly_switched_from JSONB NOT NULL DEFAULT '[]',

    -- LLM-generated summary (one paragraph, human-readable)
    profile_summary TEXT,

    last_computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Expression-based unique index (required for ON CONFLICT with COALESCE)
CREATE UNIQUE INDEX IF NOT EXISTS idx_b2b_product_profiles_vendor_category
    ON b2b_product_profiles (vendor_name, COALESCE(product_category, ''));

CREATE INDEX IF NOT EXISTS idx_b2b_product_profiles_vendor
    ON b2b_product_profiles (vendor_name);

CREATE INDEX IF NOT EXISTS idx_b2b_product_profiles_category
    ON b2b_product_profiles (product_category)
    WHERE product_category IS NOT NULL;
