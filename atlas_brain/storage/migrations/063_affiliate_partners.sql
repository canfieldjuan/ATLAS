CREATE TABLE IF NOT EXISTS affiliate_partners (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT NOT NULL,
    product_name    TEXT NOT NULL,
    product_aliases TEXT[] NOT NULL DEFAULT '{}',
    category        TEXT,
    affiliate_url   TEXT NOT NULL,
    commission_type TEXT NOT NULL DEFAULT 'unknown',
    commission_value TEXT,
    notes           TEXT,
    enabled         BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_affiliate_partners_product
    ON affiliate_partners (LOWER(product_name));

CREATE TABLE IF NOT EXISTS affiliate_clicks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    partner_id      UUID NOT NULL REFERENCES affiliate_partners(id) ON DELETE CASCADE,
    review_id       UUID REFERENCES b2b_reviews(id) ON DELETE SET NULL,
    referrer        TEXT,
    clicked_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_affiliate_clicks_partner
    ON affiliate_clicks (partner_id, clicked_at DESC);
