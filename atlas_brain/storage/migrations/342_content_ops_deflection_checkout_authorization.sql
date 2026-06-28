-- Persist the checkout terms authorized for each paid-gated deflection report.
-- The Stripe webhook uses these nullable columns to bind completion to the
-- per-report price selected before Checkout Session creation.

ALTER TABLE content_ops_deflection_reports
    ADD COLUMN IF NOT EXISTS checkout_price_variant TEXT,
    ADD COLUMN IF NOT EXISTS checkout_amount_cents INTEGER,
    ADD COLUMN IF NOT EXISTS checkout_currency TEXT,
    ADD COLUMN IF NOT EXISTS checkout_price_id TEXT,
    ADD COLUMN IF NOT EXISTS checkout_authorized_at TIMESTAMPTZ;
