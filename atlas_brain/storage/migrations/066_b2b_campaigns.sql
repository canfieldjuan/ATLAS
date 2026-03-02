-- B2B ABM Campaign Engine: campaign storage for generated outreach content
-- Each row is one channel variant (email_cold, linkedin, email_followup) for a target company.

CREATE TABLE IF NOT EXISTS b2b_campaigns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Target
    company_name TEXT NOT NULL,
    vendor_name TEXT NOT NULL,
    product_category TEXT,

    -- Context snapshot at generation time
    opportunity_score INT,
    urgency_score NUMERIC(3,1),
    pain_categories JSONB,
    competitors_considering JSONB,
    seat_count INT,
    contract_end TEXT,
    decision_timeline TEXT,
    buying_stage TEXT,
    role_type TEXT,
    key_quotes JSONB,
    source_review_ids UUID[],

    -- Generated content
    channel TEXT NOT NULL,
    subject TEXT,
    body TEXT NOT NULL,
    cta TEXT,

    -- Workflow
    status TEXT NOT NULL DEFAULT 'draft'
        CHECK (status IN ('draft', 'approved', 'sent', 'expired')),
    batch_id TEXT,
    llm_model TEXT,

    -- Tracking
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    approved_at TIMESTAMPTZ,
    sent_at TIMESTAMPTZ,
    opened_at TIMESTAMPTZ,
    clicked_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_b2b_campaigns_company ON b2b_campaigns (company_name, vendor_name);
CREATE INDEX IF NOT EXISTS idx_b2b_campaigns_status ON b2b_campaigns (status);
CREATE INDEX IF NOT EXISTS idx_b2b_campaigns_batch ON b2b_campaigns (batch_id);
