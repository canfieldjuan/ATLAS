-- Configurable card templates for account-level intelligence in vendor briefings.

CREATE TABLE IF NOT EXISTS card_templates (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name             TEXT UNIQUE NOT NULL,
    label            TEXT NOT NULL,
    required_fields  TEXT[] NOT NULL DEFAULT '{}',
    optional_fields  TEXT[] NOT NULL DEFAULT '{}',
    prompt_template  TEXT NOT NULL DEFAULT '',
    reasoning_depth  INT NOT NULL DEFAULT 0,
    enabled          BOOLEAN NOT NULL DEFAULT true,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_card_templates_enabled
    ON card_templates (enabled) WHERE enabled = true;

-- Seed: sales_action card (depth 2 enrichment)
INSERT INTO card_templates (name, label, required_fields, optional_fields, prompt_template, reasoning_depth)
VALUES (
    'sales_action',
    'Sales Action Card',
    ARRAY['company', 'urgency', 'vendor_name', 'pain_breakdown'],
    ARRAY['evidence', 'top_displacement_targets', 'top_feature_gaps', 'budget_context'],
    'You are a B2B sales intelligence analyst. Given the following churn data about {company} (a customer of {vendor_name}), produce a JSON object with exactly these keys:

1. "situation": 1-2 sentences connecting the data points -- what is happening with this account and why.
2. "approach": What the sales rep should lead with in outreach and why. Be specific.
3. "why_now": Why this account is actionable right now (timing, urgency, contract signals).
4. "urgency_label": One of "critical", "high", "moderate" based on the data.
5. "talking_points": Array of 2-3 short bullet points for a sales conversation.

Ground every statement in the data below. Do not speculate beyond what the data supports.

Account data:
- Company: {company}
- Urgency score: {urgency}/10
- Vendor (incumbent): {vendor_name}
- Pain drivers: {pain_breakdown}
- Evidence quotes: {evidence}
- Competitors gaining: {top_displacement_targets}
- Feature gaps: {top_feature_gaps}
- Budget context: {budget_context}

Return ONLY valid JSON. No markdown fences. No explanation outside the JSON.',
    2
)
ON CONFLICT (name) DO NOTHING;
