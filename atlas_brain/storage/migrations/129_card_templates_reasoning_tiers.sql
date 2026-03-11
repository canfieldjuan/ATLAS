-- Tiered reasoning system for account cards: depth 0/1/2.
-- Adds target_mode and system_prompt columns, seeds CoT and multi-pass templates.

ALTER TABLE card_templates
    ADD COLUMN IF NOT EXISTS target_mode TEXT NOT NULL DEFAULT 'both',
    ADD COLUMN IF NOT EXISTS system_prompt TEXT NOT NULL DEFAULT '';

-- Update existing sales_action template to include new reviewer metadata fields
UPDATE card_templates
SET required_fields = ARRAY['company', 'urgency', 'vendor_name', 'pain_breakdown',
                            'title', 'company_size', 'industry'],
    optional_fields = ARRAY['evidence', 'top_displacement_targets', 'top_feature_gaps',
                            'budget_context', 'persona', 'company_tier', 'urgency_label']
WHERE name = 'sales_action';

-- Seed: depth 1 Chain-of-Thought template
INSERT INTO card_templates (name, label, required_fields, optional_fields,
    prompt_template, reasoning_depth, target_mode, system_prompt)
VALUES (
    'sales_action_cot',
    'Sales Action Card (Analyzed)',
    ARRAY['company','urgency','vendor_name','pain_breakdown','title','company_size','industry'],
    ARRAY['evidence','top_displacement_targets','top_feature_gaps','budget_context',
          'persona','company_tier','urgency_label','target_mode'],
    'Analyze this B2B account step by step, then produce a final intelligence card.

STEP 1 - PERSONA ANALYSIS: What does a {persona} at a {company_tier} {industry} company care about? What retention/growth levers matter to this role?

STEP 2 - SIGNAL INTERPRETATION: Given {urgency_label} urgency (score {urgency}/10) and the pain data below, what is happening with this account? Connect the dots between pain drivers, evidence, and competitive movement.

STEP 3 - TIMING ASSESSMENT: Why is this actionable NOW? Consider urgency trajectory, budget signals, and any renewal or contract indicators.

STEP 4 - APPROACH SELECTION: {target_mode}

Account data:
- Company: {company}
- Title: {title}
- Company size: {company_size}
- Industry: {industry}
- Urgency score: {urgency}/10
- Vendor: {vendor_name}
- Pain drivers: {pain_breakdown}
- Evidence quotes: {evidence}
- Competitors gaining: {top_displacement_targets}
- Feature gaps: {top_feature_gaps}
- Budget context: {budget_context}

Return a JSON object with these keys:
- "reasoning_chain": {"persona_insight": "...", "signal_interpretation": "...", "timing_assessment": "..."}
- "situation": 1-2 sentences connecting the data points.
- "approach": What the sales rep should lead with and why. Be specific to the persona.
- "why_now": Why this account is actionable right now.
- "urgency_label": One of "critical", "high", "moderate", "watch".
- "talking_points": Array of 2-3 short bullet points tailored to the persona.
- "confidence": Float 0-1. Below 0.5 if data is thin, above 0.8 only with strong multi-signal evidence.

Return ONLY valid JSON. No markdown fences.',
    1, 'both',
    'You are a B2B sales intelligence analyst. Analyze step by step, ground every claim in data. Return only valid JSON.'
) ON CONFLICT (name) DO NOTHING;

-- Seed: depth 2 decomposition template (Call 1)
INSERT INTO card_templates (name, label, required_fields, optional_fields,
    prompt_template, reasoning_depth, target_mode, system_prompt)
VALUES (
    'sales_action_decompose',
    'Sales Action Card (Decompose)',
    ARRAY['company','urgency','vendor_name','pain_breakdown','title','company_size','industry'],
    ARRAY['evidence','top_displacement_targets','top_feature_gaps','budget_context',
          'persona','company_tier','urgency_label','target_mode'],
    'Perform deep account analysis by breaking this into sub-problems.

A. PERSONA-PAIN MAPPING: Rank the pain drivers by relevance to a {persona} at a {company_tier} {industry} company. Not just severity -- what would this specific role care about most?

B. COMPETITIVE POSITIONING: For each displacement target, map which pain drives the switch and rate the evidence strength (strong/moderate/weak).

C. TIMING SIGNAL ANALYSIS: Evaluate urgency trajectory, budget signals, and industry renewal cycles. Is urgency rising, stable, or cooling?

D. OUTREACH STRATEGY SEARCH: Generate 3 candidate outreach approaches with different hooks, value propositions, and urgency levers. Then select the strongest with a rationale.

Account data:
- Company: {company}
- Title: {title}
- Company size: {company_size}
- Industry: {industry}
- Urgency score: {urgency}/10
- Vendor: {vendor_name}
- Pain drivers: {pain_breakdown}
- Evidence quotes: {evidence}
- Competitors gaining: {top_displacement_targets}
- Feature gaps: {top_feature_gaps}
- Budget context: {budget_context}

Return a JSON object with these keys:
- "persona_pain_ranking": Array of {"pain": "...", "relevance_to_persona": "...", "rank": N}
- "competitive_analysis": Array of {"competitor": "...", "driving_pain": "...", "evidence_strength": "strong|moderate|weak"}
- "timing_signals": {"trajectory": "rising|stable|cooling", "budget_indicators": "...", "renewal_signals": "..."}
- "candidate_strategies": Array of 3 {"hook": "...", "value_prop": "...", "urgency_lever": "..."}
- "selected_strategy": {"index": N, "rationale": "..."}

Return ONLY valid JSON. No markdown fences.',
    2, 'both',
    'You are a B2B sales intelligence analyst performing deep account analysis. Return only valid JSON.'
) ON CONFLICT (name) DO NOTHING;

-- Seed: depth 2 synthesis template (Call 2)
INSERT INTO card_templates (name, label, required_fields, optional_fields,
    prompt_template, reasoning_depth, target_mode, system_prompt)
VALUES (
    'sales_action_synthesize',
    'Sales Action Card (Synthesize)',
    ARRAY['company','urgency','vendor_name'],
    ARRAY['target_mode','persona'],
    'Synthesize the decomposition analysis into a final intelligence card.

SELF-CORRECTION CHECKLIST -- verify each before finalizing:
1. Does every claim trace to specific data from the account or analysis?
2. Is urgency_label consistent with the score ({urgency}/10)?
3. Are talking points tailored to the persona ({persona})?
4. Does the approach match the target mode ({target_mode})?
5. Is confidence calibrated? (< 0.5 if thin data, > 0.8 only with strong evidence)

Decomposition analysis:
{decomposition}

Original account data:
{baseline_json}

Return a JSON object with these keys:
- "situation": 1-2 sentences connecting the data points.
- "approach": What the sales rep should lead with and why. Specific to persona and mode.
- "why_now": Why this account is actionable right now.
- "urgency_label": One of "critical", "high", "moderate", "watch".
- "talking_points": Array of 2-3 short bullet points tailored to the persona.
- "confidence": Float 0-1, calibrated per checklist item 5.
- "evidence_chain": Array of {"claim": "...", "supporting_data_point": "..."} linking each key claim to data.
- "corrections_applied": Array of strings describing any self-corrections made during synthesis.

Return ONLY valid JSON. No markdown fences.',
    2, 'both',
    'You are a B2B sales intelligence analyst. Synthesize and self-correct. Return only valid JSON.'
) ON CONFLICT (name) DO NOTHING;
