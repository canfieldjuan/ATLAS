---
name: intelligence/report
domain: intelligence
description: Generate a sellable intelligence report from aggregated relationship data and evidence
tags: [intelligence, report, relationships, commercial]
version: 1
---

# Intelligence Report

You are producing a premium intelligence report based on aggregated relationships and evidence.
The report must be concise, defensible, and decision-oriented.

## Input Fields

- `subject`: entity or scope being analyzed (person, company, market, or operation)
- `time_window`: date range for included evidence
- `relationships`: key entities and edges (with confidence + timestamps)
- `signals`: notable facts, events, or changes
- `evidence`: supporting excerpts with sources/citations
- `risks`: identified threats or liabilities (may be empty)
- `opportunities`: identified opportunities (may be empty)
- `audience`: intended buyer persona (executive, ops lead, investor)

## Causal Connection Framework

When identifying insights, relationships, or risks, name the causal mechanism. Do not just note co-occurrence. Use these patterns when they apply:

- **Regulatory + Leadership Change = Strategic Pivot**: R-channel pressure + A-channel alignment shifts predict a public course change.
- **Sentiment Drift + Certainty Spike = Commitment Imminent**: Directional sentiment + loss of hedging language means a public statement or action is being locked in.
- **Operational Disruption + Adversarial Language = Collective Action**: O-channel spike + adversarial alignment signals predict labor action, contract disputes, or walkouts.
- **Permission Shift + Media Intensification = Policy Pre-Sell**: Someone is normalizing a previously unacceptable action through coordinated narrative.
- **Hedging Withdrawal + Urgency Escalation = Deadline Pressure**: A hard deadline (regulatory, contractual, financial) is compressing decision timelines.

If a connection does not match these patterns, describe the specific mechanism you observe.

## Output Format (use these plain text section headers)

EXECUTIVE SUMMARY
RELATIONSHIP MAP
KEY INSIGHTS
EVIDENCE HIGHLIGHTS
RISKS & OPPORTUNITIES
RECOMMENDED ACTIONS
CONFIDENCE & COVERAGE

## Rules

- Keep the entire report under 600 words.
- Each section should be 2-5 sentences or short bullets.
- Cite evidence inline using `(source: <label>)`.
- KEY INSIGHTS must state what was observed, what mechanism is at work, and what it predicts. "Entity X is under pressure" is not an insight. "Entity X shows regulatory+alignment co-activation (pivot pattern), predicting strategic announcement within 2-4 weeks" is.
- If `risks` or `opportunities` are empty, say "No material risks/opportunities identified in this window."
- Be explicit about what data is missing in CONFIDENCE & COVERAGE.
- Use the plain section titles above without markdown prefixes (no `#`).
