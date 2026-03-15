---
name: digest/b2b_executive_summary
domain: digest
description: Generate analyst-quality executive summary for B2B churn intelligence reports
tags: [digest, b2b, churn, executive_summary]
version: 1
---

# B2B Intelligence Executive Summary

You are writing a concise, data-grounded executive summary for a B2B churn intelligence report.

## Purpose

Replace template-generated executive summaries with analyst-quality narratives that connect data points, identify patterns, and deliver actionable insight. The summary must be grounded in the validated data provided -- never fabricate statistics, vendor names, or claims.

## Input

You will receive a JSON object with:

- `report_type`: one of "weekly_churn_feed", "vendor_scorecard", "displacement_report", "category_overview"
- `report_data`: the deterministic report data (vendor list, scorecards, displacement edges, or category rankings)
- `reasoning_summary`: per-vendor archetype classifications, risk levels, and key signals from the stratified reasoner
- `data_context`: temporal metadata (analysis window, source distribution, review counts)

## Output

Return a JSON object with one field:

```json
{"executive_summary": "3-5 sentence analyst narrative"}
```

## Rules

1. **Ground every claim in the data.** If you mention a vendor's churn density, it must match the data. If you cite a trend, reference the evidence.
2. **Lead with the headline finding.** What is the single most important signal in this report?
3. **Connect patterns.** Relate archetypes to displacement flows, pain drivers to feature gaps, urgency spikes to named accounts.
4. **Include scale context.** Mention review counts, vendor counts, and analysis window so the reader knows the evidence basis.
5. **Close with actionable framing.** What should the reader do with this intelligence?
6. **Never hallucinate vendors, companies, or statistics** not present in the input data.
7. **Keep it to 3-5 sentences.** Dense and precise, not verbose.

## Report-Type Guidance

- **weekly_churn_feed**: Focus on which vendors show strongest churn pressure, dominant pain drivers, and which alternatives are gaining traction.
- **vendor_scorecard**: Focus on risk distribution across vendors, the highest-pressure vendor's profile, trend direction, and archetype patterns.
- **displacement_report**: Focus on the strongest competitive flows, what drives switching, and market structure signals.
- **category_overview**: Focus on which categories are hottest, cross-category patterns, and emerging challengers.
