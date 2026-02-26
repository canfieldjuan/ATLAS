---
name: intelligence/report
domain: intelligence
description: Generate a sellable intelligence report from aggregated relationship data and evidence
tags: [intelligence, report, relationships, commercial]
version: 1
---

# Intelligence Report

/no_think

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
- If `risks` or `opportunities` are empty, say "No material risks/opportunities identified in this window."
- Be explicit about what data is missing in CONFIDENCE & COVERAGE.
- Use the plain section titles above without markdown prefixes (no `#`).
