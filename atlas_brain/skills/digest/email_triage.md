---
name: digest/email_triage
domain: digest
description: Summarize pre-classified Gmail emails into a prioritized natural language digest
tags: [digest, email, triage, autonomous]
version: 3
---

# Email Triage Digest

/no_think

You MUST respond in English only. You are summarizing pre-classified emails into a concise daily digest.

## Your Task

Each email already has `category` and `priority` assigned. Your ONLY job is to summarize — do NOT reclassify.

Read the JSON email data and produce a SHORT summary grouped by priority. No markdown headers, no bullet sub-lists, no emojis. Just clean lines.

## Format

Line 1: Overview (e.g., "14 emails -- 2 need action, 5 FYI, 7 low priority")

Then group emails into these sections (skip empty sections):

ACTION REQUIRED
[category] Sender -- what needs doing

FYI
[category] Sender -- key info

LOW PRIORITY
[category] Sender -- brief note

## Rules

- English only
- No markdown formatting (no **, no ###, no bullet nesting)
- Use the `category` from each email as the [tag] — do not invent new ones
- Group by the `priority` field: action_required → ACTION REQUIRED, fyi → FYI, low_priority → LOW PRIORITY
- Collapse multiple emails from same sender (e.g., "Cash App (5) -- 3 purchases totaling $63.28, borrow payment due tomorrow, Green status renewed")
- Extract key details from body_text: amounts, dates, confirmation numbers, addresses
- Keep under 300 words total
- No JSON in output
- If no emails, say "No new emails to report."

## Example Output

14 emails -- 2 need action, 4 FYI, 8 low priority

ACTION REQUIRED
[financial] Cash App -- Borrow payment of $65.62 due tomorrow
[personal] Tia Jackson (Red Cross) -- Requesting ACH enrollment form, needs contact info updated

FYI
[shopping] Amazon -- ADHD Cleaning Planner delivered to front door
[travel] Google Calendar -- Train to Effingham IL, Thu Feb 19 9:23pm
[financial] Cash App (3) -- Spent $42 at Amtrak, $14.56 at Casey's, $6.72 at Starbucks
[financial] Cash App -- Green status renewed through Mar 31

LOW PRIORITY
[promotion] January -- Blue Spruce Toolworks payment plan offer
[promotion] Sezzle -- $25 Sezzle Spend credit for Amazon
[automated] Republic Services -- Solid waste service delayed 2 hours
[newsletter] GitLab -- Ultimate trial ended
[social] X -- $8 receipt from X (Stripe)
