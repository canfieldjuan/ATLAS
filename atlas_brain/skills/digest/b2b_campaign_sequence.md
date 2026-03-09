---
name: digest/b2b_campaign_sequence
domain: digest
description: Generate the next email in a B2B campaign sequence based on engagement signals
tags: [b2b, campaign, outreach, sequence]
version: 2
---

# B2B Campaign Sequence - Next Step Generator

/no_think

You are generating the **next email** in a multi-step B2B outreach sequence. You have the full history of what was sent and how the recipient engaged.

## Company

Name: {company_name}

## Company Context

{company_context}

## Selling Context

{selling_context}

## Sequence State

Step: {current_step} of {max_steps}
Days since last email: {days_since_last}

## Engagement Summary

{engagement_summary}

## Previous Emails in This Sequence

{previous_emails}

## Strategy Rules

Choose your angle based on what the engagement signals tell you:

**Opened but did not click**: The subject line worked but the CTA did not land. Try a completely different value proposition or angle. Do NOT repeat the same pitch.

**Opened AND clicked**: This is a warm lead. Be direct. Push for a meeting or call. Reference what they clicked on if known.

**No opens at all**: The subject line failed. Use a radically different approach -- different format, different hook, different angle. Consider a question-based or curiosity-driven subject.

**Step 3+ with no engagement**: This is a "break-up" email. Keep it very short and light. Close the loop gracefully. No hard sell. Example: "Looks like the timing isn't right -- no worries. If things change, I'm here."

### Per-Step Engagement Patterns

The engagement summary includes a per-step breakdown showing which specific emails were opened/clicked. Use this to write smarter follow-ups:

- **Earlier steps opened but most recent step ignored**: The recipient may have lost interest or the last angle missed. Escalate urgency or try a completely different format (e.g., switch from value prop to social proof, or from long-form to a short question).
- **Only one specific step was clicked**: That step's topic/angle is a proven interest signal. Reference or build on that topic -- don't repeat it verbatim, but lean into the same theme.
- **Engagement tapered off progressively** (Step 1 opened, Step 2 opened, Step 3 nothing): The novelty wore off. A radically different approach is needed -- new format, new hook, new framing.
- **No engagement on any step**: Every previous subject line and angle failed. Do not iterate on what didn't work. Start fresh with a completely different strategy.

**Reply received -- interested**: Drop the sales pitch entirely. Focus on scheduling a call or meeting. Be helpful, not salesy.

**Reply received -- not now**: Acknowledge the timing. Offer a useful resource (case study, guide) with no strings attached. Set a future touchpoint.

**Reply received -- question**: Answer the question directly and thoroughly. Then gently transition back to the value proposition.

## Output Format

Respond with ONLY this JSON (no markdown fences, no extra text):

{
    "subject": "Email subject line",
    "body": "Full HTML email body",
    "cta": "The primary call-to-action text",
    "angle_reasoning": "One sentence explaining why this angle was chosen based on the engagement signals"
}

## Rules

- NEVER repeat the same subject line or opening as a previous email in the sequence
- NEVER copy-paste content from previous emails -- each step must be fresh
- Keep the body concise -- under 150 words for cold outreach, under 100 for break-up emails
- Use the recipient's company name naturally, not mechanically
- The CTA should be ONE clear action (reply, book a call, visit a link)
- Body should be valid HTML (use <p>, <br>, <a> tags) but keep formatting minimal -- no heavy templates
- Sign off with the sender's name from the selling context. If a sender title is available, include it. If the sender name reads like a brand or company, sign off with the sender company or "the team" instead of dropping the signature.
- angle_reasoning is for internal debugging -- be honest about what signal drove your decision
- If this is a break-up email (last step), make it clear you won't email again unless they respond
- Do not reuse adversarial competitive-awareness lines. If market timing matters, frame it as buyer activity or category movement, never as "your competitors get this too".
- When the selling context includes `blog_posts`, reference ONE blog link per email as published analysis. Rotate posts across sequence steps -- never repeat the same post link in two emails. Frame as: "We recently published..." or "Our latest analysis covers..."
- When the company context includes `target_persona`, maintain persona-consistent tone across all steps: `executive` = ROI/TCO/strategic focus with `economic_buyer` tone; `technical` = feature gaps/integration/migration with `evaluator` tone; `operations` = support quality/reliability/team productivity with `champion` tone. Match `role_type` from the context if present.
- Subject lines must stay under 50 characters and must not reveal competitor names, premium account details, or the main paid insight.
