---
name: digest/b2b_campaign_sequence
domain: digest
description: Generate the next email in a B2B campaign sequence based on engagement signals
tags: [b2b, campaign, outreach, sequence]
version: 1
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
- Sign off with the sender's name from the selling context
- angle_reasoning is for internal debugging -- be honest about what signal drove your decision
- If this is a break-up email (last step), make it clear you won't email again unless they respond
