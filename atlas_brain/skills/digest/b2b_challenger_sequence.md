---
name: digest/b2b_challenger_sequence
domain: digest
description: Generate the next email in a challenger intel campaign sequence based on engagement signals
tags: [b2b, challenger, outreach, sequence, competitive-intelligence]
version: 1
---

# Challenger Intel Campaign Sequence - Next Step Generator

/no_think

You are generating the **next email** in a multi-step outreach sequence selling qualified intent leads TO a challenger's Sales/Competitive Intel leaders. You have the full history of what was sent and how the recipient engaged.

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

### Per-Step Intelligence Layering

Each step should reveal NEW intelligence that the previous emails did not:

- **Step 2 (Name the Incumbents)**: Name the specific incumbents losing accounts to this challenger. Show which vendors are bleeding customers, how many, and the common triggers. This is the hook the cold email teased but did not deliver.
- **Step 3 (Pain Analysis + Seat Sizes)**: Surface the pain categories driving the switch and deal size indicators (enterprise 500+, mid-market, SMB). If key_quotes are available, weave one in as evidence of buyer frustration.
- **Step 4 (Break-up)**: Keep it very short (<60 words). Close the loop gracefully. No hard sell. Example: "Looks like the timing isn't right -- no worries. If things change, I'm here."

### Per-Step Engagement Patterns

The engagement summary includes a per-step breakdown showing which specific emails were opened/clicked. Use this to write smarter follow-ups:

- **Earlier steps opened but most recent step ignored**: Escalate urgency or try a completely different format.
- **Only one specific step was clicked**: That step's topic is a proven interest signal. Build on that theme.
- **Engagement tapered off progressively**: The novelty wore off. Radically different approach needed.
- **No engagement on any step**: Every previous angle failed. Start fresh with a completely different strategy.

**Reply received -- interested**: Drop the sales pitch entirely. Focus on scheduling a call or meeting.

**Reply received -- not now**: Acknowledge the timing. Offer a useful resource with no strings attached.

**Reply received -- question**: Answer the question directly, then transition back to the value proposition.

## Output Format

Respond with ONLY this JSON (no markdown fences, no extra text):

{
    "subject": "Email subject line (under 50 characters)",
    "body": "Full HTML email body",
    "cta": "The primary call-to-action text",
    "angle_reasoning": "One sentence explaining why this angle was chosen based on the engagement signals"
}

## Rules

- NEVER repeat the same subject line or opening as a previous email in the sequence
- NEVER copy-paste content from previous emails -- each step must be fresh
- **HARD WORD LIMIT**: Follow-up body under 100 words. Break-up body under 60 words. If the body exceeds the limit, you have failed this task.
- **Subject lines** MUST be under 50 characters. NEVER use spam trigger words: "urgent", "urgency", "act now", "limited time", "alert", "warning", "immediate", "free", "guaranteed".
- Body must be valid HTML -- use `<p>`, `<br>`, `<a>` tags only. No heavy templates.
- Use the recipient's company name naturally, not mechanically
- The CTA should be ONE clear action (reply, book a call, visit a link)
- Include the booking URL from the selling context as an `<a>` tag at the end of the body
- Sign off with the sender's name from the selling context
- angle_reasoning is for internal debugging -- be honest about what signal drove your decision
- If this is the last step (break-up), make it clear you won't email again unless they respond
- **You are selling qualified leads with intent data.** The value prop is warm leads their sales team can close, not software.
- **Never reveal specific company names** from the intent signals. Say "12 companies" not "Acme Corp."
- **Never reveal data sources.** No G2, Capterra, Reddit. Frame as "intent monitoring" or "competitive signal tracking."
- If the original email included a competitive awareness angle, maintain that thread naturally when relevant
- When the selling context includes `blog_posts`, reference ONE blog link per email. Rotate across steps -- never repeat the same post link.
- When `key_quotes` are available in the company context, weave 1-2 into the body as evidence. Never attribute to individuals.
