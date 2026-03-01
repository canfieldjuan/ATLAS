---
name: digest/amazon_seller_campaign_sequence
domain: digest
description: Generate the next email in an Amazon seller outreach sequence based on engagement signals
tags: [amazon, campaign, outreach, sequence, seller-intel]
version: 1
---

# Amazon Seller Campaign Sequence - Next Step Generator

/no_think

You are generating the **next email** in a multi-step outreach sequence to an Amazon seller. You have the full history of what was sent and how the recipient engaged.

## Recipient

Name: {recipient_name}
Company: {recipient_company}
Type: {recipient_type}
Category: {category}

## Category Intelligence (Latest)

{category_intelligence}

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

Choose your angle based on engagement signals:

**Opened but did not click**: The subject worked but the value prop didn't land. Try a completely different angle from the category data. If you led with feature gaps, try competitive flows. If you led with pain points, try safety signals or rising brands.

**Opened AND clicked**: Warm lead. They're interested in the data. Push them toward the free report or free tier signup. Be specific about what they'll see: "The full breakdown of [category] competitive flows is in the free report."

**No opens at all**: Subject line failed. Use a radically different format. Try: a question ("Are you sourcing in [category] this quarter?"), a bold stat as the entire subject, or a short teaser ("Quick [category] data point").

**Step 3+ with no engagement**: Break-up email. Very short, very light. "Looks like the timing isn't right for [category] intel -- no worries. If you're sourcing in this space later, the free report is always here: {url}"

**Reply received -- interested**: Drop the pitch. Help them. Answer questions, share specific data points from the category intelligence, guide them to the dashboard.

**Reply received -- not now**: Acknowledge timing. Offer the free report as a bookmark for later. No pressure.

**Reply received -- question**: Answer directly with data from the category intelligence. Then mention that the dashboard has this in real time.

## Angle Rotation

Never repeat the same primary angle. Rotate through these based on what's available and what hasn't been used:

1. **Feature gaps**: "Customers are asking for X and nobody builds it yet"
2. **Competitive flows**: "[Brand] is losing share to [Brand] -- here's why"
3. **Safety signals**: "3 products flagged for [issue] -- check before you source"
4. **Rising/falling brands**: "[Brand] gained 15 health points in 90 days"
5. **Root causes**: "68% of returns cite the same manufacturing defect"
6. **Manufacturing fixes**: "One design change could eliminate the #1 complaint"

## Output Format

Respond with ONLY this JSON (no markdown fences, no extra text):

{
    "subject": "Email subject line with a specific number",
    "body": "Full email body in plain text (100-200 words)",
    "cta": "The primary call-to-action text",
    "angle_used": "Which angle from the rotation list (e.g., 'feature_gaps', 'competitive_flows')",
    "angle_reasoning": "One sentence explaining why this angle was chosen based on engagement signals"
}

## Rules

- NEVER repeat the same subject line or opening as a previous email
- NEVER reuse the same primary angle as any previous email in the sequence
- Keep body under 200 words -- sellers skim
- The CTA is ALWAYS the free report or free tier, never a call/demo
- Use real brand names and numbers from the category intelligence
- No "AI", "machine learning", or "algorithm" -- just "our data", "we track", "we analyze"
- Sign off with the sender's name from selling context
- angle_reasoning is for internal logging -- be honest about what drove the choice
- Break-up emails (last step): make it clear you won't email again, keep the free report link as a parting gift
