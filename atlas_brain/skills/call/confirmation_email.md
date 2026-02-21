---
name: call/confirmation_email
domain: call
description: Draft a post-booking confirmation email to a customer
tags: [call, email, booking, confirmation]
version: 1
---

# Appointment Confirmation Email Draft

/no_think

You are drafting a confirmation email on behalf of {business_name} to a customer who called to request a cleaning estimate or appointment.

## Customer Information

{customer_info}

## Instructions

Write a warm, professional confirmation email that:
- Thanks the customer for calling {business_name}
- Confirms their request has been received and logged
- Lets them know someone will follow up shortly to confirm the exact date and time
- Briefly describes what to expect (in-home estimate or cleaning visit)
- Encourages them to call or reply if they have questions

## Output Format

Return ONLY the email in this exact format with no markdown, no code fences, and no extra commentary:

SUBJECT: [concise subject line]

[email body]

Keep the body under 180 words. Use a friendly, professional tone. Sign off as {business_name}.
