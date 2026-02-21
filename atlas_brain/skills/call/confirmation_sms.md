---
name: call/confirmation_sms
domain: call
description: Draft a post-booking confirmation SMS to a customer
tags: [call, sms, booking, confirmation]
version: 1
---

# Appointment Confirmation SMS Draft

/no_think

You are drafting a brief SMS confirmation on behalf of {business_name} to a customer who called to request a cleaning estimate or appointment.

## Customer Information

{customer_info}

## Instructions

Write a short, friendly SMS message that:
- Thanks the customer briefly
- Confirms their request was received
- States someone will follow up to confirm timing
- Identifies the sender as {business_name}
- Ends with: Reply STOP to opt out.

## Output Format

Return ONLY the SMS text. No labels, no formatting, no explanations.

Keep it under 300 characters total. Plain text only.
