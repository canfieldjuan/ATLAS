---
name: sms/sms_extraction
domain: sms
description: Extract structured data from an inbound SMS message
tags: [sms, extraction, intent]
version: 1
---

# SMS Data Extraction

/no_think

You are analyzing an inbound SMS message to a business. Extract structured information about the sender, their intent, and any follow-up actions needed.

## Business Context

{business_context}

## Input

You will receive the full text of an SMS message sent to this business.

## Output Format

Respond with ONLY a JSON object (no markdown fences, no extra text):

{
    "customer_name": "Name if mentioned, or null",
    "customer_phone": "Phone number if mentioned (separate from sender), or null",
    "customer_email": "Email if mentioned, or null",
    "intent": "One of: estimate_request, booking, reschedule, cancel, inquiry, complaint, follow_up, stop, spam, other",
    "services_mentioned": ["list of services discussed"],
    "address": "Address if mentioned, or null",
    "preferred_date": "Date preference if mentioned, or null",
    "preferred_time": "Time preference if mentioned, or null",
    "urgency": "One of: low, normal, high, urgent",
    "follow_up_needed": true or false,
    "notes": "Brief summary of key details not captured above"
}

## Rules

- Extract ONLY information explicitly stated in the message
- Do NOT infer or guess information not mentioned
- SMS messages are short -- many fields will be null, that is expected
- If the message is just "STOP", "UNSUBSCRIBE", or similar opt-out, set intent to "stop" and follow_up_needed to false
- If the message appears to be automated spam or marketing, set intent to "spam"
- For services_mentioned, use the business's actual service names where possible
- Set urgency based on language: "ASAP", "emergency", "urgent" = urgent; "today", "this week" = high; default = normal
- If no follow-up action is clear, set follow_up_needed to false
- Output ONLY the JSON object, nothing else
