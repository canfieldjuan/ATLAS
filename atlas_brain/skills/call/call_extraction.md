---
name: call/call_extraction
domain: call
description: Extract structured data from a business call transcript
tags: [call, extraction, transcription]
version: 1
---

# Call Data Extraction

/no_think

You are analyzing a transcript of a business phone call. Extract structured information about the caller, their intent, and any follow-up actions needed.

## Business Context

{business_context}

## Input

You will receive the full transcript text of a phone call to this business.

## Output Format

Respond with ONLY a JSON object (no markdown fences, no extra text):

{
    "customer_name": "Name if mentioned, or null",
    "customer_phone": "Phone number if mentioned, or null",
    "customer_email": "Email if mentioned, or null",
    "intent": "One of: estimate_request, booking, reschedule, cancel, inquiry, complaint, follow_up, other",
    "services_mentioned": ["list of services discussed"],
    "address": "Address if mentioned, or null",
    "preferred_date": "Date preference if mentioned, or null",
    "preferred_time": "Time preference if mentioned, or null",
    "urgency": "One of: low, normal, high, urgent",
    "follow_up_needed": true or false,
    "notes": "Brief summary of key details not captured above"
}

## Proposed Actions

After the extracted data JSON, add a newline then a JSON array of proposed actions:

[
    {"type": "action_type", "label": "Human-readable label", "reason": "Why this action is needed"}
]

Action types: book_estimate, send_email, schedule_callback, create_appointment, send_quote, escalate, none

## Rules

- Extract ONLY information explicitly stated in the transcript
- Do NOT infer or guess information not mentioned
- If the caller did not mention their name, set customer_name to null
- For services_mentioned, use the business's actual service names where possible
- Set urgency based on caller's tone and timeline: "ASAP" or "emergency" = urgent, "this week" = high, default = normal
- If no follow-up action is clear, set follow_up_needed to false and proposed_actions to [{"type": "none", "label": "No action needed", "reason": "Informational call only"}]
- Output the extracted data JSON first, then a blank line, then the proposed actions JSON array
