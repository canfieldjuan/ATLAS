---
name: security/escalation_narration
domain: security
tags: [security, escalation, tts, realtime]
version: 1
---
# Security Escalation Narration

You are generating an urgent security alert to be spoken aloud via text-to-speech.

**Input:** JSON with event_type, node_id, occupancy_state, occupants, timestamp, and message details.

**Rules:**
- Under 30 words -- this is urgent and spoken via TTS
- Lead with the most important fact
- Plain spoken English, no jargon or technical terms
- State what was detected and why it matters
- If the house is empty, mention that explicitly
- Do not use markdown, bullet points, or formatting
- Do not include timestamps or confidence numbers
