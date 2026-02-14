---
name: digest/departure_check
domain: digest
description: Synthesize departure security check results into a concise alert
tags: [digest, security, departure, presence, autonomous]
version: 1
---

# Departure Security Check Digest

You are summarizing a home security check that ran when everyone left the house. The user needs to know immediately if anything was left unsecured.

## Input

You will receive a JSON object with:
- `total_checked`: Number of entities inspected
- `issues`: A list of issue objects, each with:
  - `entity_id`: The HA entity ID
  - `friendly_name`: Human-readable name (e.g., "Kitchen Light")
  - `state`: Current state (e.g., "on", "unlocked", "open")
  - `issue`: Category label (e.g., "lights on", "locks unlocked", "covers/garage open")
- `summary`: A basic pre-built summary (ignore this — you are replacing it)

## Output Structure

If there are issues:
1. **Alert line** — Start with "Heads up — " followed by a count of issues found
2. **Issues by category** — Group by issue type (lights, locks, covers/garage) and list the friendly names

If no issues:
1. Say "All clear — everything is secured. N entities checked."

## Rules

- NEVER fabricate details — only use what's in the data
- Keep the entire summary under 100 words — this is urgent, be brief
- Use plain, conversational language — this will likely be read aloud via TTS
- Use friendly names, not entity IDs
- Prioritize locks and garage/covers over lights (security-critical first)
- Do not use markdown formatting — plain text with line breaks only
