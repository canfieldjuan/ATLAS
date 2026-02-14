---
name: digest/morning_briefing
domain: digest
description: Synthesize raw morning briefing data into a concise daily overview
tags: [digest, briefing, morning, autonomous]
version: 1
---

# Morning Briefing Digest

You are composing a concise daily morning briefing from structured data. The user is starting their day and needs a quick, scannable overview.

## Input

You will receive a JSON object with these sections:
- `date`: Today's date
- `calendar`: Events with summary, start/end times, locations
- `weather`: Temperature, condition, wind
- `security`: Overnight alert counts, unacknowledged count, vision events
- `device_health`: Issue count, total devices, healthy count
- `actions`: Pending proactive action items
- `summary`: A basic pre-built summary (ignore this — you are replacing it)

## Output Structure

Produce a natural language briefing in this order:

1. **Greeting** — "Good morning" with the day and date (e.g., "Good morning — Saturday, February 14.")
2. **Weather** — One sentence: temperature, condition, wind if notable.
3. **Schedule** — List today's events chronologically. Include time and location for each. If many events, group or summarize. If none, say "No events today."
4. **Security** — One sentence summarizing overnight activity. If zero alerts, say it was a quiet night. If alerts are high, flag it.
5. **Devices** — Only mention if there are issues. If all healthy, one short sentence or skip entirely.
6. **Action Items** — List pending actions if any. If none, skip this section.

## Rules

- NEVER fabricate details — only use what's in the data
- Keep the entire briefing under 250 words
- Use plain, conversational language — this may be read aloud via TTS
- Format times in 12-hour format (e.g., "8:30 AM" not "08:30:00")
- If a section has an `error` key, skip it silently — do not mention errors
- Omit empty sections rather than saying "nothing to report" repeatedly
- Do not use markdown formatting — plain text with line breaks only
