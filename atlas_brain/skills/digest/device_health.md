---
name: digest/device_health
domain: digest
description: Synthesize Home Assistant device health data into a readable status report
tags: [digest, devices, health, autonomous]
version: 1
---

# Device Health Digest

You are summarizing a Home Assistant device health check into a concise status report.

## Input

You will receive a JSON object with:
- `total_entities`: Total number of HA entities checked
- `healthy`: Number of entities with no issues
- `issues`: A list of issue objects, each with:
  - `entity_id`: The HA entity ID (e.g., "sensor.living_room_temperature")
  - `friendly_name`: Human-readable name
  - `issue`: One of "unavailable", "unknown", "low_battery", "stale"
  - `battery_pct`: Battery percentage (only for low_battery issues)
  - `last_updated`: ISO timestamp (only for stale issues)
- `summary`: A basic pre-built summary (ignore this — you are replacing it)

## Output Structure

1. **Overview** — One sentence: how many healthy out of total, overall status (good / some issues / needs attention)
2. **Issues by category** — Group issues and summarize each category:
   - Unavailable devices: list by friendly name
   - Low battery: list with percentage
   - Stale devices: mention how long since last update
   - Unknown state: list by friendly name
3. Skip any category with zero issues.

## Rules

- NEVER fabricate details — only use what's in the data
- Keep the entire summary under 200 words
- Use friendly names, not entity IDs, when available
- Use plain, conversational language — this may be read aloud via TTS
- If there are more than 5 devices in a category, list the first 5 and say "and N more"
- If everything is healthy, simply say "All N devices are healthy — no issues detected."
- Do not use markdown formatting — plain text with line breaks only
