# LLM Synthesis Post-Processing — Progress Log

## Status: COMPLETE — All 6 builtin tasks wired

## What Was Done

### 1. Config (`atlas_brain/config.py`)
- Added `synthesis_enabled`, `synthesis_max_tokens`, `synthesis_temperature` to `AutonomousConfig`
- Env vars: `ATLAS_AUTONOMOUS__SYNTHESIS_ENABLED`, `ATLAS_AUTONOMOUS__SYNTHESIS_MAX_TOKENS`, `ATLAS_AUTONOMOUS__SYNTHESIS_TEMPERATURE`

### 2. Runner (`atlas_brain/autonomous/runner.py`)
- Added `_synthesize_with_skill()` method: loads skill, calls LLM with skill content as system prompt and JSON data as user message, strips `<think>` tags
- Modified `_run_builtin()`: checks `task.metadata["synthesis_skill"]`, calls synthesis, returns natural language with `metadata.raw_result` preserved
- Fallback: if synthesis unavailable (disabled, no skill, no LLM, error), returns `str(result)` as before

### 3. Digest Skills (`atlas_brain/skills/digest/`)
All 6 builtin tasks have synthesis skills:

| Task | Skill | Verified |
|------|-------|----------|
| gmail_digest | `digest/email_triage` | Yes — prioritized email summary |
| morning_briefing | `digest/morning_briefing` | Yes — TTS-friendly daily overview |
| security_summary | `digest/security_summary` | Yes — situational awareness update |
| device_health_check | `digest/device_health` | Yes — HA device status report |
| proactive_actions | `digest/proactive_actions` | Yes — conversation action items |
| departure_check | `digest/departure_check` | Yes — unsecured lights/locks/garage alert |

### 4. Environment (`.env`)
- Added 3 synthesis config lines after `ATLAS_AUTONOMOUS__ENABLED`

### 5. Tests (`tests/test_synthesis.py`)
- 14 tests, all passing:
  - Skill loading (3): exists, metadata, content
  - Think-tag stripping (3): single, multiline, no tags
  - _synthesize_with_skill (5): disabled, missing skill, no LLM, success, think-tag strip
  - _run_builtin integration (3): without synthesis, with synthesis, fallback on failure

### 6. Task Activation (via REST API, stored in DB)
- gmail_digest, morning_briefing, security_summary, device_health_check, proactive_actions: `synthesis_skill` added to existing task metadata
- departure_check: new task created (disabled, manual-trigger only — not yet wired to presence_departure hook)

## Activation

To enable synthesis for a task, add `synthesis_skill` to its metadata:

```bash
curl -X PUT http://127.0.0.1:8000/api/v1/autonomous/{task_id} \
  -H "Content-Type: application/json" \
  -d '{"metadata": {"builtin_handler": "gmail_digest", "synthesis_skill": "digest/email_triage"}}'
```

## Architecture Notes

- Synthesis is a post-processing step in `_run_builtin()` — builtin handlers remain pure data fetchers
- Skills are markdown docs loaded by `SkillRegistry.rglob("*.md")` — auto-discovered from `atlas_brain/skills/`
- LLM called synchronously via `llm.chat()` under the global `cuda_lock` from `orchestration/__init__.py`
- New `.md` skill files require a server restart (the SkillRegistry is lazy-loaded once, doesn't watch for new files)
- The `departure_check` task exists but is not yet triggered by presence events — the hook system only dispatches `hook`-type tasks, not `builtin` tasks. Future work needed to bridge presence_departure alerts to builtin handlers.

## Next Steps
- Wire departure_check to presence_departure transition (hook-to-builtin bridge or direct callback)
- Design retrieval workflow for voice-queryable digests (e.g., "Hey Atlas, what's in my email?")
- Consider notification delivery (push synthesized summaries to user via WebSocket/TTS)
