# Voice Pipeline Tool Routing Fix

**Date**: 2026-02-05
**Status**: Implemented

## Problem

Voice commands for scheduling, reminders, calendar queries, and email were broken due to three disconnection points:

1. **Fast-path tool execution fired BEFORE workflow regex** - Intent router intercepted `get_calendar`, `list_reminders` as tool_use but they weren't registered, causing TOOL_NOT_FOUND
2. **Workflow tools intentionally removed from registry (2026-02-02)** - Any tool_use classification for unregistered tools failed
3. **Workflow regex patterns too narrow** - "set an alarm", "wake me up", "email John about..." not caught

## Solution (4 Phases)

### Phase 1: Re-register Read-Only Tools
- `atlas_brain/tools/__init__.py`: Registered `calendar_tool` and `list_reminders_tool` in the tool registry
- These are read-only query tools safe for direct execution without workflow state

### Phase 2: Reorder classify_intent
- `atlas_brain/agents/graphs/atlas.py`: Moved `_detect_workflow_intent()` call BEFORE intent router fast-path
- Removed read-only "show my reminders" pattern from `_REMINDER_PATTERNS` (now handled by fast-path)

### Phase 3: Tool-Name-to-Workflow Bridge
- `atlas_brain/services/intent_router.py`: Added `TOOL_NAME_TO_WORKFLOW` mapping (`set_reminder` -> `reminder`, `complete_reminder` -> `reminder`, `send_email` -> `email`)
- `atlas_brain/agents/graphs/atlas.py`: Added bridge logic after intent router - if tool_use classified but no tool in registry, redirect to corresponding workflow

### Phase 4: Expanded Regex Patterns
- **Reminders**: Added `wake me up`, `remember to`, `set an alarm`, `add/create reminder/alarm`, `alert/notify me`
- **Bookings**: Added `make an appointment`
- **Email**: Added `send a message to`, `email X about/regarding`
- **Calendar**: Added `put on calendar`, `new/add meeting/event on/at/for`

## Files Modified

| File | Changes |
|------|---------|
| `atlas_brain/tools/__init__.py` | Registered 2 read-only tools, updated comment |
| `atlas_brain/agents/graphs/atlas.py` | Reordered classify_intent, removed read regex, added bridge, expanded patterns |
| `atlas_brain/services/intent_router.py` | Added `TOOL_NAME_TO_WORKFLOW` constant |

## Routing Flow (After Fix)

```
User input
  ↓
1. Workflow regex check (FIRST) → workflow_start if matched
  ↓ (no match)
2. Intent router (DistilBERT) classification
  ↓
3a. Fast-path tool (parameterless, registered) → execute directly
3b. tool_use + unregistered workflow tool → Bridge → workflow_start
3c. conversation/device_command → normal flow
```

## Test Matrix

| Voice Command | Before | After |
|---|---|---|
| "what's on my calendar?" | TOOL_NOT_FOUND | Fast-path -> calendar_tool |
| "show my reminders" | TOOL_NOT_FOUND | Fast-path -> list_reminders_tool |
| "set an alarm for 7am" | TOOL_NOT_FOUND | Regex -> reminder workflow |
| "wake me up at 6" | TOOL_NOT_FOUND | Regex -> reminder workflow |
| "email John about meeting" | TOOL_NOT_FOUND | Regex -> email workflow |
| "make an appointment" | TOOL_NOT_FOUND | Regex -> booking workflow |
| "remind me to call mom" | Works | Works (regex, pre-router) |
| "turn on the lights" | Works | Works (unchanged) |
