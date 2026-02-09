# Workflow Wiring Implementation

**Date:** 2026-02-02
**Status:** COMPLETE - All Phases Implemented
**Related:** booking.py, reminder.py, email.py, calendar.py, atlas.py

---

## Problem Statement

### Issue 1: Workflows Never Initiated
All 4 LangGraph workflows (booking, reminder, email, calendar) exist and support multi-turn conversations, but **none are ever started**. The `continue_workflow` function in `atlas.py` only continues existing workflows - there's no code to detect workflow intents and initiate them.

### Issue 2: Tools Exposed as Standalone
Workflow-internal tools are registered in the public `tool_registry`, allowing the LLM to call them directly instead of using the multi-turn workflow. This bypasses the designed conversation flow.

### Issue 3: Conversation Mode Timeout
Separate issue: 8-second timeout may be too short for complex responses. (To be addressed in Phase 3)

---

## Current State Analysis

### Files Verified:
| File | Purpose | Status |
|------|---------|--------|
| `atlas_brain/tools/__init__.py` | Tool registry | Lines 97-109 register workflow tools |
| `atlas_brain/agents/graphs/atlas.py` | Main agent | No workflow initiation logic |
| `atlas_brain/agents/graphs/booking.py` | Booking workflow | Multi-turn ready, imports tools directly |
| `atlas_brain/agents/graphs/reminder.py` | Reminder workflow | Multi-turn ready, uses ReminderService |
| `atlas_brain/agents/graphs/email.py` | Email workflow | Multi-turn ready, imports tools directly |
| `atlas_brain/agents/graphs/calendar.py` | Calendar workflow | Multi-turn ready, imports tools directly |
| `atlas_brain/services/intent_router.py` | Intent classification | No workflow categories defined |

### Breaking Change Analysis:
- **NO BREAKING CHANGES** expected from removing tools from registry
- Workflow files import tools directly from their modules (e.g., `from ...tools.scheduling import book_appointment_tool`)
- `real_services.py` imports `calendar_tool` directly from module, not registry
- Tools remain available for internal workflow use

---

## Implementation Plan

### Phase 1: Remove Workflow Tools from Public Registry
**Goal:** Prevent LLM from calling workflow tools directly

**File:** `atlas_brain/tools/__init__.py`

**Remove from registry (lines 97-109):**
```python
# REMOVE these registrations:
tool_registry.register(calendar_tool)              # Line 97
tool_registry.register(reminder_tool)              # Line 98
tool_registry.register(list_reminders_tool)        # Line 99
tool_registry.register(complete_reminder_tool)     # Line 100
tool_registry.register(email_tool)                 # Line 102
tool_registry.register(estimate_email_tool)        # Line 103
tool_registry.register(proposal_email_tool)        # Line 104
tool_registry.register(check_availability_tool)    # Line 105
tool_registry.register(book_appointment_tool)      # Line 106
tool_registry.register(cancel_appointment_tool)    # Line 107
tool_registry.register(reschedule_appointment_tool) # Line 108
tool_registry.register(lookup_customer_tool)       # Line 109
```

**KEEP registered:**
```python
tool_registry.register(weather_tool)      # Line 93 - KEEP
tool_registry.register(traffic_tool)      # Line 94 - KEEP
tool_registry.register(location_tool)     # Line 95 - KEEP
tool_registry.register(time_tool)         # Line 96 - KEEP
tool_registry.register(notify_tool)       # Line 101 - KEEP
# All presence tools - KEEP
# All security tools - KEEP
# All display tools - KEEP
```

**Validation:**
- Verify workflow files still import tools directly
- Test that workflows can still execute tools internally

---

### Phase 2: Add Workflow Intent Detection

**Goal:** Detect when user wants to start a workflow and route appropriately

#### Step 2A: Add Workflow Classification Function

**File:** `atlas_brain/agents/graphs/atlas.py`

**Insertion Point:** After line 59 (after `_is_cancel_intent` function)

**New Code:**
```python
# Workflow intent patterns
_BOOKING_PATTERNS = [
    re.compile(r"book\s+(?:an?\s+)?appointment", re.IGNORECASE),
    re.compile(r"schedule\s+(?:an?\s+)?appointment", re.IGNORECASE),
    re.compile(r"(?:i\s+)?(?:need|want)\s+to\s+(?:book|schedule)", re.IGNORECASE),
    re.compile(r"set\s+up\s+(?:an?\s+)?appointment", re.IGNORECASE),
]

_REMINDER_PATTERNS = [
    re.compile(r"remind\s+me", re.IGNORECASE),
    re.compile(r"set\s+(?:a\s+)?reminder", re.IGNORECASE),
    re.compile(r"(?:list|show|what\s+are)\s+(?:my\s+)?reminders?", re.IGNORECASE),
    re.compile(r"(?:delete|remove|complete|done\s+with)\s+(?:the\s+)?reminder", re.IGNORECASE),
]

_EMAIL_PATTERNS = [
    re.compile(r"send\s+(?:an?\s+)?email", re.IGNORECASE),
    re.compile(r"email\s+(?:to\s+)?(?:\w+)", re.IGNORECASE),
    re.compile(r"(?:draft|compose|write)\s+(?:an?\s+)?email", re.IGNORECASE),
]

_CALENDAR_PATTERNS = [
    re.compile(r"add\s+(?:to\s+)?(?:my\s+)?calendar", re.IGNORECASE),
    re.compile(r"create\s+(?:a\s+)?(?:calendar\s+)?event", re.IGNORECASE),
    re.compile(r"schedule\s+(?:a\s+)?(?:meeting|event)", re.IGNORECASE),
    re.compile(r"(?:what'?s?\s+)?(?:on\s+)?my\s+calendar", re.IGNORECASE),
]


def _detect_workflow_intent(text: str) -> Optional[str]:
    """
    Detect if text indicates a workflow should be started.

    Returns workflow type or None.
    """
    for pattern in _BOOKING_PATTERNS:
        if pattern.search(text):
            return "booking"

    for pattern in _REMINDER_PATTERNS:
        if pattern.search(text):
            return "reminder"

    for pattern in _EMAIL_PATTERNS:
        if pattern.search(text):
            return "email"

    for pattern in _CALENDAR_PATTERNS:
        if pattern.search(text):
            return "calendar"

    return None
```

#### Step 2B: Add Workflow Start Node

**File:** `atlas_brain/agents/graphs/atlas.py`

**Insertion Point:** After `classify_intent` function (around line 237)

**New Code:**
```python
async def start_workflow(state: AtlasAgentState) -> AtlasAgentState:
    """Start a new workflow based on detected intent."""
    start_time = time.perf_counter()
    workflow_type = state.get("workflow_to_start")
    session_id = state.get("session_id")
    input_text = state.get("input_text", "")

    if workflow_type == "booking":
        result = await run_booking_workflow(
            input_text=input_text,
            session_id=session_id,
        )
        response = result.get("response", "")

    elif workflow_type == "reminder":
        result = await run_reminder_workflow(
            input_text=input_text,
            session_id=session_id,
        )
        response = result.get("response", "")

    elif workflow_type == "email":
        result = await run_email_workflow(
            input_text=input_text,
            session_id=session_id,
        )
        response = result.get("response", "")

    elif workflow_type == "calendar":
        result = await run_calendar_workflow(
            input_text=input_text,
            session_id=session_id,
        )
        response = result.get("response", "")

    else:
        response = "I'm not sure which workflow to start."

    total_ms = (time.perf_counter() - start_time) * 1000

    return {
        **state,
        "response": response,
        "action_type": "workflow_started",
        "workflow_type": workflow_type,
        "act_ms": total_ms,
    }
```

#### Step 2C: Modify classify_intent to Detect Workflows

**File:** `atlas_brain/agents/graphs/atlas.py`

**Modify:** `classify_intent` function

**Insert before line 228 (before return statement):**
```python
    # Check for workflow initiation BEFORE returning
    workflow_type = _detect_workflow_intent(input_text)
    if workflow_type:
        logger.info("Detected workflow intent: %s", workflow_type)
        return {
            **state,
            "action_type": "workflow_start",
            "workflow_to_start": workflow_type,
            "classify_ms": (time.perf_counter() - start_time) * 1000,
        }
```

#### Step 2D: Update Graph Structure

**File:** `atlas_brain/agents/graphs/atlas.py`

**Modify:** `build_atlas_agent_graph` function

**Add new node (after line 769):**
```python
    graph.add_node("start_workflow", start_workflow)
```

**Modify routing (update route_after_classify):**
```python
def route_after_classify(
    state: AtlasAgentState,
) -> Literal["delegate_home", "retrieve_memory", "execute", "respond", "start_workflow"]:
    """Route based on classification result."""
    action_type = state.get("action_type", "conversation")

    # Mode switch already handled
    if action_type == "mode_switch":
        return "respond"

    # NEW: Start workflow if detected
    if action_type == "workflow_start":
        return "start_workflow"

    # ... rest of existing logic ...
```

**Add edge (after line 795):**
```python
    graph.add_edge("start_workflow", END)
```

**Update conditional_edges (line 787-795):**
```python
    graph.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "delegate_home": "delegate_home",
            "retrieve_memory": "retrieve_memory",
            "execute": "execute",
            "respond": "respond",
            "start_workflow": "start_workflow",  # NEW
        },
    )
```

---

### Phase 3: Increase Conversation Timeout (Separate PR)

**File:** `atlas_brain/.env`

**Change:**
```
ATLAS_VOICE_CONVERSATION_TIMEOUT_MS=12000  # Was 8000
```

---

## Validation Plan

### Phase 1 Validation:
1. Run: `python -c "from atlas_brain.tools import tool_registry; print([t.name for t in tool_registry.list_tools()])"`
2. Verify removed tools are NOT in list
3. Verify kept tools (weather, time, etc.) ARE in list
4. Run: `python -c "from atlas_brain.agents.graphs.booking import run_booking_workflow; print('OK')"`
5. Verify workflows can still import their tools

### Phase 2 Validation:
1. Start Atlas voice pipeline
2. Say "Hey Atlas, book an appointment"
3. Verify: Atlas asks for details (name, phone, etc.)
4. Say "Hey Atlas, remind me to call mom tomorrow"
5. Verify: Reminder is created OR clarification asked

---

## Affected Files Summary

| File | Phase | Changes |
|------|-------|---------|
| `atlas_brain/tools/__init__.py` | 1 | Remove 12 tool registrations |
| `atlas_brain/agents/graphs/atlas.py` | 2 | Add workflow detection + routing |
| `.env` | 3 | Update timeout (optional) |

---

## Rollback Plan

If issues occur:
1. Phase 1: Re-add `tool_registry.register()` calls
2. Phase 2: Remove workflow detection code, revert routing changes

---

## Session Context

- Multi-turn conversations work (tested earlier)
- PostgreSQL storing conversation_turns (354 turns)
- Graphiti/Neo4j running for memory
- ASR server running on port 8081
- LLM (qwen3-30b-a3b) loaded in Ollama

---

## Approval Checklist

- [x] Phase 1 plan approved
- [x] Phase 2 plan approved
- [x] Phase 3 plan approved
- [x] Ready to implement

---

## Implementation Complete

### Phase 1 Verified (2026-02-02)
- Removed 12 workflow tools from `tool_registry` in `atlas_brain/tools/__init__.py`
- 24 info/utility tools remain registered (weather, time, security, presence, etc.)
- All 4 workflows still import their tools directly - no breaking changes

### Phase 2 Verified (2026-02-02)
- Added workflow detection patterns in `atlas_brain/agents/graphs/atlas.py`:
  - `_BOOKING_PATTERNS`, `_REMINDER_PATTERNS`, `_EMAIL_PATTERNS`, `_CALENDAR_PATTERNS`
  - `_detect_workflow_intent()` function (lines 89-111)
- Modified `classify_intent()` to check workflows before generic classification (lines 280-291)
- Added `start_workflow()` node function (lines 654-700)
- Updated `route_after_classify()` to route "workflow_start" (lines 792-804)
- Updated graph builder with new node and edges (lines 882, 914, 939)

### Validation Results
```
Workflow detection: 8/8 tests PASS
Graph nodes: 10 nodes including start_workflow
Workflow imports: 4/4 workflows import OK
Tool registry: 12/12 workflow tools removed
```

### Phase 3 Verified (2026-02-02)
- Updated `.env` line 214: `ATLAS_VOICE_CONVERSATION_TIMEOUT_MS=12000` (was 8000)
- Config loads via `env_prefix="ATLAS_VOICE_"` in `atlas_brain/config.py:531`
- Provides 12 seconds for user to respond in conversation mode (was 8 seconds)

### Booking UX Improvement (2026-02-02)
- Added LLM-based field extraction (`extract_field_with_llm()`) in `booking.py`
- Sequential field collection: name -> address -> date -> time
- Added `collecting_field` to `BookingWorkflowState` for tracking
- Updated `handle_missing_info` to ask for one field at a time
- Updated `merge_continuation_input` to use LLM extraction
- Updated routing functions for sequential flow

**Files changed:**
- `atlas_brain/agents/graphs/state.py` - Added `collecting_field` field
- `atlas_brain/agents/graphs/booking.py` - LLM extraction, sequential prompts

### Next Steps
- Restart voice pipeline to test sequential booking flow
- Monitor LLM extraction latency per turn
- Monitor for any edge cases in workflow detection patterns
