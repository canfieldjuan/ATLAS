# Multi-Turn Conversational Workflows - Progress Log

## Overview
Enable natural multi-turn slot-filling conversations for tool workflows instead of requiring all parameters upfront.

**Goal:** User says "I want to schedule an appointment" → Atlas asks for name → user gives name → Atlas asks for phone → etc.

**Start Date:** 2026-01-31
**Status:** Phase 5 Complete - All Workflows Done (Booking, Reminder, Email, Calendar)

---

## Current State Analysis

### Voice Pipeline Session Architecture (Verified Working)
| Component | Current Behavior |
|-----------|-----------------|
| session_id | Stable across turns (assigned at voice stream start) |
| Conversation history | Persisted to DB after each turn via _store_turn() |
| Turn continuity | Handled by voice pipeline (VAD → STT → agent → TTS) |
| Session timeout | Managed by voice pipeline, NOT by workflows |

**Key Insight:** The voice pipeline already handles multi-turn. The gap is in workflow state persistence between turns.

### Key Files Analyzed
| File | Location | Current State |
|------|----------|---------------|
| SessionRepository | `storage/repositories/session.py` | UPDATED - has update_metadata + clear_metadata_key |
| Session model | `storage/models.py` | Has metadata JSONB field |
| Booking workflow | `agents/graphs/booking.py` | Sets awaiting_user_input=True but ENDS |
| Atlas router | `agents/graphs/atlas.py` | No workflow state restoration |

### Current Booking Workflow Gap

```python
# booking.py:529-563 - handle_missing_info
async def handle_missing_info(state: BookingWorkflowState) -> BookingWorkflowState:
    needs_info = state.get("needs_info", [])

    if "customer_identifier" in needs_info:
        response = "Could you please tell me your name or phone number?"
    elif "date" in needs_info:
        response = "What date would you like to book?"
    # ...

    return {
        **state,
        "response": response,
        "awaiting_user_input": True,  # Signals we need more info
        "current_step": "awaiting_info",
    }
    # BUT... graph ENDS here! Next user turn starts fresh.
```

**Problem:** Workflow ends after asking clarifying question. Next user turn starts fresh graph - loses partial state.

---

## Proposed Solution: Session-Based Workflow State Persistence

### Architecture

```
Turn 1: "I want to schedule an appointment"
    │
    ├── atlas.py: classify → tool_use (booking)
    ├── booking.py: parse_request → handle_missing_info
    ├── Response: "Could you please tell me your name?"
    ├── *** SAVE STATE TO session.metadata.active_workflow ***
    └── Return to voice pipeline

Turn 2: "John Smith"
    │
    ├── atlas.py: check session.metadata.active_workflow
    │   └── Found! workflow_type=booking, current_step=awaiting_info
    ├── *** RESTORE STATE + MERGE NEW INPUT ***
    ├── booking.py: continue from awaiting_info
    ├── Response: "What date would you like?"
    ├── *** UPDATE session.metadata.active_workflow ***
    └── Return to voice pipeline

Turn 3: "Tomorrow at 2pm"
    │
    ├── atlas.py: check session.metadata.active_workflow
    ├── *** RESTORE STATE + MERGE NEW INPUT ***
    ├── booking.py: continue → check_availability → book_appointment
    ├── Response: "Booked! Confirmation #ABC123"
    ├── *** CLEAR session.metadata.active_workflow ***
    └── Done
```

### State Storage Schema

```python
# session.metadata structure
{
    "active_workflow": {
        "workflow_type": "booking",  # or "email", "reminder", etc.
        "current_step": "awaiting_info",
        "started_at": "2026-01-31T10:00:00Z",
        "partial_state": {
            "customer_name": "John Smith",  # Filled so far
            "customer_phone": None,         # Still needed
            "requested_date": None,
            "requested_time": None,
            "needs_info": ["phone", "date", "time"],
        },
        "conversation_context": [
            {"role": "user", "content": "I want to schedule an appointment"},
            {"role": "assistant", "content": "Could you please tell me your name?"},
            {"role": "user", "content": "John Smith"},
        ],
    }
}
```

---

## Phased Implementation Plan

### Phase 1: SessionRepository Metadata Update - COMPLETE
**Goal:** Add update_metadata method to SessionRepository

**Tasks:**
- [x] Add update_metadata(session_id: UUID, metadata: dict) method
- [x] Support partial updates (merge, not replace)
- [x] Add clear_metadata_key(session_id: UUID, key: str) method
- [x] Add tests for new methods

**Files modified:**
- `atlas_brain/storage/repositories/session.py` (lines 206-268)
- `tests/test_session_management.py` (added TestSessionMetadata class)
- `verify_session_metadata.py` (standalone verification script)

**Verification:**
```python
# Test update_metadata
repo = get_session_repo()
session = await repo.get_session(session_id)
await repo.update_metadata(session_id, {"active_workflow": {...}})
session = await repo.get_session(session_id)
assert session.metadata.get("active_workflow") is not None
```

---

### Phase 2: WorkflowStateManager - COMPLETE
**Goal:** Create utility for saving/restoring workflow state to session.metadata

**Tasks:**
- [x] Create WorkflowStateManager class in agents/graphs/workflow_state.py
- [x] save_workflow_state(session_id, workflow_type, step, partial_state)
- [x] restore_workflow_state(session_id) -> Optional[WorkflowState]
- [x] clear_workflow_state(session_id)
- [x] add_context_turn(session_id, role, content)
- [x] Handle workflow timeout (auto-clear after N minutes)
- [x] update_partial_state(session_id, updates, new_step) - bonus method

**Files created:**
- `atlas_brain/agents/graphs/workflow_state.py` (270 lines)
- `verify_workflow_state.py` (standalone verification script)

**Exports added to `__init__.py`:**
- ActiveWorkflowState, WorkflowStateManager, get_workflow_state_manager

**Verification:**
```python
manager = WorkflowStateManager()
await manager.save_workflow_state(
    session_id="abc-123",
    workflow_type="booking",
    step="awaiting_info",
    partial_state={"customer_name": "John"},
)
restored = await manager.restore_workflow_state("abc-123")
assert restored.workflow_type == "booking"
```

---

### Phase 3: Booking Workflow Multi-Turn Support - COMPLETE
**Goal:** Modify booking.py to persist state and continue from saved state

**Tasks:**
- [x] Add check_continuation node to detect saved workflows
- [x] Add merge_continuation_input node to merge new input with saved state
- [x] Modify handle_missing_info to save workflow state
- [x] Modify handle_customer_not_found to save workflow state
- [x] Modify suggest_alternatives to save workflow state
- [x] Modify confirm_booking to clear workflow state on success
- [x] Add is_continuation flag to BookingWorkflowState
- [x] Add routing for continuation vs new workflow

**Files to modify:**
- `atlas_brain/agents/graphs/booking.py`
- `atlas_brain/agents/graphs/state.py` (add is_continuation, previous_state fields)

**New Graph Flow:**
```
Entry: check_continuation
    │
    ├── [new request] → parse_request → ...
    │
    └── [continuation] → restore_state → merge_input → route_to_next_step → ...
```

**Verification:**
```python
# Turn 1
result1 = await run_booking_workflow("schedule appointment", session_id="test-123")
assert result1["awaiting_user_input"] == True
assert result1["response"] == "Could you please tell me your name?"

# Turn 2 (continuation)
result2 = await run_booking_workflow("John Smith", session_id="test-123")
assert "date" in result2["response"].lower()
```

---

### Phase 4: Atlas Router Integration - COMPLETE
**Goal:** Atlas router detects active workflows and routes to them

**Tasks:**
- [x] Add check_active_workflow node before classify_intent
- [x] If active workflow exists, bypass normal classification
- [x] Route directly to workflow continuation handler
- [x] Add cancel detection ("nevermind", "cancel", "stop")
- [x] Handle workflow timeouts gracefully

**Files modified:**
- `atlas_brain/agents/graphs/atlas.py`
- `atlas_brain/agents/graphs/state.py` (added active_workflow field)

**New Routing Logic:**
```python
async def check_active_workflow(state: AtlasAgentState) -> AtlasAgentState:
    """Check if session has active workflow to continue."""
    session_id = state.get("session_id")
    if not session_id:
        return state

    manager = WorkflowStateManager()
    workflow = await manager.restore_workflow_state(session_id)

    if workflow:
        # Check for cancel intent
        if is_cancel_intent(state["input_text"]):
            await manager.clear_workflow_state(session_id)
            return {**state, "response": "Okay, I've cancelled that."}

        return {
            **state,
            "active_workflow": workflow,
            "action_type": "workflow_continuation",
        }

    return state
```

**Verification:**
```python
# Start booking workflow
agent = AtlasAgentGraph(session_id="test-123")
result1 = await agent.run("I want to schedule an appointment")
assert "name" in result1["response_text"].lower()

# Continue with name (should NOT go through normal classification)
result2 = await agent.run("John Smith")
assert "date" in result2["response_text"].lower()

# Cancel
result3 = await agent.run("nevermind")
assert "cancel" in result3["response_text"].lower()
```

---

### Phase 5: Additional Workflows - COMPLETE
**Goal:** Apply multi-turn pattern to other workflows

**Candidate workflows:**
- [x] Email workflow (if recipient missing, ask) - COMPLETE
- [x] Reminder workflow (if time missing, ask) - COMPLETE
- [x] Calendar event creation (if date/time missing, ask) - COMPLETE

**Pattern to follow:**
1. Add save_partial_state node after clarification
2. Add merge_user_input node for continuations
3. Export workflow_type constant
4. Register with WorkflowStateManager

**Reminder workflow changes:**
- Added REMINDER_WORKFLOW_TYPE constant
- Added check_continuation and merge_continuation_input nodes
- Modified parse_create_request to save workflow state when clarification needed
- Modified execute_create to clear workflow state on success
- Added is_continuation and restored_from_step fields to ReminderWorkflowState
- Updated build_reminder_graph with new entry point
- Fixed parse_create_intent to handle message-only and time-only inputs
- Registered in atlas.py continue_workflow handler

**Email workflow changes:**
- Added EMAIL_WORKFLOW_TYPE constant
- Added check_continuation and merge_continuation_input nodes
- Modified generate_draft to save workflow state when clarification needed (all 3 cases)
- Modified execute_send_email/estimate/proposal to clear workflow state on success
- Added is_continuation and restored_from_step fields to EmailWorkflowState
- Updated build_email_graph with new entry point
- Added route_after_check_continuation and route_after_merge routing functions
- Registered in atlas.py continue_workflow handler

**Calendar workflow changes:**
- Created new agents/graphs/calendar.py workflow
- Added CALENDAR_WORKFLOW_TYPE constant
- Added CalendarWorkflowState to state.py
- Added check_continuation and merge_continuation_input nodes
- Added parse_create_request to save workflow state when clarification needed
- Added execute_create to clear workflow state on success
- Added create_event method to tools/calendar.py
- Added routing functions and build_calendar_graph
- Registered in atlas.py continue_workflow handler
- Created verify_calendar_multiturn.py verification script
- All 6 tests pass

---

### Phase 6: Testing & Validation
**Goal:** Full end-to-end testing

**Tasks:**
- [ ] Unit tests for SessionRepository.update_metadata
- [ ] Unit tests for WorkflowStateManager
- [ ] Integration tests for booking workflow continuation
- [ ] Integration tests for atlas router with active workflows
- [ ] Voice pipeline integration test (manual)
- [ ] Test workflow timeout behavior
- [ ] Test cancel detection

**Test file to create:**
- `test_multiturn_workflows.py`

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking existing workflows | All changes additive; existing single-turn still works |
| Session.metadata conflicts | Namespace under "active_workflow" key |
| Workflow state too large | Only store essential fields, not full LangGraph state |
| Stale workflows | Auto-expire after 5 minutes of inactivity |
| Voice pipeline timeout interference | Workflow uses session.metadata, not voice pipeline state |

---

## Success Criteria

1. [ ] User can say "schedule appointment" → Atlas asks for name
2. [ ] User gives name → Atlas asks for date/time
3. [ ] User gives date/time → booking is created
4. [ ] User can cancel mid-workflow with "nevermind"
5. [ ] Workflow auto-clears after 5 minutes of inactivity
6. [ ] Existing single-turn workflows still work unchanged
7. [ ] All tests pass

---

## Session Log

### Session 1 - 2026-01-31
- Analyzed existing codebase structure
- Identified key gap: booking.py sets awaiting_user_input but ends graph
- Confirmed session.metadata exists but SessionRepository lacks update_metadata
- Confirmed voice pipeline session_id is stable across turns
- Created phased implementation plan
- **Status:** Awaiting user approval before implementation

### Session 2 - 2026-01-31
- Implemented Phase 1: SessionRepository Metadata Update
- Added update_metadata() method with JSONB merge (lines 206-240)
- Added clear_metadata_key() method (lines 242-268)
- Added TestSessionMetadata class to tests/test_session_management.py
- Created verify_session_metadata.py for standalone verification
- Verified imports and method availability
- **Status:** Phase 1 COMPLETE

### Session 3 - 2026-02-01
- Implemented Phase 2: WorkflowStateManager
- Created workflow_state.py with ActiveWorkflowState dataclass
- Implemented save/restore/clear/add_context_turn/update_partial_state methods
- Added is_expired() with configurable timeout (default 5 minutes)
- Used timezone-aware datetimes (datetime.now(timezone.utc))
- Updated __init__.py exports
- Created verify_workflow_state.py for standalone verification
- All 6 tests pass with real PostgreSQL database (port 5433)
- **Status:** Phase 2 COMPLETE

### Session 4 - 2026-02-01
- Implemented Phase 3: Booking Workflow Multi-Turn Support
- Added is_continuation and restored_from_step fields to BookingWorkflowState
- Added check_continuation node to detect saved workflows
- Added merge_continuation_input node to merge new input with saved state
- Modified handle_missing_info to save workflow state
- Modified handle_customer_not_found to save workflow state
- Modified suggest_alternatives to save workflow state
- Modified confirm_booking to clear workflow state on success
- Added route_after_check_continuation and route_after_merge routing functions
- Updated build_booking_graph with new entry point (check_continuation)
- Created verify_booking_multiturn.py for verification
- All 5 tests pass with real PostgreSQL database
- **Status:** Phase 3 COMPLETE

### Session 5 - 2026-02-01
- Implemented Phase 4: Atlas Router Integration
- Added imports for get_workflow_state_manager and run_booking_workflow
- Added _CANCEL_PATTERNS and _is_cancel_intent helper function
- Added check_active_workflow node to detect saved workflows
- Added continue_workflow node to run booking workflow
- Added route_after_check_workflow routing function
- Updated build_atlas_agent_graph with new nodes and edges
- Added active_workflow field to AtlasAgentState
- Created verify_atlas_router_multiturn.py for verification
- All 6 tests pass with real PostgreSQL database
- **Status:** Phase 4 COMPLETE

### Session 6 - 2026-02-01
- Implemented Phase 5: Reminder Workflow Multi-Turn Support
- Added REMINDER_WORKFLOW_TYPE constant to reminder.py
- Added check_continuation and merge_continuation_input nodes
- Modified parse_create_request to save workflow state when clarification needed
- Modified execute_create to clear workflow state on success
- Added is_continuation and restored_from_step fields to ReminderWorkflowState
- Updated build_reminder_graph with new entry point and routing
- Fixed parse_create_intent to properly handle message-only and time-only inputs
- Registered reminder workflow in atlas.py continue_workflow handler
- Created verify_reminder_multiturn.py for verification
- All 5 tests pass with real PostgreSQL database
- **Status:** Phase 5 IN PROGRESS (Reminder complete, Email pending)

### Session 7 - 2026-02-01
- Implemented Phase 5: Calendar Workflow Multi-Turn Support
- Created new agents/graphs/calendar.py workflow file
- Added CALENDAR_WORKFLOW_TYPE constant
- Added CalendarWorkflowState to state.py
- Added create_event method to tools/calendar.py for event creation
- Added check_continuation and merge_continuation_input nodes
- Added parse_create_request to save workflow state when clarification needed
- Added execute_create to clear workflow state on success
- Fixed parse_create_intent to filter generic titles and reorder patterns
- Registered calendar workflow in atlas.py continue_workflow handler
- Updated __init__.py exports
- Created verify_calendar_multiturn.py for verification
- All 6 tests pass with real PostgreSQL database
- **Status:** Phase 5 COMPLETE (All workflows done)

---

## Files to Create/Modify Summary

### New Files
| File | Purpose | Status |
|------|---------|--------|
| `atlas_brain/agents/graphs/workflow_state.py` | WorkflowStateManager | DONE |
| `atlas_brain/agents/graphs/calendar.py` | Calendar workflow with multi-turn | DONE |
| `verify_session_metadata.py` | Phase 1 verification | DONE |
| `verify_workflow_state.py` | Phase 2 verification | DONE |
| `verify_booking_multiturn.py` | Phase 3 verification | DONE |
| `verify_atlas_router_multiturn.py` | Phase 4 verification | DONE |
| `verify_reminder_multiturn.py` | Phase 5 reminder verification | DONE |
| `verify_email_multiturn.py` | Phase 5 email verification | DONE |
| `verify_calendar_multiturn.py` | Phase 5 calendar verification | DONE |
| `test_multiturn_workflows.py` | Full test suite | Pending |

### Modified Files
| File | Changes | Status |
|------|---------|--------|
| `atlas_brain/storage/repositories/session.py` | Add update_metadata method | DONE |
| `tests/test_session_management.py` | Add TestSessionMetadata class | DONE |
| `atlas_brain/agents/graphs/__init__.py` | Add workflow_state exports | DONE |
| `atlas_brain/agents/graphs/booking.py` | Add continuation support | DONE |
| `atlas_brain/agents/graphs/reminder.py` | Add continuation support | DONE |
| `atlas_brain/agents/graphs/email.py` | Add continuation support | DONE |
| `atlas_brain/agents/graphs/state.py` | Add continuation + CalendarWorkflowState | DONE |
| `atlas_brain/agents/graphs/atlas.py` | Add active workflow check + routing | DONE |
| `atlas_brain/tools/calendar.py` | Add create_event method | DONE |

---

## Implementation Notes

### Why session.metadata instead of separate table?
- session.metadata already exists as JSONB
- Workflow state is transient (minutes, not days)
- Avoids schema migration complexity
- Natural cleanup when session ends

### Why not use LangGraph checkpointing?
- LangGraph checkpointing is designed for long-running async workflows
- Our use case is synchronous voice turns with short gaps
- session.metadata is simpler and already integrated
- We only need to persist ~5 fields, not entire graph state

### Cancel Detection Patterns
```python
CANCEL_PATTERNS = [
    r"^(?:never\s?mind|cancel|stop|forget\s+it|quit)$",
    r"^(?:I\s+)?(?:don'?t\s+)?(?:want\s+to\s+)?cancel",
    r"^stop\s+(?:that|this|booking|scheduling)",
]
```
