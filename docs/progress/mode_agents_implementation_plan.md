# Mode Agents Implementation Plan

**Date:** 2026-01-19
**Status:** AWAITING APPROVAL

---

## Current State Analysis

### Existing Files

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `atlas_brain/agents/atlas.py` | 723 | Main agent - handles devices, tools, conversation | MODIFY (becomes router) |
| `atlas_brain/agents/receptionist.py` | 580+ | Business booking with state machine | EXISTS (verified working) |
| `atlas_brain/agents/base.py` | 230 | BaseAgent class | NO CHANGE |
| `atlas_brain/agents/protocols.py` | 130 | AgentContext, ThinkResult, ActResult | NO CHANGE |
| `atlas_brain/agents/__init__.py` | 113 | Exports | MODIFY (add HomeAgent) |
| `atlas_brain/modes/manager.py` | 185 | Mode tracking, switching | MODIFY (add timeout) |
| `atlas_brain/modes/config.py` | 173 | Mode definitions | NO CHANGE |

### Files Using AtlasAgent (Must Not Break)

| File | Usage |
|------|-------|
| `atlas_brain/pipecat/agent_processor.py` | `create_atlas_agent()` → `agent.run(context)` |
| `atlas_brain/api/devices/control.py` | `create_atlas_agent()` → `agent.run(context)` |
| `atlas_brain/api/llm.py` | `create_atlas_agent()` → `agent.run(context)` |
| `atlas_brain/api/query/text.py` | `create_atlas_agent()` → `agent.run(context)` |
| `atlas_brain/pipecat/pipeline.py` | `create_atlas_agent()` |

**Interface Contract (MUST PRESERVE):**
```python
agent = create_atlas_agent(session_id="...")
result = await agent.run(AgentContext(...))
# result.response_text, result.success, result.action_results
```

---

## Phased Implementation

### Phase 1: Add Timeout to ModeManager

**File:** `atlas_brain/modes/manager.py`

**Changes:**
1. Add `_last_activity: float` attribute
2. Add `MODE_TIMEOUT_SECONDS = 120` constant
3. Add `update_activity()` method
4. Add `check_timeout()` method
5. Add `has_active_workflow: bool` property for workflow protection

**Insertion Points:**
- Line 40 (in `__init__`): Add `_last_activity` initialization
- After line 77: Add timeout methods

**Verification:**
- Unit test: timeout triggers after 2 mins inactivity
- Unit test: timeout does NOT trigger if `has_active_workflow=True`

---

### Phase 2: Create HomeAgent

**New File:** `atlas_brain/agents/home.py`

**Purpose:** Handle device commands (lights, TV, scenes)

**Structure:**
```python
class HomeAgent(BaseAgent):
    """Home device control agent - fast path, no state machine."""

    def __init__(self, session_id=None):
        super().__init__(name="home", description="Home device control")
        self._capabilities = ["device_control"]

    async def _do_think(self, context):
        # Parse device intent
        # Return ThinkResult with action_type="device_command"

    async def _do_act(self, context, think_result):
        # Execute device command via tools

    async def _do_respond(self, context, think_result, act_result):
        # Generate response (LLM or template)
```

**Logic to Extract from AtlasAgent:**
- Lines 166-187: Device action detection
- Lines 320-346: Device command execution
- Lines 375-376: Device response generation

**Verification:**
- Test: "Turn on the living room lights" → Device command executed
- Test: "What's the TV status" → Query executed

---

### Phase 3: Refactor AtlasAgent as Router

**File:** `atlas_brain/agents/atlas.py`

**Changes:**
1. Add `_mode_manager` attribute
2. Add `_mode_agents` dict mapping ModeType → Agent
3. Add `_workflow_state` for tracking active workflows
4. Modify `run()` to:
   - Check mode timeout
   - Check for mode switch command
   - Delegate to appropriate mode agent
5. Keep `create_atlas_agent()` interface unchanged

**Key Insertion Points:**
- Line 59-78 (`__init__`): Add mode manager, mode agents dict
- Before line 120 (`_do_think`): Add routing logic in overridden `run()`

**New `run()` method (override base):**
```python
async def run(self, context: AgentContext) -> AgentResult:
    # 1. Update activity timestamp
    self._mode_manager.update_activity()

    # 2. Check timeout (skip if workflow active)
    if not self._workflow_state:
        self._mode_manager.check_timeout()

    # 3. Check for mode switch command
    mode_switch = self._mode_manager.parse_mode_switch(context.input_text)
    if mode_switch:
        self._mode_manager.switch_mode(mode_switch)
        return AgentResult(
            success=True,
            response_text=f"Switched to {mode_switch.value} mode.",
            action_type="mode_switch",
        )

    # 4. Delegate to current mode agent
    current_mode = self._mode_manager.current_mode
    agent = self._mode_agents.get(current_mode)

    if agent:
        return await agent.run(context)
    else:
        # Fallback to self (original behavior)
        return await super().run(context)
```

**Verification:**
- Test: Mode switch command works
- Test: HOME mode delegates to HomeAgent
- Test: RECEPTIONIST mode delegates to ReceptionistAgent
- Test: Timeout works after 2 mins

---

### Phase 4: Update Exports

**File:** `atlas_brain/agents/__init__.py`

**Changes:**
- Add import for HomeAgent
- Add to `__all__` list

**Insertion Points:**
- After line 70: Add HomeAgent import
- Line 73+ (`__all__`): Add HomeAgent exports

**Verification:**
- `from atlas_brain.agents import HomeAgent` works

---

### Phase 5: Integration Testing

**Tests to Run:**
1. Voice: "Atlas switch to home mode" → Mode switches
2. Voice: "Turn on the lights" (HOME mode) → HomeAgent handles
3. Voice: "Atlas switch to scheduling mode" → RECEPTIONIST mode
4. Voice: "Book an estimate" → ReceptionistAgent handles (state machine)
5. API: `/query/text` still works
6. API: `/devices/intent` still works
7. Timeout: 2 min inactivity → Returns to HOME

---

## Files Changed Summary

| Phase | File | Action | Risk |
|-------|------|--------|------|
| 1 | `modes/manager.py` | MODIFY - add timeout | LOW |
| 2 | `agents/home.py` | CREATE | LOW |
| 3 | `agents/atlas.py` | MODIFY - add routing | MEDIUM |
| 4 | `agents/__init__.py` | MODIFY - add export | LOW |

---

## Breaking Change Analysis

### Will NOT Break:
- `create_atlas_agent()` - Same interface
- `agent.run(context)` - Same interface
- `AgentResult` - Same structure
- All API endpoints - They use `create_atlas_agent()`

### Requires Attention:
- AtlasAgent internal methods called directly (none found externally)
- Mode-specific behavior changes (HOME now delegates to HomeAgent)

---

## Rollback Strategy

Each phase is independent:

1. **Phase 1 fails:** Revert manager.py, timeout disabled
2. **Phase 2 fails:** Delete home.py, no impact
3. **Phase 3 fails:** Revert atlas.py, routing disabled
4. **Phase 4 fails:** Revert __init__.py, HomeAgent not exported

---

## Open Items Before Implementation

1. **Confirm:** Should AtlasAgent keep original logic as fallback if mode agent not found?
2. **Confirm:** Should mode switch response be customizable per mode?
3. **Confirm:** Should we log mode switches to conversation history?

---

*Awaiting approval to proceed with Phase 1*
