# Hybrid Tool Router Implementation

## Status: PHASE 1 COMPLETE
## Date: 2026-01-20
## Session Context: Two-tier routing with DistilBERT intent classifier

---

## Problem Statement

Current voice/text query flow uses LLM-based IntentParser for ALL queries (~300ms), even simple tool queries like "what time is it" that could be handled faster.

**Current Flow:**
```
Query -> IntentParser (LLM ~300ms) -> classify -> execute -> respond
Total for simple tool: ~600ms+
```

**Goal - Hybrid Flow with DistilBERT Intent Router First:**
```
Query -> DistilBERT (~3ms after warmup)
  |
  +-> Tool detected (high confidence)
  |     |-> Direct tool execution (~80ms) = ~85ms total
  |
  +-> Device command detected
  |     |-> IntentParser (LLM ~300ms) for device details
  |
  +-> Conversation/Low confidence
        |-> IntentParser or direct LLM
```

---

## Phase 1 Completed (2026-01-20)

### Changes Made

**1. Config: `atlas_brain/config.py`**
- Added `IntentRouterConfig` class (lines 593-616)
- Added `intent_router` field to Settings (line 686)
- Environment variables: `ATLAS_INTENT_ROUTER_ENABLED`, `ATLAS_INTENT_ROUTER_MODEL_ID`, etc.

**2. Service: `atlas_brain/services/intent_router.py` (NEW)**
- `IntentRouteResult` dataclass - route result with category, label, confidence
- `LABEL_TO_CATEGORY` mapping - 60 MASSIVE labels to our 3 categories
- `IntentRouter` class - wraps DistilBERT classifier
- Singleton pattern with `get_intent_router()`

**3. AgentTools: `atlas_brain/agents/tools.py`**
- Added `route_intent()` method (lines 128-153)
- TYPE_CHECKING import for IntentRouteResult

### Model Used
- `joaobarroca/distilbert-base-uncased-finetuned-massive-intent-detection-english`
- 65M parameters, DistilBERT architecture
- Fine-tuned on Amazon MASSIVE dataset (60 intents)

### Test Results
```
Query                          Category        Confidence   Time
-----------------------------------------------------------------
"what time is it"              tool_use        0.98         3ms
"turn on the lights"           device_command  0.54         3ms
"what's the weather"           tool_use        0.99         2ms
"hello how are you"            conversation    0.34         2ms
"set a reminder for 5pm"       tool_use        0.99         2ms
"tell me a joke"               conversation    0.91         2ms
```

### Configuration
```bash
# .env
ATLAS_INTENT_ROUTER_ENABLED=true
ATLAS_INTENT_ROUTER_CONFIDENCE_THRESHOLD=0.5
# ATLAS_INTENT_ROUTER_MODEL_ID=... (optional, has default)
# ATLAS_INTENT_ROUTER_DEVICE=... (optional, auto-detect)
```

---

---

## Verification Summary

### Files That Reference action_type (VERIFIED 2026-01-20)

| File | Lines | Impact |
|------|-------|--------|
| `agents/protocols.py` | 96, 160 | Definition - update comment only |
| `agents/atlas.py` | 175, 204, 241, 251, 261, 265, 269, 273, 392, 419, 466, 470, 712 | **MODIFY** |
| `agents/home.py` | 103, 153, 163, 173, 182, 185, 208, 234, 278, 282 | **MODIFY** |
| `agents/receptionist.py` | 224, 243, 248, 268, 289, 418 | No change needed (different patterns) |
| `agents/base.py` | 161, 251, 316, 325 | No change needed (fallback handling) |
| `api/query/text.py` | 54, 55, 64 | Read-only (logging/response) |
| `api/devices/control.py` | 178, 179, 187 | Read-only (logging/response) |
| `api/llm.py` | 158, 159 | Read-only (logging) |

### Existing Components (VERIFIED 2026-01-20)

| Component | Location | Status |
|-----------|----------|--------|
| `FunctionGemmaRouter` | `pipecat/router.py:64` | EXISTS - ready to use |
| `ToolRouterProcessor` | `pipecat/router.py:344` | EXISTS - ready to use |
| `is_complex_query()` | `pipecat/router.py:307` | EXISTS - ready to use |
| `is_device_command()` | `agents/tools.py:108` | EXISTS - keyword check |
| `GptOssToolService` | `pipecat/llm.py:42` | EXISTS - multi-turn tools |

### Current action_type Values (VERIFIED)
- `"none"` - no action
- `"device_command"` - device control via Home Assistant
- `"tool_use"` - tool query (time, weather, etc.)
- `"conversation"` - LLM chat
- `"mode_switch"` - special case for mode changes

### New action_type Values (PROPOSED)
- `"simple_tool"` - FunctionGemma routed, direct execution
- `"complex_tool"` - gpt-oss multi-turn tool calling

### Breaking Change Analysis
- **NONE** - action_type is a string, not an enum
- Unknown action_types fall through to default handlers in base.py
- API consumers will see new values but handle gracefully

---

## Phased Implementation Plan

### Phase 1: Add FunctionGemma Integration to AgentTools
**File:** `atlas_brain/agents/tools.py`

**Changes:**
1. Add import for FunctionGemma router
2. Add async method `route_with_functiongemma(query: str)`
3. Add `is_complex_query()` wrapper

**Insertion Point:** After line 123 (after `is_device_command`)

**Dependencies:** `pipecat/router.py` (already exists)

**Validation:** Unit test the new method in isolation

---

### Phase 2: Update ThinkResult Documentation
**File:** `atlas_brain/agents/protocols.py`

**Changes:**
1. Update comment on line 96 to include new action_types

**Current (line 96):**
```python
action_type: str  # "conversation", "device_command", "tool_use", "none"
```

**New:**
```python
action_type: str  # "conversation", "device_command", "tool_use", "simple_tool", "complex_tool", "none"
```

**Breaking Changes:** None (documentation only)

---

### Phase 3: Modify AtlasAgent._do_think()
**File:** `atlas_brain/agents/atlas.py`

**Current Flow (lines 192-308):**
```python
async def _do_think(...):
    intent = await tools.parse_intent(...)  # Always calls LLM
    if intent:
        # classify based on intent
```

**New Flow:**
```python
async def _do_think(...):
    # Step 1: FunctionGemma classification (fast)
    route_result = await tools.route_with_functiongemma(context.input_text)

    if route_result.needs_tool:
        if is_complex_query(context.input_text):
            result.action_type = "complex_tool"
            # Store route info for act phase
        else:
            result.action_type = "simple_tool"
            result.tool_name = route_result.tool_name
            result.tool_args = route_result.tool_args
        return result

    # Step 2: Check for device command (keyword check - instant)
    if tools.is_device_command(context.input_text):
        # Use existing IntentParser for device details
        intent = await tools.parse_intent(context.input_text)
        # ... existing device command handling

    # Step 3: Conversation fallback
    result.action_type = "conversation"
```

**Insertion Point:** Replace lines 208-275 in `_do_think()`

**Dependencies:**
- Phase 1 complete
- Phase 2 complete

---

### Phase 4: Modify AtlasAgent._do_act()
**File:** `atlas_brain/agents/atlas.py`

**Current (lines 373-447):**
- Handles `device_command` and `tool_use`

**New Additions:**
- Add handler for `simple_tool` (direct execution via tool_registry)
- Add handler for `complex_tool` (skip - handled in respond)

**Insertion Point:** After line 444, before `result.duration_ms`

---

### Phase 5: Modify AtlasAgent._do_respond()
**File:** `atlas_brain/agents/atlas.py`

**Current (lines 449-480):**
- Handles `device_command`, `tool_use`, conversation

**New Additions:**
- Add handler for `simple_tool` (return tool message)
- Add handler for `complex_tool` (route to gpt-oss)

**Insertion Point:** After line 477, before conversation fallback

---

### Phase 6: Apply Same Changes to HomeAgent
**File:** `atlas_brain/agents/home.py`

Apply identical pattern as Phase 3-5 to HomeAgent._do_think(), _do_act(), _do_respond()

---

### Phase 7: Validation & Testing

1. **Unit Test FunctionGemma routing:**
   ```bash
   python -c "from atlas_brain.agents.tools import get_agent_tools; ..."
   ```

2. **Integration Test via API:**
   ```bash
   curl -X POST http://localhost:8000/api/v1/query/text \
     -H "Content-Type: application/json" \
     -d '{"query_text": "what time is it"}'
   ```

3. **Verify Latency Improvement:**
   - Before: ~600ms for simple tool
   - After: ~200ms for simple tool

4. **Verify No Regression:**
   - Device commands still work
   - Complex queries still work
   - Conversation still works

---

## Exact Code Insertion Points

### tools.py - Phase 1
```
Line 123: End of is_device_command()
Insert after line 123, before line 125 (# Action execution comment)
```

### protocols.py - Phase 2
```
Line 96: action_type comment
Modify in place
```

### atlas.py - Phase 3
```
Lines 208-275: Current _do_think() logic
Replace with new hybrid flow
```

### atlas.py - Phase 4
```
Line 444: End of tool_use handler in _do_act()
Insert new handlers after
```

### atlas.py - Phase 5
```
Line 477: End of tool_use handler in _do_respond()
Insert new handlers after
```

### home.py - Phase 6
```
Lines 110-187: Current _do_think() logic
Lines 208-257: Current _do_act() logic
Lines 278-289: Current _do_respond() logic
```

---

## Rollback Plan

If issues occur:
1. Revert changes to atlas.py and home.py
2. FunctionGemma router remains available but unused
3. System falls back to current IntentParser flow

---

## Session Continuity Notes

- FunctionGemma model: `google/functiongemma-270m-it`
- gpt-oss model: `gpt-oss:20b` via Ollama
- Config flag: `ATLAS_LOAD_TOOL_ROUTER_ON_STARTUP=true`
- Router already loads in main.py:169-177

---

## Next Steps (Awaiting Approval)

1. [ ] Approve Phase 1-7 implementation plan
2. [ ] Implement Phase 1 (AgentTools integration)
3. [ ] Test Phase 1 in isolation
4. [ ] Proceed to Phase 2-5
5. [ ] Test full flow
6. [ ] Apply to HomeAgent (Phase 6)
7. [ ] Final validation
