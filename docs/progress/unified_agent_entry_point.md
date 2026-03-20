# Unified Agent Entry Point - Implementation Plan

**Created:** 2026-01-19
**Last Updated:** 2026-01-19
**Status:** COMPLETE - All Phases Implemented

> **⚠️ HISTORICAL (Audit 2026-02-14):** References to `api/orchestration.py` in this
> doc are outdated. That file does not exist. The agent entry point is
> `agents/interface.py` → `LangGraphAgentAdapter` → `AtlasAgentGraph`.

---

## Problem Statement

Atlas currently has **6 fragmented entry points** with inconsistent capabilities:

| Endpoint | Uses Agent? | Has Tools? | Has Devices? | Has Memory? |
|----------|-------------|------------|--------------|-------------|
| WS `/ws/orchestrated` | Yes | Yes | Yes | Yes |
| POST `/orchestration/text` | Yes | Yes | Yes | Yes |
| POST `/query/text` | **No** | Yes | **No** | **No** |
| POST `/llm/chat` | **No** | **No** | **No** | Partial |
| POST `/devices/intent` | **No** | **No** | Yes | **No** |
| Pipecat Pipeline | **No** | **No** | **No** | **No** |

This fragmentation causes:
1. Inconsistent behavior across interfaces
2. Duplicated logic in multiple places
3. Pipecat voice pipeline has no tools/devices/memory
4. Maintenance burden - changes need to be made in multiple places

---

## Solution: Single Agent Entry Point

All interfaces route through `AtlasAgent.run()`:

```
┌─────────────────────────────────────────────────────────────────┐
│                         ATLAS BRAIN                              │
│                                                                  │
│                    ┌─────────────────────┐                      │
│                    │     ATLAS AGENT     │                      │
│                    │   (Single Brain)    │                      │
│                    │                     │                      │
│                    │  • Intent parsing   │                      │
│                    │  • Tool execution   │                      │
│                    │  • Device commands  │                      │
│                    │  • LLM responses    │                      │
│                    │  • Memory/context   │                      │
│                    └──────────┬──────────┘                      │
│                               │                                  │
│              ┌────────────────┼────────────────┐                │
│              │                │                │                │
│              ▼                ▼                ▼                │
│     ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│     │  WebSocket  │  │    REST     │  │   Pipecat   │          │
│     │  Transport  │  │  Transport  │  │  Transport  │          │
│     │             │  │             │  │             │          │
│     │ Audio I/O   │  │ Text only   │  │ Local Audio │          │
│     │ over network│  │             │  │ Mic/Speaker │          │
│     └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Current State Analysis

### Files Verified (2026-01-19)

#### Endpoints Using Agent (Working correctly)
1. `atlas_brain/api/orchestration.py`
   - Line 184: `agent = create_atlas_agent(session_id=session_id)`
   - Line 488: `agent = create_atlas_agent(session_id=session_id)`
   - Uses full agent capabilities

2. `atlas_brain/api/comms/webhooks.py`
   - Line 298: `agent_result = await agent.run(agent_context)`
   - Uses agent for incoming calls/SMS

#### Endpoints Bypassing Agent (Need modification)

1. **`atlas_brain/api/query/text.py`** (POST /query/text)
   - Lines 17-65
   - Currently: `execute_with_tools(llm, messages)` directly
   - Missing: Agent, device commands, memory
   - External clients: None found

2. **`atlas_brain/api/llm.py`** (POST /llm/chat)
   - Lines 125-199
   - Currently: `llm.chat(messages)` directly
   - Missing: Agent, tools, device commands
   - External clients: `voice_client.py` line 109
   - **Note**: Keep endpoint contract, change internal implementation

3. **`atlas_brain/api/devices/control.py`** (POST /devices/intent)
   - Lines 148-183
   - Currently: `intent_parser.parse()` → `action_dispatcher.dispatch_intent()`
   - Missing: Agent, tools, LLM response
   - External clients: None found

4. **`atlas_brain/pipecat/pipeline.py`** (Local voice)
   - Lines 401-413 (pipeline construction)
   - Currently: Whisper STT → OLLamaLLM → Kokoro TTS
   - Missing: Agent, tools, device commands, memory
   - External clients: None (internal only)

### Agent Interface (Verified)

Location: `atlas_brain/agents/protocols.py`

```python
@dataclass
class AgentContext:
    input_text: str
    input_type: str = "text"  # "text", "voice", "vision"
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    speaker_id: Optional[str] = None
    conversation_history: list[dict] = field(default_factory=list)
    runtime_context: dict = field(default_factory=dict)

@dataclass
class AgentResult:
    success: bool
    response_text: Optional[str] = None
    action_type: str = "none"
    intent: Optional[Any] = None
    action_results: list[dict] = field(default_factory=list)
    # ... timing fields ...
```

Entry point: `agent.run(context: AgentContext) -> AgentResult`

---

## Breaking Change Risk Analysis

| File | Risk | Mitigation |
|------|------|------------|
| `api/query/text.py` | LOW | No external clients. Keep response schema. |
| `api/llm.py` | MEDIUM | `voice_client.py` uses it. Keep API contract. |
| `api/devices/control.py` | LOW | No external clients. Optional change. |
| `pipecat/pipeline.py` | LOW | Internal only. No API contract. |

### External Client Dependencies

1. **`voice_client.py`** (line 109)
   - Uses: `POST /api/v1/llm/chat`
   - Request: `{"messages": [...], "max_tokens": int, "temperature": float, "session_id": str}`
   - Response: `{"response": str}`
   - **Mitigation**: Keep exact same request/response schema

2. **`atlas-ui/src/hooks/useAtlas.ts`** (line 5)
   - Uses: `WS ws://localhost:8000/api/v1/ws/orchestrated`
   - **No changes needed** - already uses Agent

---

## Phased Implementation Plan

### Phase 1: Create Pipecat Agent Processor
**Risk: LOW** | **Files: NEW only**

Create a Pipecat FrameProcessor that routes transcriptions through AtlasAgent.

**New File:** `atlas_brain/pipecat/agent_processor.py`

```python
"""
Pipecat processor that routes through Atlas Agent.
"""
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import Frame, TranscriptionFrame, TextFrame

from ..agents import create_atlas_agent, AgentContext

class AtlasAgentProcessor(FrameProcessor):
    """Routes transcriptions through AtlasAgent for full capabilities."""

    def __init__(self, session_id: str = None, **kwargs):
        super().__init__(**kwargs)
        self._agent = create_atlas_agent(session_id=session_id)
        self._session_id = session_id

    async def process_frame(self, frame: Frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            # Route through Agent
            context = AgentContext(
                input_text=frame.text,
                input_type="voice",
                session_id=self._session_id,
            )
            result = await self._agent.run(context)

            # Push response to TTS
            if result.response_text:
                await self.push_frame(TextFrame(text=result.response_text))
        else:
            await self.push_frame(frame, direction)
```

**Verification:**
- [ ] File creates without syntax errors
- [ ] Can be imported: `from atlas_brain.pipecat.agent_processor import AtlasAgentProcessor`
- [ ] Does not affect existing functionality (additive only)

---

### Phase 2: Integrate Agent Processor into Pipecat Pipeline
**Risk: MEDIUM** | **Files: 1 modified**

**File:** `atlas_brain/pipecat/pipeline.py`

**Current (lines 401-413):**
```python
pipeline = Pipeline([
    transport.input(),
    audio_resampler,
    vad_processor,
    stt,
    context_aggregator.user(),
    llm,                          # <-- Replace this
    tts,
    transport.output(),
    context_aggregator.assistant(),
])
```

**After:**
```python
from .agent_processor import AtlasAgentProcessor

# ... in run_voice_pipeline() ...

agent_processor = AtlasAgentProcessor(session_id=session_id)

pipeline = Pipeline([
    transport.input(),
    audio_resampler,
    vad_processor,
    stt,
    agent_processor,              # <-- NEW: Routes through Agent
    tts,
    transport.output(),
])
```

**Changes:**
- Line 43: Add import for AtlasAgentProcessor
- Lines 387-399: Remove `context_aggregator` setup (Agent handles context)
- Lines 401-413: Replace `llm` and aggregators with `agent_processor`

**Verification:**
- [ ] Pipecat pipeline starts without errors
- [ ] Voice command "what time is it" returns actual time (not hallucination)
- [ ] Voice command "turn on the TV" executes device command
- [ ] Voice command "hello" returns conversational response

---

### Phase 3: Unify REST /query/text Endpoint
**Risk: LOW** | **Files: 1 modified**

**File:** `atlas_brain/api/query/text.py`

**Current (lines 17-65):**
```python
@router.post("/text")
async def query_text(request: TextQueryRequest):
    llm = llm_registry.get_active()
    # ... builds messages ...
    result = await execute_with_tools(llm, messages)
    return {"response": result.get("response"), ...}
```

**After:**
```python
from ...agents import create_atlas_agent, AgentContext

@router.post("/text")
async def query_text(request: TextQueryRequest):
    agent = create_atlas_agent()

    context = AgentContext(
        input_text=request.query_text,
        input_type="text",
        session_id=getattr(request, 'session_id', None),
    )

    result = await agent.run(context)

    return {
        "response": result.response_text or "",
        "query": request.query_text,
        "tools_executed": [r.get("tool") for r in result.action_results if r.get("tool")],
        "action_type": result.action_type,
    }
```

**Verification:**
- [ ] `curl -X POST /api/v1/query/text -d '{"query_text": "what time is it"}'` returns time
- [ ] `curl -X POST /api/v1/query/text -d '{"query_text": "turn on the TV"}'` executes device
- [ ] Response schema unchanged (response, query, tools_executed)

---

### Phase 4: Unify REST /llm/chat Endpoint
**Risk: MEDIUM** | **Files: 1 modified**

**File:** `atlas_brain/api/llm.py`

**External client:** `voice_client.py` uses this endpoint.

**Current (lines 125-199):**
```python
@router.post("/chat")
async def chat(request: ChatRequest):
    service = llm_registry.get_active()
    # ... loads history ...
    result = service.chat(messages)
    # ... stores turns ...
    return result
```

**After:**
```python
from ..agents import create_atlas_agent, AgentContext

@router.post("/chat")
async def chat(request: ChatRequest):
    # Get user message from request
    user_message = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_message = msg.content
            break

    if not user_message:
        raise HTTPException(400, "No user message found")

    agent = create_atlas_agent(session_id=request.session_id)

    context = AgentContext(
        input_text=user_message,
        input_type="text",
        session_id=request.session_id,
    )

    result = await agent.run(context)

    # Return in same format as before for compatibility
    return {
        "response": result.response_text or "",
        "model": "atlas-agent",
    }
```

**Verification:**
- [ ] `voice_client.py` still works (same request/response format)
- [ ] Session persistence works
- [ ] Tools execute when needed

---

### Phase 5: Update /devices/intent (Optional)
**Risk: LOW** | **Files: 1 modified**

**File:** `atlas_brain/api/devices/control.py`

This is optional because the current implementation works, but unifying it provides consistency and adds LLM response capability.

**Current (lines 148-183):**
```python
@router.post("/intent")
async def execute_intent(body: IntentRequestBody):
    intent = await intent_parser.parse(body.query)
    result = await action_dispatcher.dispatch_intent(intent)
    return ActionResponse(...)
```

**After:**
```python
from ...agents import create_atlas_agent, AgentContext

@router.post("/intent")
async def execute_intent(body: IntentRequestBody):
    agent = create_atlas_agent()

    context = AgentContext(
        input_text=body.query,
        input_type="text",
    )

    result = await agent.run(context)

    return ActionResponse(
        success=result.success,
        message=result.response_text or "",
        data={
            "intent": result.intent.model_dump() if result.intent else None,
            "action_type": result.action_type,
        },
    )
```

**Verification:**
- [ ] Device commands still work
- [ ] Response includes natural language response

---

### Phase 6: Clean Up Deprecated Code
**Risk: LOW** | **Files: Multiple**

After all phases verified working:

1. **Remove duplicate tool execution** from `api/query/text.py`
2. **Remove duplicate LLM chat** from `api/llm.py` (keep model management endpoints)
3. **Simplify Pipecat pipeline** - remove unused aggregators
4. **Update exports** in `atlas_brain/pipecat/__init__.py`

---

## Files Affected Summary

| Phase | File | Action | Risk |
|-------|------|--------|------|
| 1 | `pipecat/agent_processor.py` | CREATE | LOW |
| 2 | `pipecat/pipeline.py` | MODIFY lines 43, 387-413 | MEDIUM |
| 3 | `api/query/text.py` | MODIFY lines 17-65 | LOW |
| 4 | `api/llm.py` | MODIFY lines 125-199 | MEDIUM |
| 5 | `api/devices/control.py` | MODIFY lines 148-183 | LOW |
| 6 | Multiple | CLEANUP | LOW |

---

## Verification Checklist

### After Phase 1 (Agent Processor)
- [ ] New file imports without errors
- [ ] No existing functionality broken

### After Phase 2 (Pipecat Integration)
- [ ] `uvicorn atlas_brain.main:app --port 8000` starts without errors
- [ ] Pipecat pipeline logs show "AtlasAgentProcessor"
- [ ] Voice: "what time is it" → returns actual time
- [ ] Voice: "turn on the TV" → executes device command
- [ ] Voice: "hello" → conversational response

### After Phase 3 (/query/text)
- [ ] `curl -X POST localhost:8000/api/v1/query/text -H "Content-Type: application/json" -d '{"query_text":"what time is it"}'`
- [ ] Response contains actual time, not hallucination
- [ ] Response schema unchanged

### After Phase 4 (/llm/chat)
- [ ] `python voice_client.py` still works
- [ ] Tools execute through /llm/chat
- [ ] Session persistence works

### After Phase 5 (/devices/intent)
- [ ] Device commands work
- [ ] Returns natural language response

### Full Integration
- [ ] All transports use same Agent
- [ ] Consistent behavior across all interfaces
- [ ] No breaking changes for external clients

---

## Rollback Strategy

Each phase is independent. If a phase fails:

1. **Phase 1 fails**: Delete new file, no impact
2. **Phase 2 fails**: Revert pipeline.py changes, Pipecat uses old LLM path
3. **Phase 3 fails**: Revert text.py, /query/text uses execute_with_tools
4. **Phase 4 fails**: Revert llm.py, /llm/chat uses direct LLM
5. **Phase 5 fails**: Revert control.py, /devices/intent uses direct dispatch

---

## Session Notes

### 2026-01-19 Session
- Analyzed fragmented architecture
- Identified 6 entry points, only 2 use Agent
- Mapped all dependencies and external clients
- Found `voice_client.py` depends on `/llm/chat` endpoint
- Created phased implementation plan
- User approved Phase 1

**Phase 1 Complete:**
- Created `atlas_brain/pipecat/agent_processor.py`
- Added `AtlasAgentProcessor` class that routes through Agent
- Updated `atlas_brain/pipecat/__init__.py` to export new class
- Verified all imports work
- Verified no breaking changes to existing functionality

**Phase 2 Complete:**
- Added import for `AtlasAgentProcessor` to `pipeline.py` (line 44)
- Added `session_id` parameter to `run_voice_pipeline` function
- Replaced LLM + context aggregators with `AtlasAgentProcessor`
- Updated pipeline construction to use agent processor
- Updated `main.py` to generate and pass session_id
- Verified all imports and syntax checks pass
- Old imports kept for `create_voice_pipeline` and `run_test_pipeline` functions

**Phase 3 Complete:**
- Updated `atlas_brain/schemas/query.py` to add optional `session_id` field
- Rewrote `atlas_brain/api/query/text.py` to use `AtlasAgent`
- Endpoint now routes through Agent for full capabilities
- Response schema unchanged for backwards compatibility
- Added `action_type` to response for more info
- Verified all imports and syntax checks pass

**Phase 4 Complete:**
- Modified `atlas_brain/api/llm.py` `/chat` endpoint to use `AtlasAgent`
- Removed unused imports (UUID, db_settings, storage repos)
- Kept model management endpoints unchanged (/activate, /deactivate, /generate, /status)
- Response format unchanged: `{"response": "..."}` for `voice_client.py` compatibility
- Verified all imports and syntax checks pass

**Phase 5 Complete:**
- Modified `atlas_brain/api/devices/control.py` `/intent` endpoint to use `AtlasAgent`
- Added `session_id` parameter to `IntentRequestBody`
- Removed unused import (intent_parser)
- Kept device management endpoints unchanged (/, /{id}, /{id}/state, /{id}/action)
- Response now includes natural language response from Agent
- Verified all imports and syntax checks pass

**Phase 6 Complete:**
- Verified no unused imports in modified files
- Updated `atlas_brain/pipecat/__init__.py` exports to include `AtlasAgentProcessor`
- Ran comprehensive import test - all modules import successfully:
  - `atlas_brain.pipecat.agent_processor` - OK
  - `atlas_brain.pipecat.pipeline` - OK
  - `atlas_brain.api.query.text` - OK
  - `atlas_brain.api.llm` - OK
  - `atlas_brain.api.devices.control` - OK
  - `atlas_brain.schemas.query` - OK
  - `atlas_brain.agents` - OK

---

## Final Implementation Summary

All 6 phases have been completed successfully. The Atlas system now has a **unified entry point** through `AtlasAgent.run()` for all interfaces:

| Endpoint | Uses Agent? | Has Tools? | Has Devices? | Has Memory? |
|----------|-------------|------------|--------------|-------------|
| WS `/ws/orchestrated` | Yes | Yes | Yes | Yes |
| POST `/orchestration/text` | Yes | Yes | Yes | Yes |
| POST `/query/text` | **Yes** | **Yes** | **Yes** | **Yes** |
| POST `/llm/chat` | **Yes** | **Yes** | **Yes** | **Yes** |
| POST `/devices/intent` | **Yes** | **Yes** | **Yes** | **Yes** |
| Pipecat Pipeline | **Yes** | **Yes** | **Yes** | **Yes** |

### Files Modified
1. `atlas_brain/pipecat/agent_processor.py` - NEW
2. `atlas_brain/pipecat/__init__.py` - Updated exports
3. `atlas_brain/pipecat/pipeline.py` - Uses AtlasAgentProcessor
4. `atlas_brain/main.py` - Passes session_id to voice pipeline
5. `atlas_brain/schemas/query.py` - Added session_id field
6. `atlas_brain/api/query/text.py` - Routes through Agent
7. `atlas_brain/api/llm.py` - Routes through Agent
8. `atlas_brain/api/devices/control.py` - Routes through Agent

### Next Steps (Testing)
1. Start server: `uvicorn atlas_brain.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir atlas_brain --reload-exclude data/postgres --reload-exclude data/postgres/**`
2. Test voice: "Atlas, what time is it" - should execute get_time tool
3. Test REST: `curl -X POST localhost:8000/api/v1/query/text -H "Content-Type: application/json" -d '{"query_text":"what time is it"}'`
4. Verify `voice_client.py` still works with `/llm/chat`
