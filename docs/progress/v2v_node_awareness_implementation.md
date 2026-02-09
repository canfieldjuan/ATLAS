# V2V Node Awareness Implementation Plan

## Status: IMPLEMENTED (2026-01-30)

## Verified Insertion Points (Updated after merge to main)

All line numbers verified against current codebase (commit 3c466e1):

| File | Line | Change |
|------|------|--------|
| `config.py` | 638-641 | Add node_id fields to VoiceClientConfig |
| `voice/pipeline.py` | 576 | Add node_id param to __init__ |
| `voice/pipeline.py` | ~590 | Store self.node_id |
| `voice/pipeline.py` | 746, 785, 805 | Add node_id + speaker info to context dict |
| `voice/launcher.py` | 41-49 | Extract context, pass runtime_context + speaker_id to agent.process() |
| `voice/launcher.py` | 232-239 | Same for _run_agent_fallback() |
| `voice/launcher.py` | ~420 | Pass node_id to VoicePipeline |
| `voice/playback.py` | 32-36 | Add target_node param (optional) |

## Codebase Changes Since Original Plan

- Unified agent interface now in use (`agents/interface.py`)
- LangGraph agents migrated
- `agent.process()` already accepts `runtime_context` and `speaker_id` params
- Speaker ID pipeline integration already merged (lines 728, 775, 898)

## Discovery Summary

**Good News:** Edge device WebSocket support already exists at `api/edge/websocket.py` with `location_id` tracking. We replicate this pattern in the local voice pipeline.

## Files to Modify

| File | Changes | Risk |
|------|---------|------|
| `atlas_brain/config.py` | Add node config fields | Low |
| `atlas_brain/voice/pipeline.py` | Add node_id parameter, inject into context | Low |
| `atlas_brain/voice/launcher.py` | Pass node_id from config to pipeline | Low |
| `atlas_brain/voice/playback.py` | Add optional target_node param | Low |

## Implementation Phases

### Phase 1: Configuration (Low Risk)

**File:** `atlas_brain/config.py`

**Location:** After existing VoiceClientConfig class (~line 458)

**Add to existing VoiceClientConfig:**
```python
# In VoiceClientConfig class, add:
node_id: str = Field(
    default="local",
    description="Unique identifier for this voice node"
)
node_name: Optional[str] = Field(
    default="Local Voice",
    description="Human-readable name for this voice node"
)
```

**Environment variables:**
```bash
ATLAS_VOICE_NODE_ID=kitchen
ATLAS_VOICE_NODE_NAME="Kitchen Assistant"
```

**Verification:** Import config, check settings.voice.node_id exists

---

### Phase 2: Pipeline Input (Low Risk)

**File:** `atlas_brain/voice/pipeline.py`

**Step 2a: Add node_id to __init__ (line 539)**

Current:
```python
def __init__(
    self,
    wakeword_model_paths: List[str],
    # ... existing params ...
    unknown_speaker_response: str = "I don't recognize your voice.",
):
```

Add after `unknown_speaker_response`:
```python
    node_id: str = "local",
```

**Step 2b: Store node_id (around line 600)**

After existing instance variables, add:
```python
self.node_id = node_id
```

**Step 2c: Inject into context (lines 746, 785, 805)**

Current (line 746):
```python
context = {"session_id": self.session_id}
```

Change to:
```python
context = self._build_context()
```

**Step 2d: Add helper method (after _verify_speaker method ~line 940)**

Add new method:
```python
def _build_context(self) -> dict:
    """Build context dict with session, node, and speaker info."""
    ctx = {
        "session_id": self.session_id,
        "node_id": self.node_id,
    }
    if self._last_speaker_match:
        ctx["speaker_name"] = self._last_speaker_match.user_name
        if self._last_speaker_match.user_id:
            ctx["speaker_id"] = str(self._last_speaker_match.user_id)
        ctx["speaker_confidence"] = self._last_speaker_match.confidence
    return ctx
```

Apply `context = self._build_context()` at lines 746, 785, and 805.

**Verification:** Add logging to confirm node_id and speaker info in context

---

### Phase 3: Launcher Update (Low Risk)

**File:** `atlas_brain/voice/launcher.py`

**Step 3a: Update _create_agent_runner() (lines 41-49)**

Current:
```python
session_id = context_dict.get("session_id")

try:
    future = asyncio.run_coroutine_threadsafe(
        agent.process(
            input_text=transcript,
            session_id=session_id,
            input_type="voice",
        ),
```

Change to:
```python
session_id = context_dict.get("session_id")
node_id = context_dict.get("node_id")
speaker_id = context_dict.get("speaker_id")
speaker_name = context_dict.get("speaker_name")

try:
    future = asyncio.run_coroutine_threadsafe(
        agent.process(
            input_text=transcript,
            session_id=session_id,
            speaker_id=speaker_name,
            input_type="voice",
            runtime_context={"node_id": node_id, "speaker_uuid": speaker_id},
        ),
```

**Step 3b: Update _run_agent_fallback() (lines 232-239)**

Current:
```python
session_id = context_dict.get("session_id")
try:
    result = await process_with_fallback(
        input_text=transcript,
        agent_type="atlas",
        session_id=session_id,
        input_type="voice",
    )
```

Change to:
```python
session_id = context_dict.get("session_id")
node_id = context_dict.get("node_id")
speaker_id = context_dict.get("speaker_id")
speaker_name = context_dict.get("speaker_name")
try:
    result = await process_with_fallback(
        input_text=transcript,
        agent_type="atlas",
        session_id=session_id,
        speaker_id=speaker_name,
        input_type="voice",
        runtime_context={"node_id": node_id, "speaker_uuid": speaker_id},
    )
```

**Step 3c: VoicePipeline instantiation (~line 420)**

Add after existing params:
```python
    node_id=cfg.node_id,
```

**Also add logging (line ~325):**
```python
logger.info("  node_id=%s", cfg.node_id)
```

**Verification:** Start voice pipeline, check logs for node_id

---

### Phase 4: Output Routing Preparation (Low Risk)

**File:** `atlas_brain/voice/playback.py`

**Location:** speak() method (line 32)

Current:
```python
def speak(
    self,
    text: str,
    on_start: Optional[Callable[[], None]] = None,
    on_done: Optional[Callable[[], None]] = None,
):
```

Change to:
```python
def speak(
    self,
    text: str,
    target_node: Optional[str] = None,
    on_start: Optional[Callable[[], None]] = None,
    on_done: Optional[Callable[[], None]] = None,
):
```

**Note:** For Phase 4, we only add the parameter. Actual routing to remote nodes is Phase 5 (future).

**Verification:** Existing speak() calls still work (target_node defaults to None)

---

### Phase 5: Remote Node Routing (Future - Not This PR)

This phase adds actual remote node communication:
- WebSocket connection manager for edge nodes
- Route speak() to remote nodes via WebSocket
- Broadcast capability

**Deferred** - requires edge device firmware/software first.

---

## Verification Checklist

### After Phase 1:
- [ ] `python -c "from atlas_brain.config import settings; print(settings.voice.node_id)"`
- [ ] Returns "local" (default) or configured value

### After Phase 2:
- [ ] Voice pipeline starts without errors
- [ ] Context dict includes node_id (check logs)

### After Phase 3:
- [ ] Launcher logs show node_id
- [ ] Voice command processing includes node_id in agent context

### After Phase 4:
- [ ] speak() accepts target_node parameter
- [ ] Existing calls work (parameter is optional)

---

## Files NOT Modified

| File | Reason |
|------|--------|
| `agents/protocols.py` | Use existing runtime_context, no schema change needed |
| `agents/interface.py` | Already accepts runtime_context |
| `api/edge/websocket.py` | Already has location_id, no changes needed |
| `frame_processor.py` | Node tracking happens at pipeline level |

---

## Breaking Change Analysis

| Change | Breaking? | Mitigation |
|--------|-----------|------------|
| New config fields | No | Has defaults |
| New __init__ param | No | Has default |
| New speak() param | No | Optional param |
| Context dict changes | No | Additive only |

**Result: No breaking changes expected**

---

## Rollback Plan

If issues occur:
1. Revert config.py changes (remove node_id field)
2. Revert pipeline.py changes (remove node_id param and context injection)
3. Revert launcher.py changes (remove node_id passing)
4. Revert playback.py changes (remove target_node param)

All changes are additive with defaults, so partial rollback is safe.

---

## Success Criteria

1. Voice commands include node_id in agent context
2. Configuration supports node identification
3. No existing functionality broken
4. Foundation ready for future remote node routing

---

## Implementation Log

### 2026-01-30 - COMPLETED

**Phase 1: Configuration**
- Added `node_id` and `node_name` fields to `VoiceClientConfig` in `config.py`
- Environment variables: `ATLAS_VOICE_NODE_ID`, `ATLAS_VOICE_NODE_NAME`
- Defaults: `node_id="local"`, `node_name=None`

**Phase 2: Pipeline Input**
- Added `node_id` parameter to `VoicePipeline.__init__()`
- Stored as `self.node_id`
- Added `_build_context()` helper method returning node_id + speaker info
- Updated all 3 context assignments (lines 748, 787, 807) to use `_build_context()`

**Phase 3: Launcher Update**
- Added `node_id=cfg.node_id` to VoicePipeline instantiation
- Added logging: `node_id=%s, node_name=%s`
- Updated `_create_agent_runner()` to pass `runtime_context` and `speaker_id`
- Updated `_run_agent_fallback()` to pass `runtime_context` and `speaker_id`

**Phase 4: Output Routing Preparation**
- Added `target_node` parameter to `PlaybackController.speak()`
- Parameter is optional, defaults to None (local playback)
- Ready for future remote node routing

**Gap Fix (discovered during review):**
- `_stream_llm_response` was not extracting `speaker_name` from context
- `_persist_streaming_turns` was not accepting/passing `speaker_name`
- Fixed: Added `speaker_name` parameter to `_persist_streaming_turns`
- Fixed: Extract and pass `speaker_name` in streaming LLM path

**Verification:**
- All imports work correctly
- `_build_context()` returns correct node_id and speaker info
- No breaking changes (all params have defaults)
- No Unicode characters in code
- End-to-end flow tested with mock speaker match

---

*Document created: 2026-01-30*
*Status: IMPLEMENTED*
