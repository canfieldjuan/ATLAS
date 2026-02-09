# Multi-Turn Conversation Mode - Implementation Plan

**Created:** 2026-01-28
**Last Updated:** 2026-01-28
**Status:** Planning - Awaiting Approval

---

## Overview

Implement multi-turn conversation mode to allow natural follow-up interactions without requiring the wake word for every utterance. After Atlas responds, the system stays in a "conversation" state for a configurable timeout, accepting speech directly.

### Goals

- Eliminate wake word requirement for follow-up utterances within conversation window
- Maintain natural conversation flow for a distributed assistant
- Preserve wake word as conversation initiator and re-engagement mechanism
- Ensure clean state transitions and timeout handling
- No breaking changes to existing functionality

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| State location | FrameProcessor | Already manages listening/recording states |
| Timeout mechanism | Timer reset on each interaction | Natural conversation flow |
| Default timeout | 8 seconds (configurable) | Long enough for thinking, short enough to save resources |
| Exit conditions | Timeout, explicit "goodbye", or new wake word | Clear boundaries |
| VAD-gated | Yes | Only record when speech detected, not ambient noise |

---

## Current Architecture Analysis

### Files Examined

1. `atlas_brain/voice/frame_processor.py` - State machine (listening/recording)
2. `atlas_brain/voice/pipeline.py` - VoicePipeline, handles callbacks
3. `atlas_brain/voice/playback.py` - TTS playback with on_done callback
4. `atlas_brain/voice/segmenter.py` - Audio segmentation logic
5. `atlas_brain/voice/launcher.py` - Pipeline creation from config
6. `atlas_brain/config.py` - VoiceClientConfig settings

### Current State Machine (frame_processor.py)

```
States: "listening" | "recording"

Flow:
  listening + wake_detected → recording
  recording + finalize → listening (reset)
```

### Proposed State Machine

```
States: "listening" | "recording" | "conversing"

Flow:
  listening + wake_detected → recording
  recording + finalize → conversing (if conversation_mode enabled)
  conversing + speech_detected → recording  (no wake word needed)
  conversing + timeout → listening
  conversing + goodbye_detected → listening
  conversing + wake_detected → recording (re-engagement)
  recording + finalize → conversing (continue conversation)
```

---

## Files To Modify

### 1. `atlas_brain/config.py`

**Purpose:** Add conversation mode configuration options
**Insertion Point:** After line 599 (`log_interval_frames` field in VoiceClientConfig)

**New Fields:**

```python
    # Conversation mode settings
    conversation_mode_enabled: bool = Field(
        default=False,  # Disabled by default for safety
        description="Enable multi-turn conversation mode (no wake word for follow-ups)"
    )
    conversation_timeout_ms: int = Field(
        default=8000,
        description="Timeout in ms to stay in conversation mode after TTS completes"
    )
    conversation_goodbye_phrases: list[str] = Field(
        default=["goodbye", "bye", "that's all", "thanks that's it", "nevermind"],
        description="Phrases that explicitly end conversation mode"
    )
```

### 2. `atlas_brain/voice/frame_processor.py`

**Purpose:** Add "conversing" state and timeout logic
**Key Changes:**

1. **Add new `__init__` parameters (after line 43, `log_interval_frames`):**
   - `conversation_mode_enabled: bool = False`
   - `conversation_timeout_ms: int = 8000`
   - `on_conversation_timeout: Optional[Callable[[], None]] = None`

2. **Add new instance variables (after line 68, `self._last_partial`):**
   - `self.conversation_mode_enabled = conversation_mode_enabled`
   - `self.conversation_timeout_ms = conversation_timeout_ms`
   - `self.on_conversation_timeout = on_conversation_timeout`
   - `self._conversation_timer: Optional[threading.Timer] = None`

3. **Add new state "conversing" in `process_frame()` (~line 180):**
   - After recording finalize, transition to "conversing" instead of "listening"
   - In "conversing" state, VAD speech triggers recording without wake word
   - Start/reset conversation timer on state entry and after TTS completes

4. **Add conversation timeout handler methods (new methods at end of class):**
   - `_start_conversation_timer()` - Starts/resets timeout timer
   - `_cancel_conversation_timer()` - Cancels active timer
   - `_on_conversation_timeout()` - Transitions back to "listening"
   - `enter_conversation_mode()` - Called by pipeline after TTS completes

5. **Add goodbye detection in finalize (optional, Phase 3)**

### 3. `atlas_brain/voice/pipeline.py`

**Purpose:** Wire conversation mode to FrameProcessor
**Key Changes:**

1. **VoicePipeline.**init** parameters (add after line 429, `log_interval_frames`):**
   - `conversation_mode_enabled: bool = False`
   - `conversation_timeout_ms: int = 8000`

2. **FrameProcessor instantiation (line 482-500):**
   - Pass `conversation_mode_enabled=conversation_mode_enabled`
   - Pass `conversation_timeout_ms=conversation_timeout_ms`
   - Pass `on_conversation_timeout=self._on_conversation_timeout`

3. **VoicePipeline._on_playback_done (line 608-615):**
   - Add call to `self.frame_processor.enter_conversation_mode()` after wake reset

4. **Add new method `_on_conversation_timeout()` (after `_on_playback_done`):**
   - Log conversation end
   - Optionally play acknowledgment sound

### 4. `atlas_brain/voice/launcher.py`

**Purpose:** Pass config to pipeline
**Key Changes:**

1. **create_voice_pipeline VoicePipeline call (line 188-215):**
   - Add `conversation_mode_enabled=cfg.conversation_mode_enabled`
   - Add `conversation_timeout_ms=cfg.conversation_timeout_ms`

---

## Implementation Phases

### Phase 1: Configuration & State Foundation (Low Risk)

**Scope:** Add config fields and state variable scaffolding
**Files:** config.py, frame_processor.py (minimal)
**Risk:** None - additive only, defaults preserve current behavior

**Tasks:**

1. Add conversation config fields to VoiceClientConfig
2. Add instance variables to FrameProcessor (no behavior change)
3. Add `__init__` parameters with defaults
4. Verify existing tests still pass

### Phase 2: State Machine Enhancement (Medium Risk)

**Scope:** Implement "conversing" state and transitions
**Files:** frame_processor.py
**Risk:** Medium - core state machine changes

**Tasks:**

1. Add "conversing" state handling in `process_frame()`
2. Implement `_start_conversation_timer()` method
3. Implement `_on_conversation_timeout()` method
4. Implement `enter_conversation_mode()` method
5. Modify recording finalize to transition to "conversing"
6. Add VAD-triggered recording in "conversing" state
7. Unit test state transitions

### Phase 3: Pipeline Integration (Medium Risk)

**Scope:** Connect FrameProcessor to VoicePipeline
**Files:** pipeline.py, launcher.py
**Risk:** Medium - integration points

**Tasks:**

1. Update VoicePipeline.**init** to pass conversation params
2. Modify `_on_playback_done()` to enter conversation mode
3. Add conversation timeout callback
4. Update launcher to read config
5. Integration test full flow

### Phase 4: Polish & Edge Cases (Low Risk)

**Scope:** Handle edge cases and add goodbye detection
**Files:** frame_processor.py
**Risk:** Low - additive features

**Tasks:**

1. Add goodbye phrase detection (optional)
2. Handle wake word during conversation (re-engagement)
3. Handle TTS interruption during conversation
4. Add metrics/logging for conversation sessions
5. End-to-end testing

---

## Affected Files Summary

| File | Change Type | Risk Level |
|------|-------------|------------|
| `atlas_brain/config.py` | Add fields | Low |
| `atlas_brain/voice/frame_processor.py` | State machine | Medium |
| `atlas_brain/voice/pipeline.py` | Integration | Medium |
| `atlas_brain/voice/launcher.py` | Config wiring | Low |

### Files NOT Modified (Verified No Impact)

- `atlas_brain/voice/segmenter.py` - No changes needed
- `atlas_brain/voice/playback.py` - No changes needed  
- `atlas_brain/voice/audio_capture.py` - No changes needed
- `atlas_brain/voice/command_executor.py` - No changes needed

### Existing Tests (Verified)

- `tests/test_voice_pipeline_omni.py` - Tests high-level omni pipeline, no FrameProcessor mocking
- `tests/test_voice_omni.py` - Tests omni service, no FrameProcessor dependency
- No existing unit tests for FrameProcessor (new tests needed in Phase 2)

---

## Rollback Plan

All changes are additive with feature flag (`conversation_mode_enabled=False` by default in Phase 1).

1. If issues in Phase 2: Set `conversation_mode_enabled=False` in .env
2. If critical bug: Revert commits for affected phase
3. State machine reverts cleanly to listening/recording only

---

## Testing Strategy

### Unit Tests (Phase 2)

- Test state transitions: listening → recording → conversing → listening
- Test timeout fires correctly
- Test VAD triggers recording in conversing state
- Test wake word works in all states

### Integration Tests (Phase 3)

- Test full conversation flow with real audio
- Test timeout returns to listening
- Test goodbye phrase detection
- Test interruption handling

### Manual Testing (Phase 4)

- Natural conversation with follow-ups
- Verify timeout behavior
- Verify goodbye detection
- Verify barge-in during conversation

---

## Success Criteria

1. ✅ Wake word triggers initial interaction
2. ✅ After response, user can speak follow-up without wake word
3. ✅ Conversation times out after configured period
4. ✅ Wake word can re-engage during conversation
5. ✅ Existing wake-word-only flow still works when disabled
6. ✅ No breaking changes to existing tests
7. ✅ Clean state transitions logged

---

## Appendix: Code Snippets

### A. New FrameProcessor States (frame_processor.py)

```python
# In process_frame(), after recording finalize block (~line 260):

# Transition to conversation mode if enabled
if self.conversation_mode_enabled:
    self.state = "conversing"
    self._start_conversation_timer()
    logger.info("State -> conversing (conversation mode active)")
else:
    self.state = "listening"
    logger.info("State -> listening (conversation mode disabled)")

# New state handling for "conversing" (~line 185):
if self.state == "conversing":
    # Check for speech to start new recording (no wake word needed)
    is_speech = self._is_speech(frame_bytes)
    if is_speech:
        self._cancel_conversation_timer()
        self.state = "recording"
        self.segmenter.reset()
        logger.info("SPEECH DETECTED in conversation mode, recording...")
        # Connect streaming ASR if available
        if self.streaming_asr_client is not None:
            try:
                if self.streaming_asr_client.connect():
                    self._streaming_active = True
            except Exception as e:
                logger.warning("Error connecting streaming ASR: %s", e)
        return
    
    # Also allow wake word to re-engage
    if detected:
        self._cancel_conversation_timer()
        self.state = "recording"
        self.segmenter.reset()
        logger.info("WAKE WORD re-engaged during conversation")
        return
```

### B. Timer Management (frame_processor.py)

```python
def _start_conversation_timer(self):
    """Start or reset the conversation timeout timer."""
    self._cancel_conversation_timer()
    self._conversation_timer = threading.Timer(
        self.conversation_timeout_ms / 1000.0,
        self._on_conversation_timeout
    )
    self._conversation_timer.daemon = True
    self._conversation_timer.start()
    logger.debug("Conversation timer started: %dms", self.conversation_timeout_ms)

def _cancel_conversation_timer(self):
    """Cancel any active conversation timer."""
    if self._conversation_timer is not None:
        self._conversation_timer.cancel()
        self._conversation_timer = None

def _on_conversation_timeout(self):
    """Handle conversation timeout - return to listening state."""
    logger.info("Conversation timeout, returning to listening state")
    self._conversation_timer = None
    self.state = "listening"
    if self.on_conversation_timeout is not None:
        self.on_conversation_timeout()

def enter_conversation_mode(self):
    """Enter conversation mode after TTS completes. Called by pipeline."""
    if not self.conversation_mode_enabled:
        return
    if self.state != "listening":
        logger.warning("enter_conversation_mode called in state=%s, ignoring", self.state)
        return
    self.state = "conversing"
    self._start_conversation_timer()
    logger.info("Entered conversation mode (timeout=%dms)", self.conversation_timeout_ms)
```

---

## Status

**Awaiting Approval** - Please review the plan and confirm before implementation begins.
