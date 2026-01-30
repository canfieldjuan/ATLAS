# Streaming LLM to TTS Implementation Plan

**Created:** 2026-01-28
**Last Updated:** 2026-01-28
**Status:** Planning - Awaiting Approval

---

## Overview

Stream LLM tokens to TTS as sentences complete, reducing time-to-first-audio by speaking the first sentence while the LLM continues generating.

### Current Flow (Blocking)
```
transcript -> agent.run() -> LLM (full response) -> TTS (full audio)
                            [1-2s blocking]        [then speak all]
```

### Target Flow (Streaming)
```
transcript -> agent.run_stream() -> LLM tokens -> SentenceBuffer -> TTS per sentence
                                        |              |
                                   "Hello."    -> speak immediately
                                   "How are"   -> buffer
                                   " you?"     -> speak "How are you?"
```

### Expected Improvement
- First sentence speaks 500-1500ms earlier
- More natural conversation rhythm

---

## Phase 1: Add chat_stream_async to OllamaLLM

### File: `atlas_brain/services/llm/ollama.py`

**Insertion Point:** After line 314 (after `chat_async` method ends)

**Changes:**
1. Add `AsyncIterator` to imports (line 10)
2. Add `chat_stream_async` method after `chat_async`

**Current code at line 10:**
```python
from typing import Any
```

**New imports:**
```python
from typing import Any, AsyncIterator
```

**New method (insert after line 314):**
```python
async def chat_stream_async(
    self,
    messages: list[Message],
    max_tokens: int = 256,
    temperature: float = 0.7,
    **kwargs: Any,
) -> AsyncIterator[str]:
    """
    Async streaming chat completion - yields tokens as generated.

    Args:
        messages: List of Message objects
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Yields:
        Token strings as they are generated
    """
    if not self._client:
        raise RuntimeError("Ollama LLM not loaded")

    ollama_messages = []
    for msg in messages:
        ollama_messages.append({
            "role": msg.role,
            "content": msg.content,
        })

    payload = {
        "model": self.model,
        "messages": ollama_messages,
        "stream": True,
        "keep_alive": "30m",
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
        },
    }

    try:
        async with self._client.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                import json
                data = json.loads(line)
                content = data.get("message", {}).get("content", "")
                if content:
                    yield content
                if data.get("done", False):
                    break
    except httpx.HTTPError as e:
        logger.error("Ollama streaming chat error: %s", e)
        raise
```

---

## Phase 2: Add SentenceBuffer to voice pipeline

### File: `atlas_brain/voice/pipeline.py`

**Insertion Point:** After line 313 (after `NemotronAsrStreamingClient` class, before `PiperTTS` class)

**New class:**
```python
class SentenceBuffer:
    """Buffer that accumulates tokens and yields complete sentences."""

    # Sentence-ending punctuation
    SENTENCE_ENDINGS = ".!?"

    def __init__(self):
        self._buffer = ""

    def add_token(self, token: str) -> Optional[str]:
        """
        Add a token to buffer. Returns sentence if complete.

        Args:
            token: Token string from LLM

        Returns:
            Complete sentence if buffer ends with sentence punctuation, else None
        """
        self._buffer += token

        # Check if buffer ends with sentence punctuation
        stripped = self._buffer.rstrip()
        if stripped and stripped[-1] in self.SENTENCE_ENDINGS:
            sentence = stripped
            self._buffer = ""
            return sentence

        return None

    def flush(self) -> Optional[str]:
        """Flush remaining buffer content."""
        if self._buffer.strip():
            content = self._buffer.strip()
            self._buffer = ""
            return content
        return None

    def clear(self):
        """Clear the buffer."""
        self._buffer = ""
```

---

## Phase 3: Add streaming agent runner to launcher

### File: `atlas_brain/voice/launcher.py`

**Insertion Point:** After line 59 (after `_create_agent_runner` function)

**New function:**
```python
def _create_streaming_agent_runner():
    """Create a streaming agent runner that yields sentences."""
    from ..services import llm_registry
    from ..services.protocols import Message

    def runner(transcript: str, context_dict: dict, on_sentence: Callable[[str], None]) -> None:
        """
        Run agent with streaming LLM, calling on_sentence for each complete sentence.

        Args:
            transcript: User input text
            context_dict: Session context
            on_sentence: Callback for each sentence
        """
        if _event_loop is None:
            logger.error("No event loop for streaming agent")
            return

        llm = llm_registry.get_active()
        if llm is None or not hasattr(llm, "chat_stream_async"):
            logger.warning("LLM does not support streaming, falling back")
            # Fallback to non-streaming
            from ..agents.atlas import get_atlas_agent
            agent = get_atlas_agent()
            ctx = AgentContext(
                input_text=transcript,
                session_id=context_dict.get("session_id"),
            )
            future = asyncio.run_coroutine_threadsafe(agent.run(ctx), _event_loop)
            result = future.result(timeout=30.0)
            if result.response_text:
                on_sentence(result.response_text)
            return

        # Stream from LLM
        async def stream_response():
            from .pipeline import SentenceBuffer
            buffer = SentenceBuffer()

            messages = [
                Message(role="system", content=_PREFILL_SYSTEM_PROMPT),
                Message(role="user", content=transcript),
            ]

            try:
                async for token in llm.chat_stream_async(messages, max_tokens=150):
                    sentence = buffer.add_token(token)
                    if sentence:
                        on_sentence(sentence)

                # Flush remaining content
                remaining = buffer.flush()
                if remaining:
                    on_sentence(remaining)
            except Exception as e:
                logger.error("Streaming agent error: %s", e)

        future = asyncio.run_coroutine_threadsafe(stream_response(), _event_loop)
        future.result(timeout=30.0)

    return runner
```

**Also add import at top (around line 12):**
```python
from typing import Any, Callable, Dict, Optional
```

---

## Phase 4: Add streaming command handler to VoicePipeline

### File: `atlas_brain/voice/pipeline.py`

**Changes to `VoicePipeline.__init__`:**
- Add `streaming_agent_runner` parameter (after `agent_runner`)

**Insertion Point for new method:** After `_handle_streaming_command` (line 694)

**New method:**
```python
def _handle_streaming_llm_command(self, transcript: str):
    """Handle command with streaming LLM to TTS."""
    if not transcript:
        logger.warning("Empty transcript for streaming LLM")
        return

    logger.info("Streaming LLM command: %s", transcript)
    context = {"session_id": self.session_id}

    first_sentence = True

    def on_sentence(sentence: str):
        nonlocal first_sentence
        logger.info("Streaming sentence: %s", sentence[:80] if len(sentence) > 80 else sentence)

        if first_sentence:
            self.playback.speak(
                sentence,
                on_start=self._on_playback_start,
                on_done=None,  # Don't trigger conversation mode yet
            )
            first_sentence = False
        else:
            # Queue subsequent sentences
            self.playback.speak(
                sentence,
                on_start=None,
                on_done=None,
            )

    if self.streaming_agent_runner:
        self.streaming_agent_runner(transcript, context, on_sentence)
    else:
        # Fallback to non-streaming
        reply = self.agent_runner(transcript, context)
        if reply:
            on_sentence(reply)

    # Final sentence triggers conversation mode
    self._on_playback_done()
```

---

## Phase 5: Wire streaming through launcher

### File: `atlas_brain/voice/launcher.py`

**Changes to `create_voice_pipeline` function:**

After `agent_runner = _create_agent_runner()` (line 186), add:
```python
streaming_agent_runner = _create_streaming_agent_runner()
```

Update `VoicePipeline(...)` call to include:
```python
streaming_agent_runner=streaming_agent_runner,
```

---

## Phase 6: Config flag for streaming LLM

### File: `atlas_brain/config.py`

**Insertion Point:** After `piper_sample_rate` field (around line 571)

```python
streaming_llm_enabled: bool = Field(
    default=False,
    description="Enable streaming LLM to TTS (speak sentences as generated)"
)
```

---

## Implementation Order

| Phase | File | Change | Risk |
|-------|------|--------|------|
| 1 | ollama.py | Add `chat_stream_async` | Low - new method |
| 2 | pipeline.py | Add `SentenceBuffer` class | Low - new class |
| 3 | launcher.py | Add `_create_streaming_agent_runner` | Low - new function |
| 4 | pipeline.py | Add `_handle_streaming_llm_command` | Medium - new handler |
| 5 | launcher.py | Wire streaming runner | Low - param passing |
| 6 | config.py | Add config flag | Low - new field |

---

## Testing Strategy

1. **Unit test SentenceBuffer** - verify sentence detection
2. **Test OllamaLLM.chat_stream_async** - verify token streaming
3. **Integration test** - verify end-to-end streaming

---

## Rollback Plan

- Config flag `streaming_llm_enabled=False` (default) keeps old behavior
- All existing methods unchanged
- New streaming path is additive

---

## Session Log

### 2026-01-28 Session 1
- Researched streaming LLM to TTS approaches
- Analyzed current codebase architecture
- Identified exact insertion points for all files
- Created phased implementation plan
- **Status:** Approved

### 2026-01-28 Session 2 - Implementation Complete
- Phase 1: Added `chat_stream_async` to OllamaLLM (ollama.py:315-371)
- Phase 2: Added `SentenceBuffer` class to pipeline.py (lines 315-354)
- Phase 3: Added `_create_streaming_agent_runner` to launcher.py (lines 63-127)
- Phase 4: Added `_handle_streaming_llm_command` to VoicePipeline (lines 749-785)
- Phase 5: Added `streaming_llm_enabled` config flag (config.py:572-575)
- Phase 6: Wired streaming through launcher.py
- All components compile successfully
- **Status:** Ready for testing

### 2026-01-28 Session 3 - Bug Fixes
**Issue 1: PlaybackController stopping previous utterances**
- Each `speak()` call was stopping the previous one
- Fixed by concatenating sentences before speaking
- Proper `on_done` callback now fires after TTS completes

**Issue 2: Streaming bypassed intent router/tools**
- "What time is it?" was going to LLM (slow) instead of tool (fast)
- Fixed by adding intent check in streaming runner:
  - `route_query()` classifies intent first
  - "conversation" → streaming LLM
  - "tool_use"/"device_command" → regular agent (fast tools)
- **Status:** Testing

---

## Next Steps (After Approval)
1. Implement Phase 1: OllamaLLM.chat_stream_async
2. Test streaming with simple script
3. Implement Phase 2: SentenceBuffer
4. Continue through phases
5. Full integration test
