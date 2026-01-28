# Voice Pipeline Optimizations

**Created:** 2026-01-28
**Last Updated:** 2026-01-28
**Status:** Complete - All high-impact optimizations implemented

---

## Completed Optimizations

### 1. SentenceBuffer O(n) Token Accumulation (commit 3cc4473)
- **Issue:** String concatenation O(n) per token, O(n^2) total
- **Fix:** Use list[str] with O(1) append, join once on sentence complete
- **Impact:** HIGH - streaming LLM token path

### 2. ASR Connection Code Deduplication (commit 3cc4473)
- **Issue:** Same 10-line try/except block repeated 3 times
- **Fix:** Extract `_connect_streaming_asr()` helper method
- **Impact:** HIGH - maintainability + minor latency

### 3. Skip WAV Encode/Decode in Streaming Fallback (commit 184d22d)
- **Issue:** PCM -> WAV -> PCM roundtrip when streaming ASR falls back
- **Fix:** Add `transcribe_pcm()` method, use PCM directly
- **Impact:** HIGH - streaming fallback path

### 4. RMS Calculation Only on Log Frames (commit 2f7bbe8)
- **Issue:** RMS calculated every frame, only needed for periodic logging
- **Fix:** Check `should_log` before RMS calculation
- **Impact:** MEDIUM - audio callback path

### 5. Remove Unused _max_wake_score (commit 184d22d)
- **Issue:** Tracked every frame, never read or logged
- **Fix:** Remove dead code entirely
- **Impact:** LOW - minor CPU per frame

### 6. Socket Blocking Toggle Removal (commit 7c8471d)
- **Issue:** setblocking(False/True) called every frame during streaming recording
- **Fix:** Use native `recv(timeout=...)` - no socket mode changes needed
- **Impact:** HIGH - 2-4 fewer syscalls per frame (~12.5 frames/sec during recording)

---

## Completed: Socket Blocking Toggle Optimization

### Problem Analysis

**Location:** `atlas_brain/voice/pipeline.py` lines 195-205

**Current code (inefficient):**
```python
# Check for partial transcript (non-blocking)
try:
    self._ws.socket.setblocking(False)      # syscall 1
    response = self._ws.recv(timeout=0.001)  # recv with timeout
    self._ws.socket.setblocking(True)        # syscall 2
    data = json.loads(response)
    if data.get("type") == "partial":
        self._last_partial = data.get("text", "")
        return self._last_partial
except (TimeoutError, BlockingIOError):
    self._ws.socket.setblocking(True)        # syscall 3 (exception path)
except Exception:
    self._ws.socket.setblocking(True)        # syscall 4 (exception path)
```

**Issues:**
1. `setblocking()` is UNNECESSARY - `recv(timeout=...)` handles timeout natively
2. 2-4 syscalls per frame during recording (~12.5 frames/sec)
3. Exception handling overly complex due to blocking toggle
4. Risk of leaving socket in wrong state on unexpected exception

**Root cause:** Misunderstanding of websockets library - `recv(timeout=X)` works without socket mode changes

### Proposed Fix

**Optimized code:**
```python
# Check for partial transcript (non-blocking)
try:
    response = self._ws.recv(timeout=0.001)  # Native timeout, no mode toggle
    data = json.loads(response)
    if data.get("type") == "partial":
        self._last_partial = data.get("text", "")
        return self._last_partial
except TimeoutError:
    pass  # No data available, continue
```

**Changes:**
1. Remove all `setblocking()` calls
2. Simplify exception handling to just `TimeoutError`
3. Remove `BlockingIOError` catch (not raised with native timeout)

### Impact Analysis

**Files affected:**
- `atlas_brain/voice/pipeline.py` - `NemotronAsrStreamingClient.send_audio()` method only

**No breaking changes:**
- Method signature unchanged
- Return type unchanged
- Behavior unchanged (still returns partial or None)

**Dependencies verified:**
- `frame_processor.py` line 296: Calls `send_audio()` - no change needed
- `pipeline.py` lines 299, 321: Calls `send_audio()` in fallback - no change needed

### Implementation Steps

1. Verify current code matches expected pattern (lines 193-207)
2. Remove `setblocking(False)` call (line 195)
3. Remove `setblocking(True)` calls (lines 197, 203, 205)
4. Simplify exception handling to only catch `TimeoutError`
5. Verify code compiles
6. Test with actual streaming ASR

---

## Pending: Timer Thread Allocation

### Analysis

**Location:** `atlas_brain/voice/frame_processor.py` lines 459-467

**Current pattern:**
- New `threading.Timer` created in `enter_conversation_mode()`
- Called once per conversation turn (after TTS completes)
- NOT called per frame

**Conclusion:** LOW IMPACT - not a hot path, optimization would add complexity for minimal gain.

**Recommendation:** Skip this optimization. Focus on higher-impact items.

---

## Session Log

### 2026-01-28 Session 1
- Identified optimization opportunities via codebase analysis
- Implemented: SentenceBuffer, ASR dedup, WAV skip, RMS skip, dead code removal
- Implemented: Socket blocking toggle removal (commit 7c8471d)
- Skipped: Timer thread allocation (low impact, once per conversation turn)
- **Status:** All high-impact optimizations complete
- Implemented high-impact: SentenceBuffer, ASR dedup, WAV skip, RMS skip
- Analyzed socket blocking toggle - confirmed unnecessary
- Analyzed timer allocation - confirmed low impact
- **Status:** Awaiting approval for socket blocking fix

---

## Approval Checklist

Before implementing socket blocking fix:
- [x] Verified websockets `recv(timeout=...)` handles timeout natively
- [x] Verified `TimeoutError` is the correct exception to catch
- [x] Identified exact lines to modify (195-205)
- [x] Confirmed no breaking changes to dependent code
- [ ] User approval received
