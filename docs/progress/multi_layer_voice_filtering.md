# Multi-Layer Voice Filtering Implementation

## Status: Complete

**Date:** 2026-01-31

## Problem Statement

Conversation mode was picking up ambient family conversations, causing false triggers when people nearby were talking but not addressing Atlas directly.

## Solution: 5-Layer Filtering Stack

Implemented a multi-layer filtering approach to ensure only directed speech to Atlas is processed during conversation mode:

```
Layer 1: Silero VAD (prob > 0.7) - More accurate speech detection
   ↓
Layer 2: RMS Energy (> 0.008) - Proximity/loudness check
   ↓
Layer 3: Speaker Continuity (optional) - Same speaker as wake word
   ↓
Layer 4: Intent Gating (conf > 0.6) - Gate conversation continuation
   ↓
Layer 5: Turn Limit (max 3) - Require wake word after N turns
```

## Files Modified

| File | Changes |
|------|---------|
| `atlas_brain/voice/vad/__init__.py` | **NEW** - Package init |
| `atlas_brain/voice/vad/silero.py` | **NEW** - Silero VAD wrapper |
| `atlas_brain/config.py` | Added `VoiceFilterConfig` class |
| `atlas_brain/voice/pipeline.py` | VAD selection, voice filter params |
| `atlas_brain/voice/frame_processor.py` | RMS filter, turn tracking, speaker continuity |
| `atlas_brain/voice/launcher.py` | Silero preload, intent gating |

## Configuration

All settings are configurable via environment variables with the `ATLAS_VOICE_FILTER_` prefix:

```bash
# Master enable
ATLAS_VOICE_FILTER_ENABLED=true

# Layer 1: VAD backend selection
ATLAS_VOICE_FILTER_VAD_BACKEND=webrtc  # or "silero" for more accuracy
ATLAS_VOICE_FILTER_SILERO_THRESHOLD=0.7

# Layer 2: RMS energy filtering
ATLAS_VOICE_FILTER_RMS_MIN_THRESHOLD=0.008
ATLAS_VOICE_FILTER_RMS_ADAPTIVE=false
ATLAS_VOICE_FILTER_RMS_ABOVE_AMBIENT_FACTOR=3.0

# Layer 3: Speaker continuity (disabled by default)
ATLAS_VOICE_FILTER_SPEAKER_CONTINUITY_ENABLED=false
ATLAS_VOICE_FILTER_SPEAKER_CONTINUITY_THRESHOLD=0.7

# Layer 4: Intent gating
ATLAS_VOICE_FILTER_INTENT_GATING_ENABLED=true
ATLAS_VOICE_FILTER_INTENT_CONTINUATION_THRESHOLD=0.6
# Categories that allow conversation continuation (JSON array)
# ATLAS_VOICE_FILTER_INTENT_CATEGORIES_CONTINUE=["conversation", "tool_use", "device_control"]

# Layer 5: Turn limiting
ATLAS_VOICE_FILTER_TURN_LIMIT_ENABLED=true
ATLAS_VOICE_FILTER_MAX_CONVERSATION_TURNS=3
```

## Layer Details

### Layer 1: Silero VAD

Silero VAD uses a neural network model to detect speech more accurately than WebRTC VAD. It's particularly good at:
- Rejecting TV/radio speech
- Handling varying noise levels
- Reducing false positives from music/ambient sounds

**Trade-off:** Slightly higher latency (~5ms per frame) but much better accuracy.

**Usage:**
```bash
ATLAS_VOICE_FILTER_VAD_BACKEND=silero
```

The model is automatically downloaded from torch hub on first use (~1MB ONNX file).

### Layer 2: RMS Energy Filter

Filters audio based on volume/energy level. Speech directed at a nearby microphone will have higher RMS than background conversations.

- `rms_min_threshold`: Absolute minimum RMS to consider as speech
- `rms_adaptive`: If enabled, tracks ambient noise floor and requires speech to be N times above it
- `rms_above_ambient_factor`: Multiplier for adaptive threshold (e.g., 3.0x ambient)

### Layer 3: Speaker Continuity (Optional)

When enabled, stores the speaker embedding from the wake word audio and only accepts follow-up speech from the same speaker.

**Note:** Disabled by default. Requires speaker ID service to be enabled.

### Layer 4: Intent Gating

After processing each command, checks the intent classification result:
- If confidence is below `intent_continuation_threshold`, exits conversation mode
- If intent category is not in `intent_categories_continue`, exits conversation mode

This prevents continued conversation when Atlas detects irrelevant speech (e.g., someone asking another person a question).

### Layer 5: Turn Limit

Requires re-engagement with wake word after a maximum number of conversation turns. This:
- Prevents runaway conversations
- Forces explicit re-engagement after several exchanges
- Default is 3 turns before requiring wake word again

## Dependencies

For Silero VAD:
- `onnxruntime` - For ONNX model inference
- `torch` - For model download (only needed on first run)

```bash
pip install onnxruntime torch
```

## Testing

### Per-Layer Testing

1. **Silero VAD**: Compare WebRTC vs Silero on same audio
   ```bash
   ATLAS_VOICE_FILTER_VAD_BACKEND=silero python -m atlas_brain.voice.launcher
   ```

2. **Turn Limit**: Count turns in logs, verify wake word required after limit
   ```bash
   ATLAS_VOICE_FILTER_MAX_CONVERSATION_TURNS=2
   ```

3. **Intent Gating**: Watch logs for "Intent gating: exiting conversation" messages

4. **RMS Filter**: Enable debug logging to see RMS values
   ```bash
   ATLAS_VOICE_DEBUG_LOGGING=true
   ```

### Integration Testing

Test scenarios:
- TV/radio playing → should not trigger
- Multiple people talking → only wake word speaker should be processed (if speaker continuity enabled)
- Natural 5+ turn conversation → should require re-engagement after 3 turns

### Rollback Plan

Each layer can be disabled independently:
```bash
ATLAS_VOICE_FILTER_VAD_BACKEND=webrtc      # Disable Silero
ATLAS_VOICE_FILTER_INTENT_GATING_ENABLED=false
ATLAS_VOICE_FILTER_TURN_LIMIT_ENABLED=false
ATLAS_VOICE_FILTER_SPEAKER_CONTINUITY_ENABLED=false
```

To fully disable the new filtering:
```bash
ATLAS_VOICE_FILTER_ENABLED=false
```

## Future Improvements

1. **Adaptive Silero threshold**: Adjust threshold based on ambient conditions
2. **Speaker diarization**: Track multiple speakers and their conversation context
3. **Audio fingerprinting**: Detect and filter known audio sources (TV shows, music)
4. **Conversation context**: Use NLP to detect if speech is directed at Atlas vs others
