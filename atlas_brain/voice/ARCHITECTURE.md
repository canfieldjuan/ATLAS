# Voice-to-Voice (V2V) Pipeline Architecture

## Overview

The V2V pipeline is a **bidirectional audio gateway**. It handles audio I/O only - no business logic, no tool execution, no decision making.

## Responsibilities

### V2V DOES (Audio I/O Only)

**Input Side:**
- Capture audio from microphone (or other audio sources)
- Wake word detection
- Voice Activity Detection (VAD)
- Speech-to-Text (ASR)
- Speaker Identification (who is speaking)
- Output: transcript + metadata

**Output Side:**
- Receive text to speak
- Text-to-Speech (TTS)
- Play audio to speaker
- Handle barge-in/interruption

### V2V Does NOT

- Call tools
- Route intents
- Make decisions
- Execute device commands
- Store conversation history
- Manage sessions
- Contain business logic
- Know what "turn on the lights" means

## Public Interface

### 1. Input Event (V2V -> System)

When speech is detected and transcribed, V2V publishes an event:

```python
{
    "transcript": "turn on the lights",
    "speaker_id": "uuid-of-user",
    "speaker_name": "Juan",
    "speaker_confidence": 0.98,
    "session_id": "...",
    "audio_bytes": <optional>,
}
```

### 2. Output Method (System -> V2V)

Any module can request V2V to speak:

```python
v2v.speak(text: str, priority: int = 0)
```

## Module Access Pattern

```
                         +------------------+
                         |   V2V Pipeline   |
                         |   (Audio I/O)    |
                         +--------+---------+
                                  |
                    +-------------+-------------+
                    |                           |
                    v                           v
           transcript + metadata           speak(text)
                    |                           ^
                    |                           |
                    v                           |
         +-------------------------------------------+
         |              MESSAGE BUS / ROUTER         |
         +-------------------------------------------+
                    |                           ^
       +------------+------------+              |
       v            v            v              |
   +-------+   +--------+   +--------+         |
   | Agent |   | Alerts |   |Reminder|----------+
   |       |   |        |   |        |
   +---+---+   +--------+   +--------+
       |
       v
   +-------+
   | Tools |  <-- Tools are HERE, not in V2V
   +-------+
```

## Key Principles

1. **V2V has no dependencies on business logic**
   - Does not import agent, tools, capabilities
   - Pure audio processing

2. **Event-driven decoupling**
   - V2V publishes "user_speech" events
   - V2V subscribes to "speak" requests
   - Router handles the connection

3. **Multiple consumers**
   - Agent, logger, analytics can all receive transcripts
   - No tight coupling

4. **Multiple producers**
   - Agent, alerts, reminders can all call speak()
   - Priority system for conflicts

5. **Swappable components**
   - Replace Piper with ElevenLabs? Only touch V2V
   - Replace Whisper with another ASR? Only touch V2V

6. **Scalable to multiple input sources**
   - Local microphone
   - Phone call (Twilio)
   - WebSocket stream
   - All use same interface

## Future: Phone Integration

Phone calls become another V2V instance with the same interface:

```
Local V2V:  mic -> ASR -> [transcript] -> ... -> [response] -> TTS -> speaker
Phone V2V:  call audio -> ASR -> [transcript] -> ... -> [response] -> TTS -> call audio
```

Both publish the same event format, both accept speak() calls.

## Integration with AlertManager

The existing AlertManager serves as the event hub. V2V registers as:
- A **producer** of "user_speech" events
- A **consumer** (callback) for speak requests

```
AlertManager.register_callback(voice_pipeline_delivery)
```

This allows reminders, alerts, and other modules to trigger speech without knowing about V2V internals.

---

*Document created: 2026-01-29*
*Purpose: Architectural reference for V2V pipeline boundaries*
