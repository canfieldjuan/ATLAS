# Atlas Build Specification

**Last Updated**: 2025-01-22
**Status**: DRAFT - Pending review

---

## What Atlas Is

Atlas is a voice-first home assistant that:
1. Listens for wake word "Hey Atlas"
2. Understands spoken commands
3. Takes action (controls devices, answers questions, executes tools)
4. Responds with voice

**Core principle**: Everything flows through ONE unified pipeline. No parallel implementations.

---

## Priority Stack

### P0: Voice-to-Voice (CURRENT FOCUS)
The foundation. Nothing else matters until this works end-to-end.

**Flow**:
```
Microphone → Wake Word → STT → Atlas Agent → TTS → Speaker
```

**Definition of Done**:
- [ ] Wake word "Hey Atlas" activates listening
- [ ] Speech is transcribed accurately (STT)
- [ ] Transcription goes to Atlas Agent
- [ ] Agent generates response
- [ ] Response is spoken back (TTS)
- [ ] Works in a continuous loop (can handle multiple interactions)

**Current State**:
- [ ] TODO: Document actual current state

**Blocked By**: Nothing - this is P0

---

### P1: Home Assistant Integration
Voice commands control real devices.

**Flow**:
```
Voice Input → Atlas Agent → Intent Detection → HA API → Device Action → Voice Confirmation
```

**Definition of Done**:
- [ ] "Turn on the living room lights" → lights turn on → "Done" or "Lights are on"
- [ ] "What's the temperature?" → reads from HA sensor → speaks value
- [ ] Device state changes reflected in Atlas responses
- [ ] Works through the P0 voice pipeline (not a separate path)

**Blocked By**: P0 Voice-to-Voice

---

### P2: Atlas-Native Features
Standalone tools that don't require HA.

**Features** (in order):
1. **Time/Date** - "What time is it?"
2. **Weather** - "What's the weather?"
3. **Timers** - "Set a timer for 5 minutes"
4. **Reminders** - "Remind me to call mom at 3pm"
5. **Calendar** - "What's on my calendar today?"

**Definition of Done** (per feature):
- [ ] Accessible via voice through P0 pipeline
- [ ] Tool executes correctly
- [ ] Result spoken back to user

**Blocked By**: P0 Voice-to-Voice

---

## Architecture (Single Pipeline)

```
┌─────────────────────────────────────────────────────────────┐
│                     VOICE PIPELINE                          │
│                                                             │
│  ┌─────────┐    ┌─────┐    ┌─────────────┐    ┌─────┐      │
│  │ Wake    │───▶│ STT │───▶│ Atlas Agent │───▶│ TTS │      │
│  │ Word    │    │     │    │             │    │     │      │
│  └─────────┘    └─────┘    └──────┬──────┘    └─────┘      │
│                                   │                         │
│                          ┌────────┴────────┐               │
│                          │                 │               │
│                    ┌─────▼─────┐    ┌──────▼──────┐        │
│                    │ HA Tools  │    │ Native Tools│        │
│                    │ (P1)      │    │ (P2)        │        │
│                    └───────────┘    └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

**Rule**: All features MUST flow through the Atlas Agent. No direct API bypasses for voice features.

---

## Out of Scope (For Now)

These are NOT being worked on until P0-P2 are complete:

- [ ] Phone/SMS integration (PersonaPlex)
- [ ] Vision/Camera features
- [ ] Multi-room audio
- [ ] Web UI dashboard
- [ ] Mobile app
- [ ] Multi-user support
- [ ] Custom wake words

---

## Integration Requirements

For ANY feature to be considered "done":

1. **Single Entry Point**: Accessed through the voice pipeline
2. **Agent Routing**: Atlas Agent decides what to do (not hardcoded paths)
3. **Voice Response**: Result is spoken, not just logged
4. **Error Handling**: Failures are spoken ("Sorry, I couldn't do that")
5. **Tested End-to-End**: Not just unit tested, actually works voice-to-voice

---

## Components

### Required for P0
| Component | Purpose | Status |
|-----------|---------|--------|
| Wake Word | Detects "Hey Atlas" | ? |
| STT | Speech to text | ? |
| Atlas Agent | Processes input, decides action | ? |
| TTS | Text to speech | ? |
| Audio I/O | Mic input, speaker output | ? |

### Required for P1
| Component | Purpose | Status |
|-----------|---------|--------|
| HA Backend | Connects to Home Assistant | ? |
| Device Registry | Knows available devices | ? |
| Intent Parser | Maps speech to HA actions | ? |

### Required for P2
| Component | Purpose | Status |
|-----------|---------|--------|
| Tool Registry | Available tools | ? |
| Tool Executor | Runs tools | ? |
| Individual Tools | Time, weather, etc. | ? |

---

## What "Wired In" Means

A component is NOT done if:
- It only works via direct API call
- It has its own separate entry point
- It doesn't go through Atlas Agent
- It doesn't produce voice output
- It's tested in isolation but not integrated

A component IS done when:
- "Hey Atlas, [command]" → [action] → [spoken response]
- Works in the real system, not just a test script

---

## Review Checklist

Before marking anything complete:

1. [ ] Does it align with current priority (P0/P1/P2)?
2. [ ] Does it flow through the single voice pipeline?
3. [ ] Does the Atlas Agent route it (not hardcoded)?
4. [ ] Does it produce a voice response?
5. [ ] Has it been tested end-to-end with actual voice?
6. [ ] Is CONTEXT.md updated?
7. [ ] Is INTEGRATION_MAP.md updated?
