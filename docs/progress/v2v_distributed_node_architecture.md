# V2V Distributed Node Architecture

## Status: Planning

## Overview

Build node awareness into the V2V pipeline to support distributed edge devices (voice + vision) across multiple rooms/locations. This enables context-aware responses where Atlas knows WHERE commands come from.

## Architecture

```
+-----------------------------------------------------------------------------+
|                         DISTRIBUTED ATLAS SYSTEM                             |
+-----------------------------------------------------------------------------+
|                                                                              |
|   EDGE NODES                                      CENTRAL BRAIN              |
|   ----------                                      -------------              |
|                                                                              |
|   +-------------+                                +---------------------+     |
|   |   Kitchen   |--------+                       |                     |     |
|   | mic+speaker |        |                       |    Atlas Brain      |     |
|   | + camera    |        |                       |                     |     |
|   +-------------+        |     WebSocket         |  - LangGraph Agent  |     |
|                          +-------------------->  |  - LLM (Ollama)     |     |
|   +-------------+        |                       |  - Tools            |     |
|   | Living Room |--------+                       |  - Memory           |     |
|   | mic+speaker |        |                       |                     |     |
|   +-------------+        |  <------------------  |                     |     |
|                          |     Response/TTS      +---------------------+     |
|   +-------------+        |                                                   |
|   |   Bedroom   |--------+                                                   |
|   | mic+speaker |                                                            |
|   +-------------+                                                            |
|                                                                              |
+-----------------------------------------------------------------------------+
```

## Context Awareness Flow

```
"Turn on the lights" from kitchen node
            |
            v
+------------------------------------------+
|   Input Event:                           |
|   {                                      |
|     transcript: "turn on the lights",    |
|     node_id: "kitchen",                  |
|     speaker_id: "uuid-juan",             |
|     speaker_name: "Juan"                 |
|   }                                      |
+------------------------------------------+
            |
            v
Agent understands: "lights" = kitchen lights
            |
            v
+------------------------------------------+
|   Output:                                |
|   v2v.speak(                             |
|     text="Kitchen lights are on",        |
|     target_node="kitchen"                |
|   )                                      |
+------------------------------------------+
```

## V2V Pipeline Requirements

### Input (from edge nodes)
- node_id: Which node the audio came from
- All existing metadata (transcript, speaker_id, etc.)

### Output (to edge nodes)
- speak(text, target_node): Respond to specific node
- broadcast(text, exclude_nodes): Announce to all/some nodes

### Endpoints
- `ws://.../v2v/stream/{node_id}` - Edge nodes connect
- `POST /v2v/speak` - Target specific node
- `POST /v2v/broadcast` - Announce to multiple nodes

## Vision + Voice Corroboration

Voice heard in: kitchen (node_id)
Speaker ID: Juan
Camera sees: Juan in kitchen

Corroborated location with high confidence passed to agent.

## What Runs Where

| Component | Edge Node | Central Brain |
|-----------|-----------|---------------|
| Wake word detection | Yes | No |
| Audio capture | Yes | No |
| Audio streaming | Yes (to brain) | Receives |
| ASR | No | Yes |
| Speaker ID | No | Yes |
| Agent/LLM | No | Yes |
| TTS generation | No | Yes |
| TTS playback | Yes | No |
| Camera capture | Yes | No |
| Person detection | Maybe (light) | Yes (full) |

## Implementation Phases

See implementation plan (created separately after codebase analysis)

---

*Document created: 2026-01-30*
*Status: Logged, awaiting implementation plan approval*
