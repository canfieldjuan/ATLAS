# Phone Receptionist Architecture

## Status: Research / Planning

## Problem Statement

Attempted phone integration using Nvidia Personaplex encountered several issues:

### 1. Latency Issues
- **Observed**: 23-40 seconds for initial "Hello"
- **Expected**: 300ms (achieved in test server)
- **Root cause**: Full system prompt sent to LLM on every turn, no KV cache/prefill
- **Evidence**: Latency scaled with prompt length

### 2. Complex Routing Requirements
Customer intents require different flows:

| Intent | Flow Needed |
|--------|-------------|
| "Just browsing" | Info → FAQ → Soft pitch → Maybe follow-up |
| "Book for Tuesday" | Check availability → Confirm → Book |
| "What's available?" | Query calendar → List options |
| "What do you do?" | Business info → Services → Pricing |
| "Talk to a human" | Escalation → Transfer |

One giant prompt cannot handle all these cleanly.

### 3. Customization Limitations
- Personaplex difficult to configure for multi-path flows
- Limited control over prompt structure
- Hard to implement branching conversation logic

### 4. VRAM / Resource Concerns
- Can't control when customers call
- Each call consumes VRAM if using local GPU
- Local Atlas GPU shouldn't be hostage to phone volume

## Proposed Architecture

```
+-----------------------------------------------------------------------------+
|                         PHONE RECEPTIONIST SYSTEM                            |
+-----------------------------------------------------------------------------+
|                                                                              |
|   +-----------+     +-------------------------------------+                 |
|   |  Twilio   |---->|       Intent Classifier             |                 |
|   |  (Phone)  |     |  (Small, fast model - <100ms)       |                 |
|   +-----------+     +----------------+--------------------+                 |
|                                      |                                       |
|        +-------------+---------------+---------------+------------+         |
|        v             v               v               v            v         |
|   +---------+  +---------+    +-----------+   +-------+   +----------+     |
|   |  Info   |  | Booking |    |Availability|  |  FAQ  |   | Escalate |     |
|   |  Flow   |  |  Flow   |    |   Flow    |   | Flow  |   |   Flow   |     |
|   |         |  |         |    |           |   |       |   |          |     |
|   | Cached  |  | Cached  |    | Cached    |   |Cached |   | Transfer |     |
|   |prompt A |  |prompt B |    | prompt C  |   |prompt |   | to human |     |
|   +---------+  +---------+    +-----------+   +-------+   +----------+     |
|                                                                              |
|   Each flow has:                                                            |
|   - Its own SHORT, focused prompt                                           |
|   - Pre-warmed KV cache                                                     |
|   - Specific tools (calendar, booking API, etc.)                           |
|                                                                              |
+-----------------------------------------------------------------------------+
```

## LLM Stack Options

### Option A: Personaplex as Voice Layer Only
```
Twilio -> Personaplex (ASR/TTS only) -> Your LangGraph (routing + logic) -> Personaplex (TTS)
```
- Keep Personaplex for what it's good at (voice)
- Handle logic in your own LangGraph
- Best of both worlds

### Option B: Fully Custom Stack
```
Twilio -> Your ASR (Whisper/Deepgram) -> Your LangGraph -> Your TTS (Piper/ElevenLabs)
```
- Full control
- More setup work
- Can run locally or cloud

### Option C: Cloud LLM for Phone (Keep Local GPU Free)
```
Twilio -> Cloud ASR -> LangGraph + Groq/Together/OpenAI -> Cloud TTS
```
- Local Atlas GPU stays free for local voice
- Phone calls use cloud resources
- Scales with demand
- Per-call cost but no VRAM contention

## Why LangGraph Fits

LangGraph is built for exactly this:

```
START -> Intent Classifier -> [booking | info | availability | faq | escalate]
                                      |
                              Specialized subgraph
                                      |
                                  Response
```

Each subgraph has:
- Its own system prompt (short, focused)
- Its own tools (booking API, calendar, etc.)
- Its own state management
- Fallback logic

## Key Fixes for Latency

1. **Short, focused prompts** - Each flow has minimal prompt
2. **KV cache prefill** - System prompts pre-cached on startup
3. **Intent classification first** - Small fast model routes, then specialized model handles
4. **Stateful sessions** - Don't re-send full context every turn

## Separation from Local Atlas

Phone receptionist should be **completely separate** from local Atlas:

```
+-------------------+          +-------------------+
|   Local Atlas     |          | Phone Receptionist|
|                   |          |                   |
| - Local mic/speaker|         | - Twilio calls    |
| - Home automation |          | - Appointment booking|
| - Personal assistant|        | - Business hours  |
| - Uses local GPU  |          | - Uses cloud/separate GPU|
+-------------------+          +-------------------+
         |                              |
         +------------------------------+
                      |
              Shared services:
              - Calendar API
              - Booking system
              - Customer database
```

They share data/APIs but run independently.

## Next Steps (When Ready)

1. Decide on LLM stack (Option A, B, or C)
2. Build intent classifier (small, fast)
3. Design individual flow prompts
4. Implement in LangGraph
5. Test latency with prefill
6. Integrate with Twilio

---

*Document created: 2026-01-29*
*Status: Logged for future implementation*
