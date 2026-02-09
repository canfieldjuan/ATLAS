# V2V Pipeline: Tone, Voice, and Language

## Status: Logged for Future Implementation

## What Atlas Can Detect from User

| Signal | Detectable? | How |
|--------|-------------|-----|
| **Who** (speaker ID) | Yes | Resemblyzer (implemented) |
| **What** (words) | Yes | ASR transcript |
| **Speed** (urgency) | Possible | Audio analysis (words/sec) |
| **Volume** (emphasis) | Possible | RMS levels already tracked |
| **Emotion** (tone) | Possible | Requires emotion model |
| **Language** | Yes | Can detect from transcript |

## What Atlas Can Control in Response

| Aspect | Current Capability | Could Add |
|--------|-------------------|-----------|
| **Voice selection** | Single Piper voice | Multiple voices/personas |
| **Speech rate** | `length_scale` param | Dynamic based on context |
| **Response length** | LLM controlled | User preference / urgency |
| **Formality** | LLM prompt | Per-user preference |
| **Language** | LLM can switch | Auto-detect and match |

## Per-User Preferences (5-User Model)

```
+-------------------------------------------------------------------+
|                    USER PREFERENCES TABLE                          |
+-------------------------------------------------------------------+
| user_id | voice    | speech_rate | verbosity | formality          |
+-------------------------------------------------------------------+
| Juan    | "atlas"  | 1.0         | "concise" | "casual"           |
| Maria   | "luna"   | 0.9         | "normal"  | "casual"           |
| Guest   | "atlas"  | 1.0         | "verbose" | "professional"     |
+-------------------------------------------------------------------+
```

Speaker ID -> lookup preferences -> apply to response generation + TTS

## TTS Voice Options

### Piper (current)
- Multiple voice models available
- Limited emotion control
- Fast, local, free
- Has Spanish, French, German, etc.

### Coqui XTTS (upgrade option)
- Voice cloning
- Better prosody control
- More natural
- Heavier (needs GPU)

### ElevenLabs (cloud option)
- Excellent emotion/tone control
- Voice cloning
- Per-character cost
- Adds latency (API call)

## Language Detection & Switching

```
User: "Enciende las luces"
        |
        v
Language Detection: Spanish
        |
        v
Agent responds in Spanish
        |
        v
TTS uses Spanish voice model
```

## Priority Recommendations

| Feature | Complexity | User Value | Priority |
|---------|------------|------------|----------|
| Per-user voice preference | Low | High | P1 |
| Per-user verbosity | Low | Medium | P2 |
| Language auto-detect/switch | Medium | High (if multilingual) | P2 |
| Speech rate adaptation | Medium | Medium | P3 |
| Emotion detection | High | Medium | P4 |
| Emotion in TTS | High | Low-Medium | P4 |

## Architecture Note

Context for tone/preferences flows through V2V metadata:

```
{
  "transcript": "turn on the lights",
  "speaker_id": "uuid",
  "speaker_name": "Juan",
  "preferences": {
    "voice": "atlas",
    "verbosity": "concise",
    "language": "en"
  }
}
```

---

*Document created: 2026-01-29*
*Status: Logged for future implementation after core V2V complete*
