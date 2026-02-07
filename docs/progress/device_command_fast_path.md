# Device Command Fast Path - Embedding-Based Device Resolver

**Date:** 2026-02-07
**Status:** Implemented

## Problem

"Turn on the TV" took ~14 seconds end-to-end. The LLM intent parser sent a 1026-token prompt to Qwen3-30b just to extract `{"action":"turn_on","target_name":"32\" Philips Roku TV"}`. The semantic intent router already classified this as `device_command` in 10ms, but the LLM was still needed to resolve the device name.

## Solution

Created a `DeviceResolver` that reuses the `all-MiniLM-L6-v2` embedding model (already loaded by `SemanticIntentRouter`) to match queries against HA-discovered device names. Actions are extracted via a keyword map. Returns a complete `Intent` with `target_id` -- skips LLM entirely.

LLM remains as fallback for complex queries, ambiguous matches, and pronoun resolution.

## Flow Comparison

```
BEFORE (14s):
  classify_intent (10ms) -> parse_intent [LLM: 14,000ms] -> execute (7ms)

AFTER (~30ms):
  classify_intent (10ms) -> parse_intent [DeviceResolver: 20ms] -> execute (7ms)
```

## Architecture

- `DeviceResolver` shares the embedding model via `get_intent_router().get_embedder()`
- Device name aliases are generated (e.g., "32 Philips Roku TV" -> ["Roku TV", "TV"])
- Per-device centroid computed from alias embeddings
- Action extracted via keyword map (not regex, not LLM)
- Confidence threshold (0.45) and ambiguity gap (0.05) prevent false matches
- Pronoun detection routes to LLM for entity tracking

## Files Changed

| File | Change |
|------|--------|
| `atlas_brain/config.py` | Added `DeviceResolverConfig` + Settings field |
| `atlas_brain/services/intent_router.py` | Added `get_embedder()` method |
| `atlas_brain/capabilities/device_resolver.py` | New - core resolver |
| `atlas_brain/capabilities/intent_parser.py` | Fast path before LLM + invalidation |
| `atlas_brain/capabilities/homeassistant.py` | Index invalidation on HA changes |
| `.env` | New config vars |

## Fallback Cases

| Case | Behavior |
|------|----------|
| Pronouns ("turn it on") | -> LLM + entity tracker |
| No action keyword ("how's the TV") | -> LLM |
| Below confidence threshold | -> LLM |
| Ambiguous (2 devices score close) | -> LLM |
| Action not supported by device | -> LLM |
| Embedder not loaded yet | -> LLM |

## Configuration

```env
ATLAS_DEVICE_RESOLVER_ENABLED=true
ATLAS_DEVICE_RESOLVER_CONFIDENCE_THRESHOLD=0.45
ATLAS_DEVICE_RESOLVER_AMBIGUITY_GAP=0.05
```
