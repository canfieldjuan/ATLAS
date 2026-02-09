# Atlas Context - Current State

**Last Updated**: 2025-01-22
**Current Focus**: P0 - Voice-to-Voice

---

## Session Notes

### 2025-01-22
- Tabled PersonaPlex phone integration (latency issues, not P0)
- Established BUILD_SPEC and project management structure
- Priority clarified: Voice-to-Voice → HA → Native Features
- Created CANONICAL.md to track "which implementation is real"
- Created AUDITOR_PROMPT.md for local model systems auditor
- Hardware upgrade planned: 32GB VRAM, second GPU

---

## Current Work in Progress

| Task | Status | Notes |
|------|--------|-------|
| P0 Voice Pipeline | AUDITED | Pipeline exists and is wired - need E2E test |

---

## Known Debt / Incomplete Integrations

| Component | Issue | Impact |
|-----------|-------|--------|
| Pipecat | Built but not wired into Atlas | Voice doesn't flow through agent |
| Text API | Wired to Atlas differently than voice | Inconsistent behavior |
| Voice API | May not go through Atlas Agent | Bypasses routing |
| PersonaPlex | Phone integration incomplete | 22s latency, not P0 anyway |

---

## Component Status Audit

### Voice Pipeline (P0)

| Component | Exists? | Wired? | Works E2E? | Notes |
|-----------|---------|--------|------------|-------|
| Wake Word | ? | ? | ? | Need to audit |
| STT | Yes (Nemotron) | ? | ? | In services/stt/nemotron.py |
| Atlas Agent | Yes | ? | ? | In agents/atlas.py |
| TTS | ? | ? | ? | Need to audit |
| Audio I/O | ? | ? | ? | Need to audit |

### Home Assistant (P1)

| Component | Exists? | Wired? | Works E2E? | Notes |
|-----------|---------|--------|------------|-------|
| HA Backend | Yes | ? | ? | In capabilities/backends/ |
| Device Registry | Yes | ? | ? | capability_registry |
| Intent Parser | Yes | ? | ? | intent_parser.py |

### Native Tools (P2)

| Tool | Exists? | Wired? | Works E2E? | Notes |
|------|---------|--------|------------|-------|
| Time | ? | ? | ? | |
| Weather | ? | ? | ? | |
| Timers | ? | ? | ? | |
| Reminders | Yes | ? | ? | reminder system exists |
| Calendar | Yes | ? | ? | google calendar tool |

---

## Blocked Items

| Item | Blocked By | Resolution |
|------|------------|------------|
| P1 HA Integration | P0 Voice Pipeline | Complete P0 first |
| P2 Native Features | P0 Voice Pipeline | Complete P0 first |

---

## Questions to Resolve

1. What is the ACTUAL current state of the voice pipeline?
2. Does voice input currently go through Atlas Agent?
3. What STT/TTS is actually being used for local voice?
4. Is there a working wake word detector?

---

## Next Actions

1. [ ] Audit current voice pipeline components
2. [ ] Map what's wired vs what exists
3. [ ] Identify gaps in P0
4. [ ] Create plan to complete P0

---

## Architecture Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-01-22 | Single voice pipeline, no parallel paths | Prevent disconnected features |
| 2025-01-22 | All features through Atlas Agent | Central routing, consistent behavior |
| 2025-01-22 | P0 before P1 before P2 | Foundation first |
