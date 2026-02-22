# Atlas Plan

This file tracks upcoming work. Progress, decisions, and retrospectives remain in `BUILD.md`.

## Active Goals (Phase 3 – AI Model Integration)

1. **Stabilize inference**
   - Capture timing/VRAM stats so we know current headroom.
   - Persist helpful debug logs to `logs/` for future troubleshooting.

## Near-Term Backlog

- Improve STT pipeline with faster-whisper and feed transcription results back through the intent router.
- Define response schemas for text/audio/vision requests so API consumers get consistent shape + metadata.
- Add CI-friendly smoke tests (e.g., start FastAPI, hit `/ping`) to catch regressions quickly.

## Future Considerations

- Terminal authentication + multi-tenant session management for remote clients.
- Model management (version pinning, hot reload hooks, telemetry).
- Observability stack (structured logs, metrics, tracing) once Atlas begins serving real terminals.

