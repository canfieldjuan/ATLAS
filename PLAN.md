# Atlas Plan

This file tracks upcoming work. Progress, decisions, and retrospectives remain in `BUILD.md`.

## Active Goals (Phase 3 – AI Model Integration)

1. **Integrate moondream2 vision support**
   - Wire `process_vision_query` to load/run the VLM using the same lifecycle as text queries.
   - Provide end-to-end tests (manual `curl` is fine) that confirm an image + prompt round trip from API → service → model.
2. **Stabilize inference**
   - Capture timing/VRAM stats so we know current headroom.
   - Persist helpful debug logs to `logs/` for future troubleshooting.

## Near-Term Backlog

- Replace the mocked STT response with a lightweight speech model and feed transcription results back through the VLM.
- Define response schemas for text/audio/vision requests so API consumers get consistent shape + metadata.
- Add CI-friendly smoke tests (e.g., start FastAPI, hit `/ping`) to catch regressions quickly.

## Future Considerations

- Terminal authentication + multi-tenant session management for remote clients.
- Model management (version pinning, hot reload hooks, telemetry).
- Observability stack (structured logs, metrics, tracing) once Atlas begins serving real terminals.

