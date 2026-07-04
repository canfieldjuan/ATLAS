# PR-Atlas-Vision-Python-3-14-Base-Image

## Why this slice exists

Dependabot proposed moving the Atlas Vision image from `python:3.11-slim` to `python:3.14-slim`. This image installs FastAPI, OpenCV, Zeroconf, MQTT, and the Ultralytics detection stack, so the base-image bump needs CPython 3.14 wheel evidence before it can be treated as a safe Dockerfile-only update.

## Scope (this PR)

Ownership lane: video-processing/vision-runtime
Slice phase: Production hardening

1. Update `atlas_video-processing/Dockerfile.vision` from `python:3.11-slim` to `python:3.14-slim`.
2. Verify the Atlas Vision Docker requirements have CPython 3.14 Linux wheel coverage for the native dependencies most likely to break the image build.
3. Keep the existing Atlas Vision application code, command, port, healthcheck, and requirements unchanged.

### Files touched

- `atlas_video-processing/Dockerfile.vision`
- `plans/PR-Atlas-Vision-Python-3-14-Base-Image.md`

### Review Contract

Acceptance criteria:

- [ ] `atlas_video-processing/Dockerfile.vision` uses `python:3.14-slim`.
- [ ] No Atlas Vision application code, command, port, healthcheck, or requirements changes are included in this slice.
- [ ] CPython 3.14 Linux wheel coverage exists for the runtime-critical native packages: `opencv-python-headless`, `numpy`, `lapx`, `zeroconf`, `torch`, and `torchvision`.

Affected surfaces: Atlas Vision runtime image / object detection dependencies.

Risk areas: dependency compatibility / native wheel availability / container startup.

Reviewer rules triggered: R1, R2, R11, R12, R14.

## Mechanism

`atlas_video-processing/Dockerfile.vision` keeps the existing image build flow: use the slim Python base, install the current requirements file, copy `atlas_video-processing/atlas_vision`, switch to the non-root `atlas` user, expose port 5002, keep the existing healthcheck, and run the module entrypoint. Only the Python base image tag changes.

The dependency-risk check focuses on CPython 3.14 Linux wheel availability for compiled packages in the Atlas Vision stack and the PyTorch packages pulled by Ultralytics. The remaining top-level packages are also checked with binary-only no-dependency probes so the Docker build does not unexpectedly fall back to source builds for this interpreter.

## Intentional

- Docker base-image maintenance only; no Atlas Vision source changes.
- Keeps the existing `requirements.txt` constraints.
- Keeps the existing healthcheck and runtime environment variables.
- Does not add a Docker build workflow.

## Deferred

- Add a dedicated Docker build smoke for `Dockerfile.vision` if Atlas Vision becomes a release-blocking image artifact.
- Evaluate dependency pinning for the Ultralytics stack in a separate runtime reproducibility slice.

Parked hardening: none.

## Verification

- CPython 3.14 Linux wheel probe passed for `opencv-python-headless`, `numpy`, and `lapx`.
- PyTorch CPU index probe passed for CPython 3.14 Linux wheels for `torch` and `torchvision`.
- Binary-only top-level package probe passed for FastAPI, Uvicorn, Pydantic, Pydantic Settings, HTTPX, AnyIO, Zeroconf, aiomqtt, and Ultralytics.
- Docker build was not run in this environment; validation is by dependency wheel resolution plus CI.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_video-processing/Dockerfile.vision` | ~1 |
| `plans/PR-Atlas-Vision-Python-3-14-Base-Image.md` | ~74 |
| **Total** | **~75** |
