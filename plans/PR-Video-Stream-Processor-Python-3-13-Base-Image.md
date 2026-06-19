# PR-Video-Stream-Processor-Python-3-13-Base-Image

## Why this slice exists

Dependabot proposed moving the video stream processor worker image from
`python:3.9-slim-bullseye` to a Python 3.13 slim image. The generated target tag
used the old bullseye suite, so this slice keeps the Python upgrade while moving
to the currently published `python:3.13-slim-bookworm` tag.

## Scope (this PR)

Ownership lane: video-processing/stream-worker-runtime
Slice phase: Production hardening

1. Update `atlas_video-processing/processing/video_stream_processor/Dockerfile`
   from Python 3.9 slim bullseye to Python 3.13 slim bookworm.
2. Keep the worker command, package install flow, system dependency list, and
   non-root runtime user unchanged.
3. Carry the shared Security Guardrails workflow repair already proven on the
   adjacent Dependabot branches.

### Files touched

- `atlas_video-processing/processing/video_stream_processor/Dockerfile`
- `.github/workflows/security_guardrails.yml`
- `plans/PR-Video-Stream-Processor-Python-3-13-Base-Image.md`

### Review Contract

Acceptance criteria:

- [ ] The video stream processor Dockerfile uses a published Python 3.13 slim
      base image tag.
- [ ] The worker remains on Debian slim with no application-code, command, or
      requirements changes in this slice.
- [ ] The runtime dependency stack has CPython 3.13 Linux wheel coverage for
      `opencv-python-headless`.
- [ ] PR-level guardrails have the same workflow fix used by the recently green
      Dependabot branches.

Affected surfaces: video processing worker runtime image / Docker base image / CI guardrails.

Risk areas: image tag availability / native library compatibility / worker startup.

Reviewer rules triggered: R1, R2, R11, R12, R14.

## Mechanism

The Dockerfile now uses `python:3.13-slim-bookworm`. The rest of the image build
remains unchanged: install the existing OpenCV system libraries, install
`requirements.txt`, copy the worker source, create the `atlas` user, and run
`video_stream_processor.py`.

Using bookworm fixes the review finding about the generated bullseye tag while
keeping the maintenance intent of the Dependabot PR: move this worker to Python
3.13 on a slim Debian image.

## Intentional

- Docker base-image maintenance only; no video processing source changes.
- Keeps the existing OpenCV-related apt packages.
- Keeps `opencv-python-headless` unpinned because this PR is scoped to the base
  image bump.
- Does not add a Docker build workflow.

## Deferred

- Add a dedicated Docker build smoke for this worker if it becomes a required CI
  release artifact.
- Evaluate whether the OpenCV apt package list can be reduced on bookworm in a
  separate runtime-hardening slice.

## Parked hardening

None.

## Verification

- Official Python image tag source checked for the Python 3.13 slim Debian tags.
- Docker Registry probe confirmed `python:3.13-slim-bookworm` is pullable.
- CPython 3.13 Linux wheel probe passed for `opencv-python-headless`.
- Docker build was not run in this environment; validation is by tag and wheel
  resolution plus CI.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_video-processing/processing/video_stream_processor/Dockerfile` | ~1 |
| `.github/workflows/security_guardrails.yml` | shared repair |
| `plans/PR-Video-Stream-Processor-Python-3-13-Base-Image.md` | ~80 |
| **Total** | **~106** |
