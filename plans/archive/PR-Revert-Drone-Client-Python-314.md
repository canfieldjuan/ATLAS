# PR-Revert-Drone-Client-Python-314

## Why this slice exists

#1637 unexpectedly merged the `drone_client` Docker base from
`python:3.11-slim-bookworm` to `python:3.14-slim-bookworm` after the safe
#1668 base-image slice had already landed. That was a risky infra major:
Python 3.14 was explicitly grouped with the deferred Docker majors, and #1637
also merged with red PR-body contract checks rather than an Atlas plan/review
contract.

Root cause: a rebased Dependabot Docker PR changed the already-safe
`drone_client` base into a Python 3.14 bump, and it was merged outside the
deferred-major queue. This PR fixes the production artifact, not the broader
process root; the process guard remains the existing deferred-label/serialized
review discipline for risky infra majors.

## Scope (this PR)

Ownership lane: infra/docker-base-images
Slice phase: Production hardening

1. Revert only `atlas_video-processing/ingest/drone_client/Dockerfile` from
   `python:3.14-slim-bookworm` back to the #1668-verified
   `python:3.11-slim-bookworm` base.
2. Verify the Dockerfile is back on the intended base and that the image still
   builds/imports its dependency.

### Review Contract

Acceptance criteria:
- `drone_client` uses `python:3.11-slim-bookworm`, matching the #1668 safe
  subset.
- No other Dockerfile, package, or application code changes in this revert.
- Python 3.14 Docker-base bumps remain deferred to dedicated verification
  slices.

Affected surfaces:
- `atlas_video-processing/ingest/drone_client/Dockerfile`

Risk areas:
- Accidentally masking a different Docker dependency drift.
- Reverting beyond the intended one-line base-image change.

Triggered reviewer rules:
- R1 Requirements match
- R2 Test evidence
- R10 Workflow/process
- R14 Codebase verification

### Files touched

- `atlas_video-processing/ingest/drone_client/Dockerfile`
- `plans/PR-Revert-Drone-Client-Python-314.md`

## Mechanism

The Dockerfile `FROM` image is restored from `python:3.14-slim-bookworm` to
`python:3.11-slim-bookworm`. This returns the worker image to the base that was
build-verified in #1668 without touching the worker code or requirements.

## Intentional

- This is a one-line production revert, not a Python 3.14 compatibility
  investigation.
- Do not close or modify the still-deferred Python 3.14 PRs for the other
  Docker images in this slice.

## Deferred

- Dedicated Python 3.14 Docker compatibility work remains deferred for the
  graphiti/vision/drone images until each image has a focused build/runtime
  verification slice.

Parked hardening: none.

## Verification

- `docker manifest inspect python:3.11-slim-bookworm` - passed.
- `docker build -t atlas-drone-client-revert-check:py311 atlas_video-processing/ingest/drone_client` - passed.
- `docker run --rm atlas-drone-client-revert-check:py311 python -c "import sys, kafka; ..."` - passed; printed Python 3.11.15 and kafka-python 3.0.0.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_video-processing/ingest/drone_client/Dockerfile` | 2 |
| `plans/PR-Revert-Drone-Client-Python-314.md` | 89 |
| **Total** | **91** |
