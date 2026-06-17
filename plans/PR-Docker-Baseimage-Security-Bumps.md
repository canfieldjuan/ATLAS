# PR-Docker-Baseimage-Security-Bumps

Plan: plans/PR-Docker-Baseimage-Security-Bumps.md
Slice phase: Production hardening
Ownership lane: infra/docker-base-images
Reviewer rules triggered: none (Docker base-image FROM bumps only; no path
glob in docs/REVIEWER_RULES.md matches Dockerfiles).

## Why this slice exists

Dependabot opened a batch of Docker base-image security bumps. This slice lands
only the low-risk, verifiable subset and deliberately holds the risky majors
for separate, dev-gated decisions. It runs in its own worktree to stay clear of
the parallel npm security batch and the in-flight workflow-action-pin work,
which touch entirely different files.

Two bumps qualify as low-risk:

- Root Dockerfile (the Atlas brain image): nvidia/cuda 12.1.1 -> 12.2.2 (same
  -cudnn8-runtime-ubuntu22.04 variant). A CUDA minor bump within the 12.x line;
  torch/torchaudio are unpinned in requirements, so the pip wheels carry their
  own CUDA runtime and do not pin a +cu121 build that could mismatch the base.
- drone_client worker image: python 3.9-slim-buster -> 3.11-slim-bookworm.
  Retires BOTH end-of-life Python 3.9 and end-of-life Debian 10 (buster) in one
  bump (a buster-only bump would ship the security fix onto an unsupported OS).
  The only dependency is the pure-Python kafka-python, so the bookworm
  (Debian 12) base carries no native-build risk.

## Scope

Slice phase: Production hardening

Two one-line FROM edits to existing Dockerfiles. No application code, no
compose files, no workflow files, no extracted_* changes. The plan doc is the
only added file.

### Files touched

- `Dockerfile`
- `atlas_video-processing/ingest/drone_client/Dockerfile`
- `plans/PR-Docker-Baseimage-Security-Bumps.md`

## Mechanism

Bump the two FROM pins to the Dependabot-proposed tags. Both proposed tags were
confirmed to exist on their registries with docker manifest inspect before
editing. The drone-client image was then built and run to confirm the bumped
interpreter and its dependency import cleanly.

## Intentional

- Only the brain-image CUDA minor bump and the drone-client Python 3.9 -> 3.11
  bump are included, because both are verifiable as low-risk here.
- The drone-client image is fully build-verified (built, ran, imported its
  dependency). The CUDA bump is verified by tag existence plus CUDA
  minor-version compatibility reasoning, not by a full local image build (see
  Verification).

## Deferred

Held for separate, dev-gated decisions (not in this PR):

- video_stream_processor python 3.9 -> 3.13: the dependency
  (opencv-python-headless) installs cleanly on 3.13, but that Dockerfile has a
  pre-existing missing-WORKDIR bug that makes it fail to build on BOTH the old
  3.9 and the new 3.13 base. Needs a dedicated slice that fixes the Dockerfile
  and bumps Python together so the bump is build-verifiable.
- graphiti-wrapper and vision images python 3.11 -> 3.14: Python 3.14 is too
  new for the torch / graphiti / SAM dependency graphs.
- confluentinc/cp-kafka 5.5.3 -> 8.2.1: major Confluent Platform jump (KRaft
  removes ZooKeeper, but the compose still runs cp-zookeeper).
- postgres 13 -> 18: five-major data-directory migration.

## Verification

- docker manifest inspect nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04 -> tag
  EXISTS.
- docker manifest inspect python:3.11-slim-bookworm -> tag EXISTS.
- drone_client: docker build of atlas_video-processing/ingest/drone_client
  succeeded; docker run of the image with python -c "import sys, kafka" printed
  Python 3.11.15 and kafka-python 3.0.0, on Debian GNU/Linux 12 (bookworm).
- Root CUDA Dockerfile: NOT built locally (multi-GB CUDA base plus the full
  requirements install, including torch, is too heavy to build in this
  environment). Verification ceiling: tag existence, CUDA 12.x minor-compat,
  and torch being unpinned. NOTE: no CI workflow runs docker build of the brain
  image (the check workflows run pytest), so the real backstop is the
  deploy-time docker compose up --build, not CI.

## Estimated diff size

| Area | LOC |
|---|---|
| Dockerfile FROM bumps (2 files) | 4 |
| This plan doc (added) | ~95 |
| **Total** | **~99** |
