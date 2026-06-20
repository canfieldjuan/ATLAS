# PR-Atlas-Video-Kafka-Cluster-ID-Env

## Why this slice exists

Root cause: the Kafka 8 replacement PR resolved the stale Dependabot review by
using `KAFKA_CLUSTER_ID`, treating the cluster ID like a normal Kafka broker
property. Confluent's Docker KRaft startup path instead expects the generated
storage-format cluster ID in the special `CLUSTER_ID` environment variable, as
shown in the official `cp-kafka` KRaft example. This change fixes the root by
using the variable consumed by the Confluent container startup script.

## Scope (this PR)

Ownership lane: atlas-video-processing/kafka-infra
Slice phase: Production hardening

1. Rename the Kafka KRaft cluster ID compose environment key from
   `KAFKA_CLUSTER_ID` to `CLUSTER_ID`.
2. Leave all other Kafka 8, listener, KRaft role, and Postgres 18 compose
   settings unchanged.

### Files touched

- `atlas_video-processing/docker-compose.yml`
- `plans/PR-Atlas-Video-Kafka-Cluster-ID-Env.md`

### Review Contract

Acceptance criteria:

- [ ] The Kafka service uses `CLUSTER_ID` for the KRaft storage cluster ID.
- [ ] No Kafka listener, controller quorum, process role, image tag, Postgres,
      service build, workflow, or runtime application code changes are included.
- [ ] The plan names the upstream root cause and why this is not a broad Kafka
      rework.

Affected surfaces: config / local compose Kafka.

Risk areas: config correctness / local compose startup.

Reviewer rules triggered: R1, R2, R11, R12, R14.

## Mechanism

The compose environment key changes from `KAFKA_CLUSTER_ID` to `CLUSTER_ID`
while preserving the same generated ID value. Confluent's Docker startup helper
uses that value to format/start the single-node KRaft broker.

## Intentional

- This is a one-line follow-up to the already-merged Kafka 8 slice, not a
  re-review of the broader compose topology.
- No smoke container startup is added because the existing Kafka slice deferred a
  compose healthcheck/smoke path until this stack becomes a CI-exercised target.

## Deferred

- Keep the existing Kafka healthcheck/smoke-test follow-up deferred from the
  Kafka 8 slice.
- Parked hardening: none.

## Verification

- Checked Confluent's Docker image configuration reference for the `cp-kafka`
  KRaft combined-mode example; it assigns the generated cluster ID with
  `CLUSTER_ID`.
- `docker compose config` from `atlas_video-processing/` passed and rendered
  the Kafka environment with `CLUSTER_ID`; it emitted the existing obsolete
  Compose `version` warning, which is outside this slice.
- Local PR review with the PR body environment loaded: passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_video-processing/docker-compose.yml` | 2 |
| `plans/PR-Atlas-Video-Kafka-Cluster-ID-Env.md` | 78 |
| **Total** | **80** |
