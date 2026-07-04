# PR-Atlas-Video-Processing-Kafka-8

## Why this slice exists

Dependabot proposed moving the `atlas_video-processing` compose Kafka image from
Confluent Platform 5.5.3 to 8.2.1. Confluent Platform 8.x Kafka containers no
longer support the old ZooKeeper-mode settings, so this slice keeps the image
bump while moving the local compose broker to single-node KRaft and preserving
the client listener surfaces used by the rest of the stack.

## Scope (this PR)

Ownership lane: atlas-video-processing/kafka-infra
Slice phase: Production hardening

1. Update the compose Kafka image from `confluentinc/cp-kafka:5.5.3` to
   `confluentinc/cp-kafka:8.2.1`.
2. Remove the local ZooKeeper service and ZooKeeper broker wiring.
3. Configure a single-node KRaft broker/controller with `KAFKA_CLUSTER_ID`, node
   ID, controller quorum, controller listener, and listener map.
4. Preserve existing local client endpoints as `kafka:9092` for compose services
   and `localhost:9093` for host access.

### Files touched

- `atlas_video-processing/docker-compose.yml`
- `plans/PR-Atlas-Video-Processing-Kafka-8.md`

### Review Contract

Acceptance criteria:

- [ ] The Kafka service uses the Confluent 8.2.1 image.
- [ ] The compose file no longer configures Kafka through ZooKeeper settings.
- [ ] The Kafka service has the required single-node KRaft roles, node ID,
      controller quorum, controller listener, listener map, and Kafka cluster ID.
- [ ] Existing local client endpoints remain available as `kafka:9092` for
      compose services and `localhost:9093` for host access.
- [ ] No ingest, processing, control, LLM service, Postgres image, runtime
      application code, workflow, or repository-wide guardrail changes are
      included in this slice.

Affected surfaces: config / local compose Kafka.

Risk areas: backcompat / config correctness / deployment safety.

Reviewer rules triggered: R1, R2, R11, R12, R14.

## Mechanism

The `zookeeper` service and `KAFKA_ZOOKEEPER_CONNECT` wiring are removed. The
`kafka` service now runs as a combined KRaft broker/controller with a stable
local `KAFKA_CLUSTER_ID`, node ID, controller quorum, listener map, and
controller listener. Compose services still use `kafka:9092`, and host tools
still use `localhost:9093`.

## Intentional

- Compose-only dependency upgrade; application code and service build contexts
  are unchanged.
- The Postgres service remains on the already-merged `postgres:18` setup from
  the separate Postgres slice.
- No Kafka data volume is added in this slice, matching the existing ephemeral
  local compose behavior.

## Deferred

- Add a container healthcheck or smoke test for the compose Kafka bootstrap path
  if this stack becomes a CI-exercised integration target.
- Parked hardening: none.

## Verification

- Inspected the existing #1632 compose diff and review threads.
- Updated `atlas_video-processing/docker-compose.yml` to remove ZooKeeper wiring
  and configure single-node KRaft while preserving `kafka:9092` and
  `localhost:9093` listener surfaces.
- Used `KAFKA_CLUSTER_ID` for the KRaft cluster ID to address the Copilot review
  finding on the stale Dependabot branch.
- Docker compose was not run in this environment; validation is by compose-file
  inspection plus CI.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_video-processing/docker-compose.yml` | ~30 |
| `plans/PR-Atlas-Video-Processing-Kafka-8.md` | ~88 |
| **Total** | **~118** |
