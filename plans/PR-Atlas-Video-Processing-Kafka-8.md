# PR-Atlas-Video-Processing-Kafka-8

## Why this slice exists

Dependabot proposed moving the `atlas_video-processing` compose Kafka image from
Confluent Platform 5.5.3 to 8.2.1. Confluent Platform 8.x Kafka containers run
in KRaft mode rather than ZooKeeper mode, so the compose service must migrate the
broker configuration instead of only changing the image tag.

## Scope (this PR)

Ownership lane: atlas-video-processing/infra
Slice phase: Production hardening

1. Update the compose Kafka image from `confluentinc/cp-kafka:5.5.3` to
   `confluentinc/cp-kafka:8.2.1`.
2. Remove the local ZooKeeper service and ZooKeeper-dependent Kafka environment.
3. Configure the Kafka container as a single-node combined KRaft broker and
   controller for local/dev compose use.
4. Preserve the existing internal and host listener surfaces used by the rest of
   the stack: `kafka:9092` inside compose and `localhost:9093` on the host.
5. Carry the shared Security Guardrails workflow repair already proven on the
   adjacent Dependabot branches.

### Files touched

- `atlas_video-processing/docker-compose.yml`
- `.github/workflows/security_guardrails.yml`
- `plans/PR-Atlas-Video-Processing-Kafka-8.md`

### Review Contract

Acceptance criteria:

- [ ] The Kafka service uses the Confluent 8.2.1 image.
- [ ] The compose file no longer configures Kafka through ZooKeeper settings.
- [ ] The Kafka service has the required single-node KRaft roles, node ID,
      controller quorum, controller listener, listener map, and cluster ID.
- [ ] Existing local client endpoints remain available as `kafka:9092` for
      compose services and `localhost:9093` for host access.
- [ ] No ingest, processing, control, LLM service, Postgres image, or runtime
      application code changes are included in this slice.
- [ ] PR-level guardrails have the same workflow fix used by the recently green
      Dependabot branches.

Affected surfaces: config / local compose Kafka / CI guardrails.

Risk areas: backcompat / config correctness / deployment safety.

Reviewer rules triggered: R1, R2, R11, R12, R14.

## Mechanism

The `zookeeper` service is removed because it is not used by Confluent Platform
8.x Kafka. The `kafka` service is configured as a combined KRaft broker and
controller using a stable local `CLUSTER_ID`, `KAFKA_NODE_ID=1`,
`KAFKA_PROCESS_ROLES=broker,controller`, and a single-node controller quorum.

The listener configuration keeps the same client shape the rest of this compose
stack expects. Compose services continue to reach Kafka at `kafka:9092`, while
host tooling can still use `localhost:9093`.

## Intentional

- Compose-only dependency upgrade; application code and service build contexts
  are unchanged.
- The Postgres service remains on `postgres:13`; the Postgres 18 upgrade is a
  separate PR with separate stateful-data handling.
- No Kafka data volume is added in this slice, matching the existing ephemeral
  local compose behavior.

## Deferred

- A follow-up can add a container healthcheck or smoke test for the compose
  Kafka bootstrap path if this stack becomes a CI-exercised integration target.

## Parked hardening

None.

## Verification

- Inspected the PR diff and unresolved review thread: the unsafe ZooKeeper-mode
  configuration against a Confluent 8.x image was the blocking issue.
- Updated `atlas_video-processing/docker-compose.yml` to remove ZooKeeper wiring
  and configure single-node KRaft while preserving `kafka:9092` and
  `localhost:9093` listener surfaces.
- Docker compose was not run in this environment; validation is by compose-file
  inspection only.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_video-processing/docker-compose.yml` | ~26 |
| `.github/workflows/security_guardrails.yml` | shared repair |
| `plans/PR-Atlas-Video-Processing-Kafka-8.md` | ~92 |
| **Total** | **~118** |
