# PR-Workflow-Service-Image-Digests

## Why this slice exists

The security/workflow hardening queue still has mutable workflow
supply-chain refs. After the setup-python SHA pin slice, the posture audit has
only two non-action mutable service-image warnings: both migration workflows
start a GitHub Actions service container from `postgres:16`.

Root cause: these workflows name a mutable Docker tag without an immutable
digest, so the image content used by CI can change without any Atlas commit.
This fixes that root for the remaining workflow service-image warnings by
pinning both call sites to the current Docker Hub `postgres:16` OCI index
digest while leaving the readable `postgres:16` breadcrumb in place.

## Scope (this PR)

Ownership lane: security/workflow
Slice phase: Production hardening

1. Pin the two workflow `postgres:16` service images to the current Docker Hub
   OCI index digest.
2. Prove the workflow posture audit no longer reports service-image digest
   warnings for those migration workflows.

### Review Contract

Acceptance criteria:

- The only changed service image values are in
  `.github/workflows/atlas_b2b_campaign_migration_checks.yml` and
  `.github/workflows/atlas_deflection_migration_apply_checks.yml`.
- Both values keep the `postgres:16` tag and append the same
  `@sha256:081f1bc7bd5e143dbb6e487b710bbc27712cdcfaced4c071b8e47349aa1b4171`
  OCI index digest.
- `scripts/audit_workflow_security_posture.py .github/workflows` reports no
  `postgres:16 is not digest-pinned` warnings.
- No migration test commands, database credentials, ports, or health checks
  change.

Affected surfaces:

- GitHub Actions migration checks that boot Postgres service containers.
- Workflow supply-chain posture audit output.

Risk areas:

- YAML service-image syntax must remain valid for GitHub Actions.
- Digest must refer to the current `postgres:16` multi-platform image index,
  not an unrelated image or architecture-only manifest.

Triggered reviewer rules:

- R1 Requirements match
- R2 Test evidence
- R3 Security/auth
- R8 CI/workflow safety
- R13 Class-fix coverage
- R14 Codebase verification

### Files touched

- `.github/workflows/atlas_b2b_campaign_migration_checks.yml`
- `.github/workflows/atlas_deflection_migration_apply_checks.yml`
- `HARDENING.md`
- `plans/INDEX.md`
- `plans/PR-Workflow-Service-Image-Digests.md`
- `plans/archive/PR-Workflow-Setup-Python-Pin.md`

## Mechanism

Docker image references can include both the mutable tag and immutable digest:

```yaml
image: postgres:16@sha256:<oci-index-digest>
```

GitHub Actions still presents the familiar Postgres major-version tag, while
the runner pulls exactly the digest recorded in the workflow. The digest was
resolved from Docker Hub with the registry API on 2026-06-17:

```text
repository: library/postgres
tag: 16
Docker-Content-Digest: sha256:081f1bc7bd5e143dbb6e487b710bbc27712cdcfaced4c071b8e47349aa1b4171
Content-Type: application/vnd.oci.image.index.v1+json
```

## Intentional

- This does not touch `actions/checkout@v4`, `actions/setup-node@v4`, or the
  pinned security action SHAs. Several of those refs already have Dependabot
  PRs open, and this slice drains only the service-image class.
- This pins the OCI index digest rather than an amd64-only manifest so the
  reference stays platform-correct for GitHub-hosted Linux runners and future
  platform selection.
- This does not change the Postgres major version, credentials, ports, or
  health checks; it is intended to be a supply-chain immutability change only.

## Deferred

- Remaining mutable workflow action refs stay parked under "Pin remaining
  mutable workflow supply-chain refs" and should be drained through dedicated
  pinning or Dependabot-triage slices.

Parked hardening: none.

## Verification

- Registry digest resolution for `postgres:16` -- passed:
  `sha256:081f1bc7bd5e143dbb6e487b710bbc27712cdcfaced4c071b8e47349aa1b4171`,
  `application/vnd.oci.image.index.v1+json`.
- `rg -n "postgres:16" .github/workflows` -- passed; exactly two workflow
  service-image refs, both tag+digest pinned.
- `python scripts/audit_workflow_security_posture.py .github/workflows` --
  passed; no service-image digest warnings remain.
- YAML parse smoke for `.github/workflows` -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_b2b_campaign_migration_checks.yml` | 2 |
| `.github/workflows/atlas_deflection_migration_apply_checks.yml` | 2 |
| `HARDENING.md` | 4 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Workflow-Service-Image-Digests.md` | 129 |
| `plans/archive/PR-Workflow-Setup-Python-Pin.md` | 0 |
| **Total** | **140** |
