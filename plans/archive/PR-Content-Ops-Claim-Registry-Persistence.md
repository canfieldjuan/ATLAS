# PR - Content-Ops Claim Registry Persistence

## Why this slice exists

Issue #1338 says the content-ops review-contract core has landed through the
first wiring slice, but issue #1353 still calls out the largest integration gap:
there is no tenant-owned claim and messaging registry store. PR #1363 created a
host review service with an injected tenant registry reader; this slice plugs a
real tenant-scoped Postgres-backed repository into that boundary so the review
workflow can verify marketer-provided claims against durable approved wording.

This is the next wiring slice rather than operating-model slice 5. Calibration
examples and adversarial review are useful, but the verify-only marketer v1
cannot be wired until approved claims can be persisted, listed, updated, and
expired per tenant.

The diff is expected to exceed the 400 LOC soft cap because this PR carries a
deployable migration, the repository adapter, focused fake-pool tests for each
tenant-scope mutation and failure branch, and a dedicated Atlas workflow so the
host-package tests run in CI.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Vertical slice

Add the first durable tenant claim and messaging registry behind the existing
review-service reader port:

1. Add a tenant-scoped Postgres migration for active registry claims with
   approved wording, risk tier, optional expiration date, metadata, timestamps,
   and a soft-archive column.
2. Add a small Atlas repository module that can create, update, list, expire,
   and archive active registry claim records for one account.
3. Expose a repository reader method that returns the existing
   `RegistryClaim` mapping shape used by the deterministic claims map.
4. Prove tenant isolation, decoded input handling, invalid-id fail-closed
   behavior, and service compatibility with focused tests.
5. Enroll the new tests in the extracted pipeline wrapper and a dedicated Atlas
   workflow.

### Review Contract

- Acceptance criteria:
  - [ ] The migration stores registry claims with `account_id` and active-row
        uniqueness scoped to one tenant.
  - [ ] Create, update, expire, archive, and list operations filter by
        `account_id` and never mutate a cross-tenant row.
  - [ ] The repository reader implements the review-service registry reader
        shape and returns claims keyed by registry id.
  - [ ] Missing or invalid tenant scope fails closed without a database read.
  - [ ] Decoded input treats missing or non-text optional fields as missing
        rather than raising.
  - [ ] Invalid risk-tier values fail before persistence.
  - [ ] No LLM, FastAPI, MCP transport, or generated-asset lifecycle mapping is
        introduced in this slice.
- Affected surfaces: database, service, auth/tenant scope, CI enrollment.
- Risk areas: tenant isolation, migration safety, decoded input robustness,
  concurrency, backcompat, maintainability.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R8, R10, R12.

### Files touched

- `atlas_brain/_content_ops_claim_registry.py`
- `atlas_brain/_content_ops_review_workflow.py`
- `atlas_brain/storage/migrations/334_content_ops_claim_registry.sql`
- `.github/workflows/atlas_content_ops_claim_registry_checks.yml`
- `tests/test_atlas_content_ops_review_workflow.py`
- `tests/test_content_ops_claim_registry.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `plans/PR-Content-Ops-Claim-Registry-Persistence.md`

## Mechanism

The migration creates one active registry row per tenant and registry id. Active
row uniqueness is enforced on normalized registry ids for each `account_id`
while archived rows remain available for audit history. Expiration is separate
from archiving: an expired claim remains readable so the claims map can emit an
expired status instead of silently forgetting the claim.

The repository follows the existing Content Ops persistence pattern: async pool
methods, UUID tenant ids, display-safe dataclasses, JSONB metadata, and tenant
scope in every mutation query. The reader method accepts a `TenantScope`,
validates the account id, reads only active rows for that tenant, and converts
them into the extracted package's `RegistryClaim` values without importing any
transport layer.

Post-review update: invalid or malformed tenant scope now raises a typed
registry-read error, and the review-service boundary converts that into a
blocked verdict instead of treating it as an empty registry. Registry ids are
canonicalized to lowercased, trimmed values on write and read so the Python
lookup key matches the database uniqueness contract.

## Intentional

- This does not build marketer MCP tools. The future MCP server should wrap the
  review service and this repository instead of owning business logic.
- This does not extract claims from prose. Issue #1353 keeps v1 verify-only, so
  callers still provide structured claims.
- Expiring a claim sets its expiration date but does not archive it. The review
  engine must still see expired entries so expired claims block deterministically.
- The migration introduces the table only; no backfill is needed because no
  prior durable claim registry exists.

## Deferred

- `PR-Content-Ops-Review-Status-Mapping`: map review decisions onto generated
  asset lifecycle statuses after the registry reader is real.
- `PR-Content-Ops-Quality-Gate-Coverage-Rows`: map deterministic quality-gate
  and brand-voice findings into Content-PR coverage rows.
- `PR-Marketer-Verification-MCP`: expose verify-only marketer tools after the
  service, registry, tenant-binding bridge, and status mapping are wired.
- `PR-Content-Ops-Calibration-Library`: deterministic calibration/adversarial
  examples from operating-model slice 5.
- Parked hardening: none expected unless implementation surfaces non-blocking
  registry observability or admin UI gaps.

## Verification

- Focused pytest command for the claim-registry persistence and review-service
  workflow tests -- 31 passed.
- Extracted pipeline CI enrollment audit command -- 155 matching tests are
  enrolled.
- Focused pytest command for the CI-enrollment audit regression tests -- 18
  passed.
- ASCII Python policy command -- passed.
- Local PR review command with a prepared PR body file -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_content_ops_claim_registry_checks.yml` | 52 |
| `atlas_brain/_content_ops_claim_registry.py` | 348 |
| `atlas_brain/_content_ops_review_workflow.py` | 9 |
| `atlas_brain/storage/migrations/334_content_ops_claim_registry.sql` | 36 |
| `tests/test_atlas_content_ops_review_workflow.py` | 31 |
| `tests/test_content_ops_claim_registry.py` | 431 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `plans/PR-Content-Ops-Claim-Registry-Persistence.md` | 141 |
| **Total** | **1050** |
