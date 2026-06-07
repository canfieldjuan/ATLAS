# PR - Content-Ops Review Service Wiring

## Why this slice exists

`docs/content_ops_operating_model.md` and issue #1338 say slices 1-4 landed the
deterministic review-contract core, but the engine is still an island: the
Content-PR verdict, claims map, and coverage matrix have no non-test caller.
Issue #1353 narrows the delivery direction: the marketer MCP is future work,
but the wiring phase must be built tenant-scoped and tool-shaped from day one so
that the eventual MCP server is a thin transport wrapper.

This slice starts the wiring phase instead of slice 5. Slice 5's calibration
library and adversarial pass are useful, but optional for the verify-only v1;
the blocking product gap is that a marketer's structured draft cannot yet be
verified through a host service.

The diff is over the 400 LOC soft cap because the tests prove each required
failure branch of the service boundary: missing tenant scope, missing coverage,
unresolved coverage, failed coverage, decoded string status, mismatched claims,
expired claims, and blocking comments. The production service stays under 200
LOC; the overage is the failure-detection proof AGENTS.md requires for gates
plus the dedicated Atlas workflow enrollment required for a new test importing
the host package.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Vertical slice

Add the first host-layer callable review workflow:

1. Add a small Atlas service module that accepts a tenant scope, a structured
   review request, and an injected tenant registry reader.
2. Build a tenant-specific claims map from caller-provided structured claims and
   registry entries, then delegate the final decision to the existing
   deterministic Content-PR verdict engine.
3. Return a tool-shaped result containing the decision, reasons, mapped claims,
   and Content-PR envelope so a future MCP tool can wrap it without embedding
   review logic.
4. Prove tenant isolation and fail-closed behavior with focused tests.

### Review Contract

- Acceptance criteria:
  - [ ] The service requires a `TenantScope` with a non-empty account id before
        reading any registry data.
  - [ ] The registry reader receives the tenant scope, not a free-form account
        id from the request payload.
  - [ ] Caller-provided structured claims are mapped against the tenant
        registry before the verdict is computed.
  - [ ] Missing scope, missing coverage, unresolved coverage, failed coverage,
        mismatched claims, expired claims, and blocking comments produce the
        expected non-approval decisions.
  - [ ] Decoded/string enum values continue to block by equality, not identity.
  - [ ] No LLM, DB, FastAPI, or MCP transport logic is introduced in this
        slice.
- Affected surfaces: service, auth/tenant scope, CI enrollment.
- Risk areas: tenant isolation, backcompat, decoded input robustness,
  maintainability.
- Reviewer rules triggered: R1, R2, R3, R5, R10, R12.

### Files touched

- `atlas_brain/_content_ops_review_workflow.py`
- `.github/workflows/atlas_content_ops_review_workflow_checks.yml`
- `tests/test_atlas_content_ops_review_workflow.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `plans/PR-Content-Ops-Review-Service-Wiring.md`

## Mechanism

The service owns only orchestration. It takes a typed request containing rule
packet versions, coverage rows, extracted claims, comments, and an evaluation
date. It asks an injected registry reader for that tenant's approved claims,
uses the existing claims-map module to classify the draft claims, then builds a
Content-PR and delegates the decision to the existing verdict engine.

The tenant registry reader is a narrow port in this host service, not a
database implementation. That gives the later Postgres registry slice a stable
boundary to implement and gives the later MCP server a single callable service
to wrap.

## Intentional

- This is wiring, not slice 5: calibration examples, adversarial pass models,
  and model-disagreement orchestration stay out.
- This does not build the marketer MCP server. Transport comes after the
  service, registry persistence, tenant-binding bridge, and verdict/status
  mapping are proven.
- This does not add Postgres claim-registry persistence yet. The service is
  tenant-scoped and registry-shaped now; the next slice plugs a tenant
  Postgres repository into the reader port.
- Claims extraction from prose remains caller-owned for v1. The marketer's
  model must provide structured claims, matching issue #1353's "verify, don't
  generate" decision.
- The existing generated-assets lifecycle status strings are not changed here.

## Deferred

- `PR-Content-Ops-Claim-Registry-Persistence`: tenant-scoped claim/messaging
  registry table plus CRUD/list repository that implements this slice's
  registry-reader boundary.
- `PR-Content-Ops-Review-Status-Mapping`: map review decisions onto the
  generated-asset lifecycle vocabulary.
- `PR-Content-Ops-Quality-Gate-Coverage-Rows`: map deterministic
  `extracted_quality_gate` findings and brand-voice banned-term checks into
  coverage rows.
- `PR-Marketer-Verification-MCP`: write-capable OAuth MCP connector and tool
  surface after the service and registry are wired.
- Parked hardening: none expected unless implementation surfaces non-blocking
  tenant-binding or observability gaps.

## Verification

- Focused pytest command for the review workflow test -- 10 passed.
- Extracted pipeline CI enrollment audit command -- 154 matching tests are
  enrolled.
- ASCII Python policy command -- passed.
- Local PR review command with the PR body file -- to run before push.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_content_ops_review_workflow_checks.yml` | 46 |
| `atlas_brain/_content_ops_review_workflow.py` | 189 |
| `tests/test_atlas_content_ops_review_workflow.py` | 253 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `plans/PR-Content-Ops-Review-Service-Wiring.md` | 130 |
| **Total** | **619** |
