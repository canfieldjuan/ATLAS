# PR: Atlas Support Ticket Provider API Route Proof

## Why this slice exists

PR-Content-Ops-Host-Ticket-Input-Provider mounted the Atlas support-ticket input
provider into the real Content Ops API router, and
PR-Support-Ticket-Input-Provider-Route-Parity proved the extracted router seam
works across preview, plan, and execute. The remaining narrow gap is host route
proof: the actual `atlas_brain.api` mounted `/content-ops` routes should apply
the provider when support-ticket source material is sent through the Atlas API
surface.

This slice adds host-level route tests only. It does not add new behavior.

## Scope (this PR)

Ownership lane: content-ops/input-provider-ticket-package

Slice phase: Functional validation

1. Add a host-mounted `/content-ops/preview` test proving inline support-ticket
   source material is expanded into FAQ Report defaults by the Atlas route.
2. Add a host-mounted `/content-ops/plan` test proving the same provider-built
   package produces runnable `faq_markdown`, `landing_page`, and `blog_post`
   plan steps.
3. Keep these tests behind the existing `asyncpg` import guard because
   importing `atlas_brain.api` pulls in the host database module.

### Files touched

- `tests/test_atlas_content_ops_input_provider.py`
- `plans/PR-Atlas-Support-Ticket-Provider-API-Route-Proof.md`

## Mechanism

The tests reuse the existing fresh API package helper to import
`atlas_brain.api` from a clean module slot, then pull the mounted
`/content-ops/preview` and `/content-ops/plan` endpoint functions from
`api.router.routes`. Calling those endpoints directly avoids network setup and
auth dependency execution while still exercising the host router mount and its
configured `input_provider`.

The route payload uses a realistic support-ticket row with `subject` and
`body`, which also protects the host classifier parity fixed in the prior PR.

## Intentional

- No new production code. This is a route-proof validation slice.
- No execute-route host test. Execute needs host execution services and DB/LLM
  wiring; the extracted route parity PR already covers deterministic execute
  behavior without model or database cost.
- The tests skip when `asyncpg` is not importable because the dependency-light
  extracted CI lane does not install the host database driver. Lower-level
  provider tests still run in that lane.

## Deferred

- Future PR owned by the ingestion lane: persisted upload/import source loading
  can be tested when that host lookup contract exists.
- Future PR owned by the FAQ session: standalone FAQ article route coverage.
- Parked hardening: none.

## Verification

- `py_compile` for `tests/test_atlas_content_ops_input_provider.py` - passed.
- `pytest tests/test_atlas_content_ops_input_provider.py -q` - 9 passed.
- `pytest tests/test_atlas_content_ops_input_provider.py tests/test_extracted_support_ticket_input_provider.py tests/test_extracted_support_ticket_input_package.py -q` - 35 passed.
- `scripts/local_pr_review.sh` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~70 |
| Host route tests | ~65 |
| **Total** | **~135** |
