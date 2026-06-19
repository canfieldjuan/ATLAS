# PR-Invoicing-MCP-OAuth-Test-Enrollment

## Why this slice exists

Four invoicing test files -- `tests/test_invoicing_readonly_mcp.py`,
`tests/test_invoicing_readonly_oauth.py`,
`tests/test_invoicing_draft_writer_mcp.py`, and
`tests/test_invoicing_draft_writer_oauth.py` -- are enrolled in **zero**
workflows. They never run in CI, so the read-only/draft-writer invoicing MCP
servers and their OAuth providers (which expose customer financial data) ship
with no automated test execution. That is a silent-coverage hole: a regression
in `invoicing_readonly_oauth.py` or `invoicing_draft_writer_oauth.py` would not
be caught by any check.

This slice enrolls those four files in the existing `.github/workflows/atlas_invoicing_checks.yml`
workflow so they run on changes to the invoicing surface.

## Scope (this PR)

Ownership lane: ci/coverage
Slice phase: Production hardening

1. Add the four invoicing MCP/OAuth test files to `.github/workflows/atlas_invoicing_checks.yml`
   as a dedicated `Run invoicing MCP + OAuth surface tests` step.
2. Add `pull_request` and `push` path triggers for the production modules they
   exercise (`invoicing_readonly_server.py`, `invoicing_draft_writer_server.py`,
   `invoicing_readonly_oauth.py`, `invoicing_draft_writer_oauth.py`,
   `atlas_brain/mcp/auth.py`, `atlas_brain/config_defaults.py`) and for the
   four test files themselves.

### Files touched

- `.github/workflows/atlas_invoicing_checks.yml`
- `plans/PR-Invoicing-MCP-OAuth-Test-Enrollment.md`

### Review Contract

Acceptance criteria:

- [ ] The four test files run in CI via `.github/workflows/atlas_invoicing_checks.yml`.
- [ ] Path triggers cover the production OAuth/server modules they exercise.
- [ ] No production code changes; CI config only.

Affected surfaces: CI only.

Risk areas: none beyond surfacing any pre-existing failure in the four
previously-unrun tests (the intended signal).

Reviewer rules triggered: R1, R12.

## Mechanism

The four files are unit tests (no `integration`/`e2e` markers, no `db_pool`).
The two `_oauth` files import `mcp.server.auth.provider` directly; that import
resolves because the workflow installs `requirements.txt` (`mcp>=1.26.0`) and,
running in a dedicated invoicing-only job, no sibling test stubs `mcp` into
`sys.modules` ahead of them. No production OAuth code is touched and `mcp` is
not re-pinned -- the import was never broken, only unrun.

## Intentional

- Enrollment only; the test bodies and production code are untouched.
- Added to the existing invoicing workflow rather than a new one, since they
  exercise the invoicing surface that workflow already guards.

## Deferred

- Broaden `audit_extracted_pipeline_ci_enrollment.py` (make it workflow-aware)
  so un-enrolled test files are flagged at PR time, not discovered ad hoc.
- A repo-wide unit backstop (separate slice) as the standing catch-all for
  un-enrolled files.

Parked hardening: none.

## Verification

- `python -c "import yaml; yaml.safe_load(open('.github/workflows/atlas_invoicing_checks.yml'))"`
  -- valid.
- The four test files run on this PR via the new path trigger; their CI result
  is the live proof of enrollment.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_invoicing_checks.yml` | ~30 |
| `plans/PR-Invoicing-MCP-OAuth-Test-Enrollment.md` | ~75 |
| **Total** | **~105** |
