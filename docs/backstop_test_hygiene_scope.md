# Scope: Test-Hygiene Cleanup to Re-enable the Repo-Wide Backstop

Goal: make the unit suite runnable as a whole so the held repo-wide backstop
(`claude/pr-ci-repo-wide-unit-backstop`, closed PR #1707) can return and go
green. The backstop excludes `integration`/`e2e` markers and runs everything
else; its first run could not run the suite because some tests cannot be
collected and others silently need services. This breaks that into ordered,
independent slices. Counts are from the actual #1707 backstop run + local audit.

## What the backstop surfaced (root causes, not "missing dependency")

`mcp>=1.26.0` IS in `requirements.txt` (installed in CI). The failures are
labeling, a real production API mismatch, and stale test mocks -- not an absent
dep.

---

## Slice A -- Mark mislabeled DB-backed tests as `integration`  [DONE]

Tests that use the `db_pool` fixture (a real Postgres pool) but carry no
`integration` marker get run by `not integration and not e2e` against a
database that isn't there. They are integration tests; mark them so.

Auditing the original candidate list corrected it (verify-before-marking
paid off):

- Already correctly marked (class-level `@pytest.mark.integration`), no
  change needed: `test_session_id_normalization`, `test_default_task_seeding`,
  `test_voice_session_reuse`, `test_entity_context`.
- Wholly DB-bound -> module-level `pytestmark = pytest.mark.integration`:
  `test_b2b_enrichment_batch_integration`,
  `test_b2b_enrichment_repair_batch_integration`,
  `test_campaign_recipient_dedup`.
- **Mixed** file -> per-test markers on the five `db_pool` tests only,
  leaving the pure-function tests (which use an in-test `_NullPool` stub) in
  the backstop: `test_evidence_gate`.

Risk: low/mechanical. Size: small. Landed: module markers on 3 files,
per-test markers on 5 tests in `test_evidence_gate`; the 6 pure-unit tests
there stay in the backstop.

## Slice B -- Fix the never-run invoicing MCP/OAuth tests (highest value)

`test_invoicing_readonly_mcp`, `test_invoicing_readonly_oauth`,
`test_invoicing_draft_writer_mcp`, `test_invoicing_draft_writer_oauth` are
enrolled in **zero** workflows -- they never run in CI today. They fail to
import because production code
(`atlas_brain/mcp/invoicing_readonly_oauth.py:19`) does
`from mcp.server.auth.provider import ...`, but the installed `mcp>=1.26.0`
has no `mcp.server.auth` (`'mcp.server' is not a package`). This is a real,
untested production import.

Fix: determine which `mcp` version exposes `mcp.server.auth` (or what it was
renamed to) and either pin/upgrade `mcp` or update the import to the current
API; then enroll these tests in `atlas_invoicing_checks.yml` so they actually
run. Risk: touches production OAuth server code + a dependency pin. Size:
medium (the meatiest slice; do not bundle with the others).

## Slice C -- Fix stale/leaky MCP test mocks (content-ops MCP tests)

`test_mcp_content_ops_deflection_readonly` (enrolled in 1 workflow) and
`test_mcp_content_ops_marketer_verify` (2) DO run today, yet failed in the
full-suite collection with `_MockFastMCP.tool() got an unexpected keyword
argument 'structured_output'` and `'_MockFastMCP' object has no attribute
'custom_route'`. The `_MockFastMCP` stand-ins are defined inside the test files
and have drifted behind the server code.

Investigate first: do they pass in isolation (their own workflow) but break
under whole-suite collection because a mock / `sys.modules` patch from another
`*_mcp.py` test leaks across files? If so the fix is isolation; if the mock is
simply stale, update it to match the FastMCP API (`structured_output`,
`custom_route`). Risk: test-only. Size: small-medium.

## Slice D -- Categorize and handle the remaining collection errors

Per-file, ~5 files:
- `tests/test_cloud_latency.py` -- imports `openai`; it is an external-API
  latency test. Mark `e2e` (or skip-if-no-key). 
- `tests/test_graphiti_wrapper_health.py` -- exercises the Graphiti service;
  mark `e2e`.
- `tests/test_atlas_content_ops_input_provider.py` -- `asyncpg.__spec__ is not
  set` (an import/mocking quirk); fix the import or guard.
- `tests/test_competitive_intelligence.py`, `tests/test_b2b_phase4_causality_gate.py`
  -- need a per-file look (likely import-time service/dep dependence).

Risk: low per file. Size: small.

## Slice E -- Reintroduce the backstop

Restore the backstop workflow + plan from
`claude/pr-ci-repo-wide-unit-backstop` (branched fresh off `main`), confirm the
suite now collects and runs with only intended `integration`/`e2e` skips, and
decide advisory vs. gating. Keep `--continue-on-collection-errors` as a
belt-and-suspenders. Size: small (the workflow already exists). Gated on A-D.

---

## Sequencing

- A, C, D are independent test-file changes -- parallelizable.
- B is the substantive one (production + dependency); isolate it.
- E is last, gated on A-D landing.

## Meta note

That four invoicing MCP/OAuth test files are enrolled in zero workflows is
itself the exact problem the backstop exists to catch: tests that pass review
and ship while never running. Slice B both fixes them and enrolls them; the
backstop (slice E) is the standing guard so it cannot recur silently.

Related but separate: broaden `audit_extracted_pipeline_ci_enrollment.py`
(make it workflow-aware) so un-enrolled files are flagged at PR time, not only
caught by the nightly backstop.
