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

## Slice B -- Enroll the never-run invoicing MCP/OAuth tests (NOT a prod fix)

`test_invoicing_readonly_mcp`, `test_invoicing_readonly_oauth`,
`test_invoicing_draft_writer_mcp`, `test_invoicing_draft_writer_oauth` are
enrolled in **zero** workflows -- they never run in CI today. That part of
the finding stands.

**Correction (the earlier framing was wrong):** the production import is
NOT broken. `atlas_brain/mcp/invoicing_readonly_oauth.py:19` does
`from mcp.server.auth.provider import OAuthAuthorizationServerProvider, ...`,
which are real symbols in the official `mcp>=1.26.0` SDK. That module is
live -- it is imported by the live `invoicing_readonly_server.py:24` and is
the shipped OAuth connector pattern (Claude/Codex connect through it; see
`docs/MCP_CHATGPT_OAUTH_ROLLOUT_RUNBOOK.md`). It is neither dead code nor a
version mismatch. The `'mcp.server' is not a package` error seen in the
full-suite run was **not** a real import failure -- it was caused by sibling
tests poisoning `sys.modules` (see Slice C), plus this sandbox simply not
having `mcp` installed (`No module named 'mcp'`; CI installs it via
`requirements.txt`).

Fix: just enroll these four files in `atlas_invoicing_checks.yml`. Do not
touch production OAuth code and do not re-pin `mcp`. The enrollment must land
*together with* the Slice C isolation fix, or the leak will make them fail.
Risk: CI-config only. Size: small.

## Slice C -- Stop the global `mcp` MagicMock leak (the real root cause)  [DONE]

Landed via `tests/_mcp_stub.py` (`stub_mcp` context manager: save -> plant
fake -> import server -> restore `sys.modules`) applied to all eight files.
The `mcp` setdefault blocks are gone; each test plants its fake only for its
own server import, so nothing survives to poison sibling collection. Verified
the helper's save/restore semantics directly, including the exact
invoicing-poison scenario (a real `mcp.server.auth.provider` is no longer
shadowed after a stub block). Full collection is CI-verified (this sandbox
lacks `torch`/`mcp`).

Confirmed root cause along the way: the b2b tool submodules only ever call
`@mcp.tool()` (no kwargs) and `b2b/server.py` is the *sole* importer of
`mcp.server.fastmcp`. So the `tool() got unexpected keyword 'structured_output'`
/ missing `custom_route` errors were never "stale mocks" -- they were the
content-ops server inheriting the b2b passthrough fake because `setdefault`
let whichever module collected first win. Fixing isolation fixes both that and
the `'mcp.server' is not a package` failure; no fake signature changes needed.

This is the linchpin, not a side issue. Eight tests
(`test_b2b_churn_mcp`, `test_b2b_products_mcp`, `test_b2b_signals_mcp_inputs`,
`test_b2b_vendor_registry_mcp`, `test_b2b_scrape_targets_mcp_inputs`,
`test_b2b_evidence_mcp`, `test_b2b_source_impact`,
`test_mcp_content_ops_marketer_verify`) do, at **module top level with no
teardown**, e.g.:

```
sys.modules.setdefault("mcp", MagicMock())
sys.modules.setdefault("mcp.server", MagicMock())
sys.modules.setdefault("mcp.server.fastmcp", _fastmcp_mod)  # _fastmcp_mod.FastMCP = _MockFastMCP
```

`setdefault` plants the fake whenever `mcp` has not been imported *yet* --
true even when real `mcp` is installed, if no earlier test imported it. Once
planted, `mcp` is a MagicMock for the rest of the pytest session, so any
later test importing real `mcp.server.auth.provider` (the invoicing OAuth
tests) gets `'mcp.server' is not a package`. In their own single-file
workflows these b2b tests pass; under whole-suite collection they poison
everything downstream. This is exactly why the backstop choked on `mcp`.

Two distinct test-only problems to fix:
1. **Isolation/leak:** make the stub session-safe -- use a `conftest`
   fixture with proper teardown, or only stub when real `mcp` is genuinely
   absent and restore `sys.modules` after, so real-`mcp` tests are unaffected.
2. **Stale fakes:** the hand-rolled `_MockFastMCP` has drifted behind the
   real FastMCP API (`tool(... structured_output=...)`, `custom_route`).
   Where the stub exists only to dodge a heavy import, prefer importing real
   `mcp` (it is in `requirements.txt`) over maintaining a divergent fake.

Risk: test-only. Size: small-medium.

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
and ship while never running. Slice B enrolls them (production is fine -- the
OAuth server is live via Claude/Codex); the backstop (slice E) is the standing
guard so it cannot recur silently.

Related but separate: broaden `audit_extracted_pipeline_ci_enrollment.py`
(make it workflow-aware) so un-enrolled files are flagged at PR time, not only
caught by the nightly backstop.

Deferred (same bug class as Slice C, not the backstop blocker): the same eight
tests still stub heavy deps (`torch`, `asyncpg`, `numpy`, ...) into
`sys.modules` with `setdefault` and no teardown. Those fakes can leak across
modules too, but they did not cause the invoicing collection failure and a
fake `torch` is mostly inert in the unit backstop. Fold them into the same
save/restore window (or a shared autouse fixture) in a follow-up once Slice E
confirms which, if any, still bite.
