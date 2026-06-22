# PR-Deflection-Delta-Read-Surface

## Why this slice exists

#1316's next delta-report step is a paid-gated read surface for the persisted
`deflection_delta.v1` artifact. #1771 added the pure comparator and #1795 added
tenant-scoped persistence, but the stored delta is still only reachable from
tests/internal store methods. A monthly subscription cannot use it until the
existing report read surfaces can fetch it safely.

Root cause: persisted deflection deltas exist behind the
`DeflectionReportArtifactStore` boundary, but there is no shared paid/read
constructor that gates source reports and returns a customer-facing delta
payload. This PR fixes that root for read-only HTTP and MCP surfaces. Monthly
automation, delivery email, and result-page rendering remain separate slices.

Diff-size note: this exceeds the 400 LOC target because the smallest useful read
surface needs the shared paid gate, HTTP route, MCP tool, allowlist projection,
MCP inventory docs/fixtures, and source-report relock tests in one PR. Splitting
the docs/tests from the tool would leave the new surface under-audited.

## Scope (this PR)

Ownership lane: issue-1316/deflection-delta-read-surface
Slice phase: Vertical slice

1. Add a shared access-layer helper for fetching a persisted delta by current
   report and optional baseline report while requiring both source reports to
   remain paid for the same tenant.
2. Add a paid-gated HTTP read endpoint under the existing deflection report
   control surface.
3. Add a read-only MCP tool that returns the same allowlisted delta payload for
   the bound tenant.
4. Update MCP tool-count/docs fixtures because the read-only server grows from
   two tools to three.
5. Add focused tests for tenant scoping, unpaid/relocked source reports,
   missing baselines/deltas, raw-artifact exclusion, and MCP no-account
   fail-closed behavior.

### Review Contract

Acceptance criteria:
- The read surface returns only a persisted `deflection_delta.v1` payload and
  pair metadata; it does not compute deltas on demand and does not expose raw
  report artifacts, markdown, evidence export rows, source IDs, or ticket text.
- Both current and baseline reports must exist, belong to the resolved tenant,
  and still be paid at read time; relocking/refunding either source report makes
  the delta unavailable.
- When no baseline is supplied, the read surface uses the existing
  `select_previous_paid_report(...)` store boundary rather than guessing across
  tenants or unpaid reports.
- When a baseline is supplied, the read surface verifies that exact report pair
  is still paid and tenant-scoped before reading the stored delta.
- The MCP tool fails closed before touching storage when no tenant binding is
  configured, and the server remains read-only with no generation/unlock
  mutation tools.
- MCP docs, audited tool counts, and fixtures are updated in the same PR as the
  new tool.

Affected surfaces:
- `extracted_content_pipeline/deflection_report_access.py`
- `extracted_content_pipeline/api/control_surfaces.py`
- `atlas_brain/mcp/content_ops_deflection_readonly_server.py`
- `tests/test_content_ops_deflection_delta_persistence.py`
- `tests/test_extracted_content_deflection_submit.py`
- `tests/test_mcp_content_ops_deflection_readonly.py`
- MCP docs/tool-count audit fixtures

Risk areas:
- Treating a previously persisted delta as paid forever after source reports are
  relocked.
- Returning raw stored report payloads through a buyer-facing read endpoint.
- Diverging HTTP and MCP behavior by implementing separate payload rules.
- Forgetting MCP tool inventory docs, which makes local review fail late.

Reviewer rules triggered: R1, R2, R3, R4, R5, R8, R10, R13, R14.

### Files touched

- `CLAUDE.md`
- `atlas_brain/mcp/content_ops_deflection_readonly_server.py`
- `extracted_content_pipeline/api/control_surfaces.py`
- `extracted_content_pipeline/deflection_report_access.py`
- `plans/PR-Deflection-Delta-Read-Surface.md`
- `tests/test_audit_mcp_tool_names_match_docs.py`
- `tests/test_content_ops_deflection_delta_persistence.py`
- `tests/test_extracted_content_deflection_submit.py`
- `tests/test_mcp_content_ops_deflection_readonly.py`
- `tests/test_pre_push_audit.py`

## Mechanism

The access helper resolves the current report through
`DeflectionReportArtifactStore.get_artifact_record(...)` and requires `paid` to
still be true. If the caller supplies `baseline_request_id`, the helper loads
that exact baseline report through the same tenant-scoped store boundary and
requires it to still be paid. If the caller omits a baseline, the helper uses
`select_previous_paid_report(...)`, which already anchors selection to the paid
current report's `created_at`.

Only after both source reports pass that paid/tenant gate does the helper call
`get_deflection_delta(...)` for the exact `(account_id, current_request_id,
baseline_request_id)` pair. A small payload constructor then allowlists the
pair IDs, timestamps, and known `deflection_delta.v1` top-level/item/CSAT fields.
HTTP and MCP both call that shared path, so one contract controls both read
surfaces.

## Intentional

- No on-demand compute from the read endpoint. If the delta was not generated
  and stored by #1795's helper, the read surface returns unavailable instead of
  silently doing work on a GET/MCP read path.
- No free snapshot delta surface. Deltas compare paid full report models and
  stay behind paid/source-report gates.
- No delivery email, result-page UI, autonomous monthly job, calendar-window
  baseline resolver, or explicit recompute endpoint in this slice.
- The MCP tool is added to the existing read-only server instead of creating a
  new MCP server because it shares the same tenant binding, auth mode, and
  deflection report store boundary.

## Deferred

- #1316 monthly automation and delivery/upsell email.
- Result-page rendering for paid delta reports.
- Calendar/source-window baseline resolver and explicit recompute/override
  workflow.
- Multi-version recompute audit history for the same current/baseline pair.

Parked hardening: none.

## Verification

- Focused read-surface pytest -- 33 passed.
- MCP tool-name docs audit -- passed.
- MCP tool-count docs audit -- passed.
- Python compile for touched Python files -- passed.
- Extracted content pipeline validation -- passed.
- Extracted reasoning-import guard -- clean.
- Extracted standalone audit -- Atlas runtime import findings: 0.
- ASCII Python policy check -- passed.
- Full extracted pipeline bundle -- reasoning core 295 passed; extracted
  content 4882 passed, 15 skipped, 1 existing torch warning.
- Pending before push: local PR review.

## Estimated diff size

| File | LOC |
|---|---:|
| `CLAUDE.md` | 11 |
| `atlas_brain/mcp/content_ops_deflection_readonly_server.py` | 67 |
| `extracted_content_pipeline/api/control_surfaces.py` | 31 |
| `extracted_content_pipeline/deflection_report_access.py` | 210 |
| `plans/PR-Deflection-Delta-Read-Surface.md` | 159 |
| `tests/test_audit_mcp_tool_names_match_docs.py` | 10 |
| `tests/test_content_ops_deflection_delta_persistence.py` | 96 |
| `tests/test_extracted_content_deflection_submit.py` | 118 |
| `tests/test_mcp_content_ops_deflection_readonly.py` | 115 |
| `tests/test_pre_push_audit.py` | 6 |
| **Total** | **823** |
