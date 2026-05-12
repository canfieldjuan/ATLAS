# PR-Audit-MCP-Tool-Names

## Why this slice exists

Oversized PR #484 bundled three Tier-1 auditors and was rejected as
unreviewable. This split lands only the MCP tool-name inventory auditor:
it catches drift where `CLAUDE.md` claims a server exposes a tool list
that no longer matches the real `@mcp.tool` functions.

## Scope (this PR)

1. Add `scripts/audit_mcp_tool_names_match_docs.py`.
2. Add fixture tests for known and unknown MCP server header parsing.
3. Refresh the in-flight coordination row for this split.

### Files touched

- `plans/PR-Audit-MCP-Tool-Names.md`
- `scripts/audit_mcp_tool_names_match_docs.py`
- `tests/test_audit_mcp_tool_names_match_docs.py`
- `docs/extraction/coordination/inflight.md`

## Mechanism

`doc_claims()` scans `CLAUDE.md` for `### <Name> MCP Server` sections,
collects backticked snake_case identifiers from each known section, and
returns both known claims and unknown server headings. Unknown headings
are reported as drift instead of being silently skipped.

`tool_names_in_file()` parses each MCP server with `ast` and collects
function names decorated with `@mcp.tool` or `@mcp.tool(...)`.
`actual_for()` handles the B2B server as a summed directory because its
tools are split across `atlas_brain/mcp/b2b/*.py`.

## Intentional

- The auditor is not wired into `scripts/pre_push_audit.sh` yet because
  current `CLAUDE.md` has known tool-inventory drift. Wrapper integration
  waits until the docs are reconciled or the drift is accepted.
- The extraction uses section-wide backticked snake_case identifiers,
  not a brittle single-line `Tools:` parser, because the docs use multiple
  continuation lines.
- This PR does not touch scraping/review-source count auditing.

## Deferred

- Add the auditor to the pre-push wrapper after the documented MCP tool
  inventories are cleaned up.
- Split or drop the remaining #484 review-source count auditor separately.
- Close or replace oversized #484 after its useful pieces are harvested.

## Verification

```bash
python -m pytest tests/test_audit_mcp_tool_names_match_docs.py
python -m py_compile scripts/audit_mcp_tool_names_match_docs.py tests/test_audit_mcp_tool_names_match_docs.py
python scripts/audit_plan_doc.py plans/PR-Audit-MCP-Tool-Names.md
python scripts/audit_plan_doc_files_touched.py plans/PR-Audit-MCP-Tool-Names.md origin/main
python scripts/audit_plan_doc_diff_size.py plans/PR-Audit-MCP-Tool-Names.md origin/main
git diff --check
```

`python scripts/audit_mcp_tool_names_match_docs.py` is expected to
return non-zero on current main because #484 already surfaced preexisting
`CLAUDE.md` tool-inventory drift.

## Estimated diff size

| File | LOC churn (approx) |
|---|---:|
| `scripts/audit_mcp_tool_names_match_docs.py` | 129 |
| `tests/test_audit_mcp_tool_names_match_docs.py` | 61 |
| `plans/PR-Audit-MCP-Tool-Names.md` | 75 |
| `docs/extraction/coordination/inflight.md` | 4 |
| **Total** | **~269** |
