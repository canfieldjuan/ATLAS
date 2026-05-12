# PR-Audit-MCP-Port-Assignments

## Why this slice exists

The reviewer/auditor kit now checks plan shape, files touched, and diff size.
The next mechanical drift class from the oversized audit PRs is configuration
documentation: `CLAUDE.md` can claim MCP ports that do not match
`atlas_brain/config.py`.

## Scope (this PR)

Add `scripts/audit_mcp_port_assignments.py`, focused parser/classifier/CLI
tests, and the coordination row refresh.

### Files touched

- `scripts/audit_mcp_port_assignments.py`
- `tests/test_audit_mcp_port_assignments.py`
- `plans/PR-Audit-MCP-Port-Assignments.md`
- `docs/extraction/coordination/inflight.md`

## Mechanism

The auditor parses `MCPConfig` with `ast`, extracts each `<name>_port`
`Field(default=N)`, and compares those defaults with two `CLAUDE.md` claim
forms:

- `ATLAS_MCP_<NAME>_PORT=N` environment-variable examples.
- `# SSE HTTP mode (port N)` examples followed by
  `python -m atlas_brain.mcp.<name>_server --sse`.

It reports `OK`, `MISSING`, `DRIFT`, `CONFLICT`, or `EXTRA`.

## Intentional

- No changes to `CLAUDE.md` or `atlas_brain/config.py`; current docs and
  config already match.
- The script validates port claims only. Tool counts and tool names are
  separate auditors.
- No wrapper integration in this PR. The wrapper comes after individual
  auditors land.

## Deferred

Wrapper/CI integration, MCP enable/disable env auditing, tool inventory
auditing, and PR-description port-claim checks.

## Verification

```bash
python -m pytest tests/test_audit_mcp_port_assignments.py
python scripts/audit_mcp_port_assignments.py
python scripts/audit_plan_doc.py plans/PR-Audit-MCP-Port-Assignments.md
python scripts/audit_plan_doc_files_touched.py plans/PR-Audit-MCP-Port-Assignments.md origin/main
python scripts/audit_plan_doc_diff_size.py plans/PR-Audit-MCP-Port-Assignments.md origin/main
python -m py_compile scripts/audit_mcp_port_assignments.py tests/test_audit_mcp_port_assignments.py
git diff --check
```

## Estimated diff size

| File | LOC churn (approx) |
|---|---:|
| `scripts/audit_mcp_port_assignments.py` | 160 |
| `tests/test_audit_mcp_port_assignments.py` | 161 |
| `plans/PR-Audit-MCP-Port-Assignments.md` | 68 |
| `docs/extraction/coordination/inflight.md` | 4 |
| **Total** | **~393** |
