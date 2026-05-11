# PR-Audit-Claude-Md-MCP-Counts

## Why this slice exists

The oversized audit-kit PR #483 bundled three scripts and a long plan.
Review rejected it under the repo's ~400 LOC review-size gate. This
split lands the first useful piece only: a mechanical check that
compares MCP tool-count claims in `CLAUDE.md` to the actual
`@mcp.tool` decorators in the repo.

This catches the exact drift class that caused repeated review
round-trips: a documented MCP server count says one number while the
code exports another.

## Scope (this PR)

1. Add `scripts/audit_claude_md_claims.py`.
2. Add focused fixture tests for its parser edge cases.
3. Refresh the coordination row for this split slice.

### Files touched

- `scripts/audit_claude_md_claims.py`
- `tests/test_audit_claude_md_claims.py`
- `plans/PR-Audit-Claude-Md-MCP-Counts.md`
- `docs/extraction/coordination/inflight.md`

## Mechanism

`audit_claude_md_claims.py` parses `### <Name> MCP Server (N tools)`
headings in `CLAUDE.md`, maps each known server name to its source
file, counts exact `@mcp.tool` / `@mcp.tool(...)` decorator lines, and
reports `OK` or `DRIFT`. It also reports malformed MCP headings and
missing expected server headings instead of silently skipping them.

The B2B Churn server is split across `atlas_brain/mcp/b2b/*.py`, so
the auditor sums that directory through a sentinel mapping. Unknown
MCP server headings are reported as `UNKNOWN` instead of silently
skipped.

## Intentional

- No wrapper script in this PR. The pre-push wrapper is a later slice
  after the individual auditors land.
- No plan-doc-shape auditor in this PR. That becomes the next split
  from #483.
- Soft counts like `60+` intentionally report drift because the code
  can provide an exact count.
- Tests import the script directly through `importlib.util` because
  `scripts/` is not a package.

## Deferred

- `scripts/audit_plan_doc.py` split from #483.
- `scripts/pre_push_audit.sh` wrapper after the individual auditors
  exist on main.
- Broader Tier-1/Tier-2 audit scripts from #484 and #485.
- CI wiring.

## Verification

```bash
python -m pytest tests/test_audit_claude_md_claims.py
python -m py_compile scripts/audit_claude_md_claims.py tests/test_audit_claude_md_claims.py
python scripts/audit_claude_md_claims.py
git diff --check
```

## Estimated diff size

| File | LOC (approx) |
|---|---:|
| `scripts/audit_claude_md_claims.py` | 100 |
| `tests/test_audit_claude_md_claims.py` | 105 |
| `plans/PR-Audit-Claude-Md-MCP-Counts.md` | 70 |
| `docs/extraction/coordination/inflight.md` | 2 |
| **Total** | **~285** |
