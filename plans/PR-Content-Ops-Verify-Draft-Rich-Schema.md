# PR-Content-Ops-Verify-Draft-Rich-Schema

## Why this slice exists

#1433 closed the ChatGPT adapter contract gap. The Claude-rich `verify_draft`
tool still exposed only eight shallow parameters, hiding nested payload shapes.

This slice closes that presentation gap without changing verifier behavior,
tenant binding, OAuth, or decoded-input tolerance.

The diff is slightly above 400 LOC because the review fix added required
contract-to-backend regression tests for both schema P2s and a Python 3.10
annotation-wrapper assertion fix.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Product polish

1. Add nested schema hints and descriptions to existing `verify_draft`
   parameters.
2. Keep decoded-input tolerance: malformed values still reach existing
   normalizers and fail closed rather than raising at schema validation.
3. Prove the schema hints match backend-accepted payload shapes.
4. Leave the ChatGPT `search` and `fetch` adapter contract unchanged.

### Files touched

- `plans/PR-Content-Ops-Verify-Draft-Rich-Schema.md`
- `atlas_brain/mcp/content_ops_marketer_verify_server.py`
- `tests/test_mcp_content_ops_marketer_verify.py`

### Review Contract

- Acceptance criteria: rich schema hints exist, backend field names match,
  malformed inputs fail closed, and the ChatGPT adapter is unchanged.
- Affected surfaces: Content Ops marketer Claude-rich MCP tool schema.
- Risk areas: MCP schema compatibility, usability, decoded-input tolerance.
- Reviewer rules triggered: R1, R2, R5

## Mechanism

Use Pydantic field metadata on `Any`-typed tool parameters. FastMCP publishes
the richer JSON-schema descriptions, while Pydantic still accepts decoded values
of any type and the existing normalizers decide how to fail closed.

The metadata mirrors backend field names for rule packets, coverage rows,
claims, quality reports, brand voice, comments, and the ISO review date.

## Intentional

- This is schema presentation only: no new fields, renames, or verdict changes.
- Parameters remain `Any`-typed at validation time so decoded non-object inputs
  still flow to the existing fail-closed normalizers.
- This does not modify the ChatGPT adapter contract from #1433.
- This does not start #1435 reliability-gate work.
- AI reconciliation: rebase fixed stale branches; schema fixes resolved both P2s;
  CI fix made annotation assertions portable across Python versions.

## Deferred

- #1435 reliability-gate work remains parked until labeled triples exist.
- Explicit adapter mode discriminators remain deferred unless live usage proves
  the #1433 contract example is still confusing.
- Parked hardening: none.

## Verification

- Passed: focused MCP tests for `tests/test_mcp_content_ops_marketer_verify.py`
  (38 passed after review fix).
- Passed: real FastMCP schema smoke for `verify_draft` nested parameter hints.
- Passed: dedicated Content Ops MCP workflow pytest sweep (96 passed after
  review fix).
- Passed: extracted pipeline wrapper `scripts/run_extracted_pipeline_checks.sh`
  (3569 passed, 10 skipped after review fix).
- Passed: py_compile for `atlas_brain/mcp/content_ops_marketer_verify_server.py`.
- Passed: `git diff --check`.
- Pending: body-aware `scripts/local_pr_review.sh`.

## Estimated diff size

| Area | LOC |
| --- | ---: |
| Total | 413 |
