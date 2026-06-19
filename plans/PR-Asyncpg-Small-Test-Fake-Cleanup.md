# PR: Asyncpg small test fake cleanup

## Why this slice exists

#1713 neutralized suite-wide `asyncpg` MagicMock poisoning at the pytest harness boundary. #1712 then merged the advisory repo-wide backstop and auditor credit that lets touched unit tests rely on the backstop instead of bespoke per-file workflow enrollment. This slice removes the easiest remaining `asyncpg` import-time fakes from small test files without widening into the largest historical mock lists.

## Scope (this PR)

Ownership lane: testing/backstop-hygiene
Slice phase: Robust testing

1. Remove `asyncpg` and `asyncpg.exceptions` from small MCP test files that only need unrelated MCP fakes.
2. Remove `asyncpg` from small scraper/parser optional dependency fake lists.
3. Leave large broad-mock files for a later chunk to keep review size bounded.
4. Add this plan doc for the local PR review contract.

### Files touched

- `plans/PR-Asyncpg-Small-Test-Fake-Cleanup.md`
- `tests/test_b2b_evidence_mcp.py`
- `tests/test_b2b_products_mcp.py`
- `tests/test_b2b_scrape_targets_mcp_inputs.py`
- `tests/test_b2b_signals_mcp_inputs.py`
- `tests/test_b2b_vendor_registry_mcp.py`
- `tests/test_trustpilot_parser.py`
- `tests/test_twitter_parser.py`

### Review Contract

Acceptance criteria:

- [ ] No touched test registers `asyncpg` or `asyncpg.exceptions` as a broad `MagicMock` at import time.
- [ ] Existing local MCP/service fakes remain in place where the tests need them.
- [ ] No production code changes.
- [ ] No backstop workflow or pytest marker boundary changes.

Affected surfaces: test import isolation only.

Risk areas: import errors if a touched test secretly depended on the `asyncpg` fake. Mitigated by limiting changes to tests whose pools are local `MagicMock`/`AsyncMock` objects or parser tests that do not use `asyncpg`.

Reviewer rules triggered: R1.

## Mechanism

The touched files used broad `sys.modules.setdefault(..., MagicMock())` lists to fake optional dependencies during import. Removing `asyncpg` from those lists ensures they no longer plant a process-wide fake driver. Tests that need MCP shims, parser optional dependencies, or local mocked pools keep those fakes unchanged.

## Intentional

- Keep the cleanup physical and local now that #1712 credits unit tests to the repo-wide backstop.
- Avoid changing the #1713 harness behavior in the same slice.
- Defer the largest broad-mock files so this PR stays small and easy to review.

## Deferred

- Larger files such as `tests/test_b2b_churn_mcp.py` and `tests/test_structured_review_parser_supplements.py` can be handled in a separate chunk.
- Residual true unit failures should wait until the backstop rerun shows the remaining profile after these fake-planter removals.
- Parked hardening: none.

## Verification

- Bundled runtime command passed: `python -m py_compile tests/test_b2b_churn_mcp.py tests/test_b2b_signals_mcp_inputs.py tests/test_b2b_scrape_targets_mcp_inputs.py tests/test_b2b_products_mcp.py tests/test_b2b_evidence_mcp.py tests/test_b2b_vendor_registry_mcp.py tests/test_structured_review_parser_supplements.py tests/test_twitter_parser.py tests/test_trustpilot_parser.py`.
- CI Pre-push Audit passed on this branch after #1712 merged the backstop-aware enrollment auditor.
- CI PR Body Contract passed after the PR body gained the required Parked hardening section.
- CI AI Reconciliation passed.
- CI Maturity Sweep passed.

## Estimated diff size

| File group | LOC |
|---|---:|
| Plan doc | ~75 |
| Small test fake cleanup | ~12 deletions |
| **Total** | **~87 touched / 12 removed** |
