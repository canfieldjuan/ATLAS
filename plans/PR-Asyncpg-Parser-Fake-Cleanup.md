# PR: Asyncpg parser fake cleanup

## Why this slice exists

#1717 removed the easiest small `asyncpg` import-time fakes from MCP and parser tests. This follow-up removes the remaining parser-side broad fake while keeping the very large B2B churn MCP file for its own reviewable chunk.

## Scope (this PR)

Ownership lane: testing/backstop-hygiene
Slice phase: Robust testing

1. Remove `asyncpg` from the broad optional-dependency fake list in the structured review parser supplement tests.
2. Keep Playwright, scraper, media, and ML optional dependency fakes that the parser tests still need for import isolation.
3. Leave the much larger B2B churn MCP broad fake list for a separate PR.
4. Add this plan doc for the local PR review contract.

### Files touched

- `plans/PR-Asyncpg-Parser-Fake-Cleanup.md`
- `tests/test_structured_review_parser_supplements.py`

### Review Contract

Acceptance criteria:

- [ ] `tests/test_structured_review_parser_supplements.py` no longer registers `asyncpg` as a broad `MagicMock` at import time.
- [ ] Existing parser optional dependency fakes remain unchanged where the tests still need them.
- [ ] No production code changes.
- [ ] No backstop workflow or pytest marker boundary changes.

Affected surfaces: parser test import isolation only.

Risk areas: import errors if the parser supplement tests secretly depended on the `asyncpg` fake. Mitigated by scoping the change to HTML parser tests whose imports are scraper/parser focused and by running a syntax check on the branch copy.

Reviewer rules triggered: R1.

## Mechanism

The touched test file used a broad `sys.modules.setdefault(..., MagicMock())` list to fake optional parser dependencies during import. Removing `asyncpg` from that list prevents this parser test from planting a process-wide fake database driver while leaving unrelated parser shims in place.

## Intentional

- Keep this chunk smaller than the churn MCP cleanup so review stays focused.
- Avoid changing #1713 harness behavior or #1712 backstop boundaries.
- Rely on the repo-wide unit backstop enrollment credit for touched unit tests.

## Deferred

- `tests/test_b2b_churn_mcp.py` still has a broad `asyncpg` and `asyncpg.exceptions` fake list and should be handled as the next focused chunk.
- Residual true unit failures should wait until the backstop rerun shows the remaining profile after fake-planter removals.
- Parked hardening: none.

## Parked hardening

Security Guardrails `startup_failure` remains parked as workflow-level hardening because that workflow is identical to `main` and separate from this test cleanup slice.

## Verification

- Bundled runtime syntax check passed: python -m py_compile against a branch copy of `tests/test_structured_review_parser_supplements.py`.
- Local branch-copy search found no remaining `asyncpg` token in `tests/test_structured_review_parser_supplements.py`.
- Full pytest not run locally because this projectless workspace does not have a checkout or repo dependencies; PR CI is expected to run the touched unit backstop gates.

## Estimated diff size

| File group | LOC |
|---|---:|
| Plan doc | ~65 |
| Parser test fake cleanup | 1 deletion |
| **Total** | **~65 touched / 1 removed** |
