# Hybrid Extraction Scope Guard

Use this checklist before opening each PR in the hybrid extraction program.

## Current program scope (in)

1. Reasoning consumer adapter seam for MCP/API overlays.
2. Contract/regression tests that preserve existing response shapes.
3. Extracted content provider-port boundary (`CampaignReasoningProviderPort`).
4. Entry-point wiring (example/postgres scripts) to provider-port compatible loader.
5. Documentation/runbook alignment for the above changes.

## Out of scope (for current wave)

1. Rewriting churn reasoning producer internals (`b2b_reasoning_synthesis`, pool compression logic).
2. Changing existing MCP/API response contracts or task signatures.
3. Schema migrations for new reasoning tables.
4. Cross-product ontology redesign.
5. LLM routing/provider behavior changes.

## PR gate: drift check

A PR is in-scope only if all are true:

- It is directly mapped to PR-1/PR-2/PR-3 items in `docs/hybrid_extraction_execution_board.md`.
- It is additive or narrowing-risk (tests/docs/adapter/port wiring), not behavior-changing.
- It does not require downstream consumer contract rewrites.
- It does not introduce new runtime dependencies outside existing extracted boundaries.

## Required PR metadata

Each PR description must include:

1. **Execution-board mapping** (e.g., “PR-2 hardening” or “PR-3 wiring”).
2. **Behavior-change statement** (“No behavior change” or explicit compatible delta).
3. **Contract impact** (none/additive/breaking).
4. **Rollback plan** (file list to revert if needed).

## Stop conditions (pause and re-scope)

Pause implementation and open a design note when any occurs:

1. Need to alter canonical reasoning field names/types.
2. Need to change task/API function signatures.
3. Need to introduce new persistence artifacts.
4. Need to import Atlas-core producer internals into extracted runtime paths.

## Next in-scope steps

1. PR-3 compatibility tests for Postgres runner using provider-port loader path.
2. Optional migration note: old loader name vs new loader wrapper for host teams.
3. Execution-board progress update with completed checkboxes only.
