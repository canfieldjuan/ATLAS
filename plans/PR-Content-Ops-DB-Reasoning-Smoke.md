# PR-Content-Ops-DB-Reasoning-Smoke

## Why this slice exists

PR #463 added the DB-backed Content Ops reasoning provider and #467/#468 made
generated-asset results expose consumed reasoning payloads. The remaining
verification gap is the host-facing execution smoke: it could prove generic
reasoning usage, but not that the Postgres reasoning adapter can feed the
same execution/audit contract.

## Scope (this PR)

1. Add a `--reasoning-provider postgres-fixture` mode to
   `scripts/smoke_extracted_content_ops_execution.py`.
2. Use the real `PostgresCampaignReasoningContextRepository` against a tiny
   in-memory asyncpg-shaped pool, so the smoke stays offline and deterministic.
3. Assert the smoke JSON includes consumed reasoning payloads from that
   Postgres fixture.
4. Update README/runbook examples and clean the stale #468 coordination row.

### Files touched

- `scripts/smoke_extracted_content_ops_execution.py`
- `tests/test_extracted_content_ops_execution_smoke.py`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/docs/host_install_runbook.md`
- `docs/extraction/coordination/inflight.md`

## Mechanism

The existing smoke fake services already exercise
`ContentOpsExecutionServices.with_reasoning_context(...)` and the step-level
`reasoning.contexts_used` audit. This PR keeps that flow and adds an alternate
offline provider fixture:

- `sample` remains the default and uses a generic object provider.
- `postgres-fixture` builds `PostgresCampaignReasoningContextRepository` with
  an in-memory pool whose `fetchrow(...)` method returns one normalized
  reasoning context for the smoke target selectors.

The fake generated-asset services call
`read_campaign_reasoning_context(...)` when the provider exposes that method,
then emit the returned context through `consumed_reasoning_contexts`. The
existing smoke validator confirms the execution result mirrors those payloads
into `step.reasoning.consumed_contexts`.

## Intentional

- No live database, network, sender, or LLM handle is opened.
- No generated-asset production service behavior changes.
- The fixture validates adapter compatibility, not production data quality.
- The default smoke command remains unchanged.

## Deferred

- Live database integration smoke with a real `EXTRACTED_DATABASE_URL`.
- UI Reasoning Context Drawer rendering.

## Verification

- Focused smoke tests.
- `python -m py_compile scripts/smoke_extracted_content_ops_execution.py`.
- Full extracted pipeline check.
- `git diff --check`.
- ASCII byte check on edited Python/test files.

## Estimated diff size

6 files, roughly +100 / -1. Under the 400 LOC soft review budget.
