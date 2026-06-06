# PR-Content-Ops-DB-Reasoning-Hardening

## Why this slice exists

The AI Content Ops deferred backlog now points at DB reasoning provider
hardening as the next highest-leverage slice. The DB repository already
persists `target_mode`, but the read path ignores it, which lets a row
saved for one asset mode satisfy another mode when selectors overlap.
The host factory also only reads the legacy top-level env var, so Atlas
settings do not expose provider selection.

## Scope (this PR)

1. Make `PostgresCampaignReasoningContextRepository` filter reads by
   `target_mode`, while preserving blank `target_mode` rows as global
   fallback contexts.
2. Add an Atlas settings field for the DB provider opt-in and use it
   when the legacy env var is unset.
3. Update tests for the read query, target-mode fallback semantics, and
   settings-backed factory behavior.
4. Claim this slice in the coordination ledger.

### Files touched

- `plans/PR-Content-Ops-DB-Reasoning-Hardening.md`
- `docs/extraction/coordination/inflight.md`
- `atlas_brain/config.py`
- `atlas_brain/_content_ops_reasoning.py`
- `extracted_content_pipeline/campaign_reasoning_postgres.py`
- `extracted_content_pipeline/storage/migrations/277_campaign_reasoning_contexts.sql`
- `scripts/smoke_extracted_content_ops_execution.py`
- `tests/test_atlas_content_ops_reasoning.py`
- `tests/test_extracted_campaign_reasoning_postgres.py`

## Mechanism

The repository read query keeps the existing account + selector overlap
predicate and adds target-mode gating:

```sql
AND ($3 = '' OR target_mode = $3 OR target_mode = '')
```

That keeps legacy/global rows usable when `target_mode` is blank, while
preventing a `challenger_intel` row from satisfying a
`vendor_retention` request. Ordering stays selector-priority first, then
prefers exact target-mode rows over blank fallback rows at the same
selector priority.

The host factory keeps `ATLAS_CONTENT_OPS_REASONING_DB_ENABLED` as the
highest-precedence override. If it is unset, `_read_db_enabled()` falls
back to `settings.b2b_campaign.content_ops_reasoning_db_enabled`.

## Intentional

- No upsert change in `save_context`; upsert semantics are a separate
  backlog item and should ship with storage contract tests.
- No stale-context sweeper or admin editing workflow; both depend on
  stable storage semantics after this read-path hardening.
- No change to file-backed provider behavior; target-mode filtering is
  DB-only because the DB schema already has the column.

## Deferred

- `PR-Content-Ops-DB-Reasoning-Upsert` should add per-selector upsert
  semantics and define conflict behavior.
- `PR-Content-Ops-Reasoning-Stale-Sweeper` should delete or archive
  old contexts once retention policy is defined.
- `PR-Content-Ops-Reasoning-Admin-Editing` should add operator CRUD
  after the repository contract settles.

## Verification

- `pytest tests/test_extracted_campaign_reasoning_postgres.py tests/test_atlas_content_ops_reasoning.py -q`
  - 43 passed.
- `pytest tests/test_extracted_content_ops_execution_smoke.py::test_content_ops_execution_smoke_cli_exercises_postgres_reasoning_fixture -q`
  - 1 passed.
- `git diff --check`
  - Passed.
- `python -m py_compile atlas_brain/_content_ops_reasoning.py atlas_brain/config.py extracted_content_pipeline/campaign_reasoning_postgres.py tests/test_atlas_content_ops_reasoning.py tests/test_extracted_campaign_reasoning_postgres.py`
  - Passed.
- ASCII byte check across touched Python/docs/sql files.
  - Passed.
- `bash scripts/validate_extracted_content_pipeline.sh`
  - Passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline`
  - Passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt`
  - Passed.
- `bash scripts/check_ascii_python.sh`
  - Passed.
- `bash scripts/run_extracted_pipeline_checks.sh`
  - 1469 passed, 1 existing torch/pynvml warning.

## Estimated diff size

| Area | Estimate |
|---|---:|
| Production + migration/smoke comment | ~105 LOC |
| Tests | ~160 LOC |
| Plan + coordination | ~75 LOC |
| Total | ~390 LOC |
