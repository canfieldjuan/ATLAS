# PR: blog blueprint storage layer (Storage-1 of 2)

## Why this slice exists

After PR #456 (E3), the host's Content Ops execution-services
bundle wires 5 of 6 outputs (`signal_extraction`,
`landing_page`, `email_campaign`, `report`, `sales_brief`).
The last unwired slot is `blog_post`.

Unlike the four already-wired DB-backed outputs, the package's
`BlogBlueprintRepository` (`extracted_content_pipeline/blog_ports.py:39`)
is a Protocol with no implementation. The host generates
blueprints in-memory inside
`atlas_brain/autonomous/tasks/b2b_blog_post_generation.py`
(deterministic `PostBlueprint` dataclasses) and consumes them
immediately -- no persistence layer connects the two halves.
Without a `PostgresBlogBlueprintRepository`, the bundle factory
has nothing concrete to wire into the `BlogPostGenerationService`
constructor.

This slice adds the storage layer only -- migration + repository
+ tests. A follow-up slice (PR-Content-Ops-Execution-Services-Wire-4)
plugs the new repo into the bundle factory once it lands.

## Scope (this PR)

Three files plus the plan doc:

1. **`extracted_content_pipeline/storage/migrations/274_blog_blueprints.sql`** (new):
   Creates the `blog_blueprints` table -- structurally similar
   to `273_reports.sql` (the most recent migration). Fields:
   `id`, `account_id`, `target_mode`, `topic_type`, `slug`,
   `suggested_title`, `payload` (JSONB -- the full blueprint
   dict the LLM consumes), `status`, `created_at`, `consumed_at`.
   Indexes on `(account_id, target_mode, status, created_at)`
   for the read path, plus a unique index on
   `(account_id, target_mode, slug)` to prevent duplicates from
   the host's idempotent generators.

2. **`extracted_content_pipeline/blog_blueprint_postgres.py`** (new):
   `PostgresBlogBlueprintRepository(pool=...)` -- frozen
   dataclass following `PostgresLandingPageRepository` /
   `PostgresReportRepository` pattern.
   - `read_blog_blueprints(*, scope, target_mode, limit,
     filters=None)` -- the protocol method consumed by
     `BlogPostGenerationService.generate()` at
     `extracted_content_pipeline/blog_generation.py:221`.
     Returns `Sequence[Mapping[str, Any]]` -- decodes each
     row's `payload` JSONB and merges in the row-level metadata
     (`id`, `target_mode`, `topic_type`, `slug`,
     `suggested_title`) so the generator sees a self-contained
     blueprint dict.
   - `save_blueprints(blueprints, *, scope)` -- write helper.
     Not part of the upstream Protocol but needed so a host-side
     adapter (the autonomous task, or a separate ETL) can land
     blueprints in the table. Returns assigned ids.
   - `mark_consumed(blueprint_ids, *, scope)` -- sets
     `consumed_at` so duplicate generation runs skip already-used
     blueprints. Optional path; the read path filters
     `consumed_at IS NULL` by default.

3. **`tests/test_extracted_blog_blueprint_postgres.py`** (new):
   ~6 tests pinning the round-trip + scope-isolation contract,
   following `test_extracted_blog_post_postgres.py` shape (fake
   pool with `fetch` / `fetchval` / `execute` recorders).
   - `save_blueprints` round-trips payload through JSONB
     round-trip
   - `read_blog_blueprints` filters by `account_id` +
     `target_mode` + `consumed_at IS NULL`
   - `read_blog_blueprints` honors `limit`
   - `read_blog_blueprints` returns merged payload + row metadata
   - `mark_consumed` sets the `consumed_at` timestamp
   - `read_blog_blueprints` with no rows returns `()`
     (defensive empty-table canary)

### What's NOT in this slice

- **Bundle factory wiring of `blog_post`.** Separate slice
  (PR-Content-Ops-Execution-Services-Wire-4); requires this
  storage layer to land first.
- **Host autonomous task changes** to persist `PostBlueprint`s
  into the new table. The host's existing in-memory pipeline
  keeps working; a future slice extends it (or adds an ETL
  step) to populate `blog_blueprints` so the wired generator
  has data to read.
- **`BlogBlueprintRepository` Protocol changes.** The upstream
  contract stays as-is; this slice only adds an implementation.

## Mechanism

The repository is a thin asyncpg adapter following the
established frozen-dataclass shape:

```python
@dataclass(frozen=True)
class PostgresBlogBlueprintRepository:
    pool: Any
    table: str = "blog_blueprints"

    async def read_blog_blueprints(
        self,
        *,
        scope: TenantScope,
        target_mode: str,
        limit: int,
        filters: Mapping[str, Any] | None = None,
    ) -> Sequence[Mapping[str, Any]]:
        rows = await self.pool.fetch(
            f"""
            SELECT id, target_mode, topic_type, slug, suggested_title, payload
            FROM {self.table}
            WHERE account_id = $1
              AND target_mode = $2
              AND consumed_at IS NULL
            ORDER BY created_at DESC
            LIMIT $3
            """,
            scope.account_id or "",
            target_mode,
            int(limit),
        )
        return tuple(_row_to_blueprint(row) for row in rows)
```

Each row's `payload` JSONB carries the rich blueprint dict
(sections / charts / tags / data_context / etc.) that the
LLM prompt consumes; `_row_to_blueprint` decodes it and
merges row metadata so the generator sees a single dict.

## Intentional

- **Migration number 274.** Continues the existing sequence
  (273_reports.sql is most recent in
  `extracted_content_pipeline/storage/migrations/`). No
  conflict with host-side migrations -- the package's
  numbering namespace is independent.
- **`payload` JSONB rather than per-field columns.** Blueprint
  shape evolves with the generators; flattening into typed
  columns would force a migration on every blueprint-shape
  tweak. JSONB also lets multiple `topic_type`s coexist
  without sparse columns.
- **Unique `(account_id, target_mode, slug)` constraint.** The
  host's blueprint generators are idempotent on
  `(target, topic_type)`; the slug derives from those. Without
  the constraint, repeat runs would multiply rows and the
  LIMIT-N read would surface duplicates.
- **`consumed_at` rather than DELETE-on-read.** Audit trail +
  ability to re-run a blueprint by clearing the timestamp.
  Default WHERE filter keeps the consumed rows out of normal
  reads.
- **`save_blueprints` outside the Protocol.** The upstream
  `BlogBlueprintRepository` Protocol is read-only by design
  (different writers per host). The package's repo concrete
  adds the writer for hosts that want blueprints in this
  table; hosts with their own blueprint store can implement
  the Protocol against it instead.
- **Frozen dataclass shape, pool-only construction.** Matches
  PR #455 / PR #456 wiring shape -- the repo's only state
  is the pool + a configurable table name (test override).

## Deferred

- Bundle wiring of `blog_post` (next slice).
- Host autonomous task changes to persist `PostBlueprint`s
  into the new table (separate slice -- the in-memory pipeline
  keeps working; the new path is opt-in).
- Per-blueprint quality-gate or expiry policy.
- Read-side filters beyond `target_mode` (e.g. `topic_type`,
  `vendor`) -- can be folded into the existing `filters`
  arg as needed; not exercised today.

## Verification

- `pytest tests/test_extracted_blog_blueprint_postgres.py`
  -- new tests pass.
- `bash scripts/check_ascii_python.sh` -- ASCII clean.
- AST parse of new module + test file.
- Existing extracted-pipeline gates (validate +
  forbid_atlas_reasoning_imports + audit_extracted_standalone)
  stay clean.

## Estimated diff size

- Migration: ~30 LOC.
- Repository: ~150 LOC.
- Tests: ~200 LOC.
- Plan doc: ~150 LOC.

Total: ~530 LOC. Slightly over the 400 LOC soft cap. The
storage layer is indivisible -- the migration, repo
implementation, and tests must ship together for the schema
to be useful. A test-only PR with no implementation has no
review value, and an implementation-only PR with no migration
has no schema to bind to.
