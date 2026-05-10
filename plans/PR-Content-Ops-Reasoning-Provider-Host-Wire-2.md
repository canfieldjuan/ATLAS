# PR: DB-backed reasoning context provider (Host-Wire-2 of N)

## Why this slice exists

PR #462 (Host-Wire-1) shipped the file-backed reference adapter
for `CampaignReasoningProviderPort` and wired it into the host
route mount. That unblocks single-tenant / staging hosts that
maintain a JSON file of pre-computed contexts. It does NOT
support multi-tenant hosts -- the file-backed provider is
host-wide, every authenticated tenant sees the same contexts,
and operators have to redeploy the file to update payloads.

This slice is the per-tenant DB-backed provider that PR #462's
"Deferred" list called out as a follow-up. It implements the
same `CampaignReasoningProviderPort` contract against a new
`campaign_reasoning_contexts` Postgres table so hosts can:

- Persist per-tenant contexts (`account_id` is a first-class
  column; the read filters by it).
- Update contexts via a CRUD path without redeploying the host.
- Index multiple selectors per row (target_id, company_name,
  contact_email, vendor_name, plus lowercase variants) so the
  same payload from PR #462's file-backed format round-trips.

The route mount wiring extends to a chooser closure (DB > file
> `None`) so hosts that already use the file-backed provider
keep working unchanged, and a host can opt into the DB path by
setting `ATLAS_CONTENT_OPS_REASONING_DB_ENABLED=true`.

## Scope (this PR)

Five files plus the plan doc:

1. **`extracted_content_pipeline/storage/migrations/277_campaign_reasoning_contexts.sql`**
   (new, ~50 LOC): the table.
   - `(id, account_id, target_mode, selectors text[], payload jsonb, created_at, updated_at)`.
   - `selectors` is a TEXT[] persisting both case-as-given and
     lowercase forms; GIN index on it supports the
     `selectors && $2::text[]` lookup predicate.
   - Composite index on `(account_id, target_mode, updated_at DESC)`
     for the tenant-scoped order-by.
   - `target_mode` is persisted but NOT filtered in the default
     read -- mirrors `FileCampaignReasoningContextProvider`'s
     `del target_mode` behavior. Persisting it lets a future
     slice add per-mode filtering without a follow-up migration.

2. **`extracted_content_pipeline/campaign_reasoning_postgres.py`**
   (new, ~165 LOC): the Postgres adapter.
   - `PostgresCampaignReasoningContextRepository` implementing
     `CampaignReasoningProviderPort.read_campaign_reasoning_context`.
   - Read path: builds the same `_candidate_keys` selector set
     as the file-backed provider (target_id + opportunity keys,
     case-as-given + lowercase) and queries
     `WHERE account_id = $1 AND selectors && $2::text[]`,
     ordered by `updated_at DESC LIMIT 1`. Decoded JSONB
     payload flows through `normalize_campaign_reasoning_context`
     so callers get the same shape regardless of how the row
     was written.
   - Write path: `save_context` (outside the upstream Protocol)
     for hosts using this table as their primary store.
     Round-trips via `campaign_reasoning_context_metadata` so
     the persisted JSONB matches the file-backed loader's
     expected layout.
   - Empty-selectors short-circuit on read (skip DB roundtrip)
     and reject on write (`ValueError` -- a row with no
     selectors is unreachable, almost certainly an ETL bug).

3. **`atlas_brain/_content_ops_reasoning.py`** (modified, ~110
   LOC delta): two additions on top of PR #462.
   - `build_postgres_content_ops_reasoning_context_provider`:
     env-gated by `ATLAS_CONTENT_OPS_REASONING_DB_ENABLED`,
     pulls the host pool via `get_db_pool`, binds the new
     repository. Three failure paths (env-disabled, missing
     pool, repo construction error) all resolve to `None`
     with WARN.
   - `select_content_ops_reasoning_context_provider`: the
     chooser closure -- DB > file > `None`. Trivial; both
     factories own their own WARN-and-fall-back so the
     chooser stays a 5-line "try then fall back".

4. **`atlas_brain/api/__init__.py`** (modified, ~2 LOC delta):
   swap the kwarg from
   `build_content_ops_reasoning_context_provider` to
   `select_content_ops_reasoning_context_provider`. Hosts
   that haven't set the new env var see no change -- the
   chooser falls through to the file-backed factory as
   before.

5. **`tests/test_extracted_campaign_reasoning_postgres.py`**
   (new, ~225 LOC): 8 adapter tests pinning candidate-selector
   construction, normalized read, miss/empty-payload returns
   None, no-selectors short-circuit, save round-trip + dedupe,
   empty-selectors rejection, and raw-mapping save acceptance.

6. **`tests/test_atlas_content_ops_reasoning.py`** (modified,
   ~140 LOC delta): 4 DB-factory tests + 3 chooser tests on
   top of PR #462's existing 7 file-factory tests.

7. **`plans/PR-Content-Ops-Reasoning-Provider-Host-Wire-2.md`**
   (this file).

### What's NOT in this slice

- **Migration runner integration.** The SQL file lands; the
  host's existing migration tooling picks it up on the next
  start. No code changes to the runner -- 277 simply slots in
  after 276 (`blog_post_account_scope.sql`), same way 274 and
  275 landed in PRs #458 and the sales-brief slice.
- **Pydantic settings nesting.** The new env var is read via
  `os.environ.get`, same pattern as PR #462. A future
  settings refactor folds both opt-ins (file path + DB
  enabled) into a `ContentOpsSettings` block.
- **Removing the file-backed factory.** Both providers stay
  available -- the file-backed adapter is a meaningful path
  for staging hosts and dependency-light deploys that don't
  run Postgres. The chooser keeps the priority explicit so
  hosts can run either or both.
- **Per-tenant isolation tests at the route layer.** The
  adapter pins `account_id = $1` on every read; the route
  mount's `_capture_content_ops_auth_user` already binds the
  authenticated tenant's `account_id` into the scope. The
  cross-tenant-leak failure mode would manifest at the route
  layer (PR #455's contract); that contract test belongs in
  the upstream package's route-layer tests, not here.
- **Operator UI for editing context rows.** Hosts populate
  the table from their own ETL today; a NocoDB view over the
  table is the natural admin path (the table is auto-discovered
  on next NocoDB start). Dedicated admin tooling is a
  follow-up.
- **`save_context` upsert / conflict semantics.** Writes are
  pure INSERTs -- a given (account, selectors) tuple can have
  multiple rows; the read returns the newest by
  `updated_at`. Hosts that need delete-old-on-update
  semantics can run a sweep, or a follow-up slice can add
  `ON CONFLICT` once the unique-key shape is settled.

## Mechanism

### Migration

```sql
-- 277_campaign_reasoning_contexts.sql
CREATE TABLE IF NOT EXISTS campaign_reasoning_contexts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id TEXT NOT NULL DEFAULT '',
    target_mode TEXT NOT NULL DEFAULT '',
    selectors TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_campaign_reasoning_contexts_selectors
    ON campaign_reasoning_contexts USING GIN (selectors);

CREATE INDEX IF NOT EXISTS idx_campaign_reasoning_contexts_account
    ON campaign_reasoning_contexts (account_id, target_mode, updated_at DESC);
```

### Adapter read path

```python
async def read_campaign_reasoning_context(
    self, *, scope, target_id, target_mode, opportunity,
):
    del target_mode  # parity with FileCampaignReasoningContextProvider
    selectors = _candidate_selectors(
        target_id=target_id, opportunity=opportunity,
    )
    if not selectors:
        return None  # no DB roundtrip
    row = await self.pool.fetchrow(
        f"""
        SELECT payload FROM {self.table}
        WHERE account_id = $1 AND selectors && $2::text[]
        ORDER BY updated_at DESC LIMIT 1
        """,
        scope.account_id or "", list(selectors),
    )
    if row is None:
        return None
    payload = decode_jsonb_field(row_to_dict(row).get("payload"), default={})
    if not isinstance(payload, Mapping) or not payload:
        return None
    context = normalize_campaign_reasoning_context(payload)
    return context if context.has_content() else None
```

### Host factory + chooser

```python
def build_postgres_content_ops_reasoning_context_provider(
    *,
    enabled_factory=None,
    pool_factory=None,
    repository_factory=None,
):
    if not (enabled_factory or _read_db_enabled)():
        return None
    try:
        pool = (pool_factory or _default_pool_factory)()
    except Exception as exc:
        logger.warning("...DB pool acquire failed: %s; ...", exc)
        return None
    if pool is None:
        logger.warning("...pool is not available; ...")
        return None
    try:
        return (repository_factory or _default_repository_factory)(pool)
    except Exception as exc:
        logger.warning("Failed to construct ... repository: %s; ...", exc)
        return None


def select_content_ops_reasoning_context_provider(
    *, db_factory=None, file_factory=None,
):
    db_pick = (db_factory or build_postgres_content_ops_reasoning_context_provider)()
    if db_pick is not None:
        return db_pick
    return (file_factory or build_content_ops_reasoning_context_provider)()
```

### Route mount delta

```python
content_ops_router = create_content_ops_control_surface_router(
    dependencies=[Depends(_capture_content_ops_auth_user)],
    execution_services_provider=lambda: (
        build_content_ops_execution_services(enable_db_services=True)
    ),
    scope_provider=build_content_ops_scope,
    reasoning_context_provider=select_content_ops_reasoning_context_provider,  # was build_content_ops_reasoning_context_provider
)
```

## Intentional

- **DB > file > None priority.** DB scales per-tenant; the
  file-backed adapter stays as a single-tenant / staging
  fallback. Both factories own their own WARN-and-fall-back
  so the chooser is trivial.
- **`del target_mode` parity with the file-backed provider.**
  `FileCampaignReasoningContextProvider` ignores `target_mode`
  in its read; this slice mirrors the behavior so the same
  test fixtures and the same operator JSON files round-trip
  through either provider unchanged. The column is persisted
  for a future per-mode filter slice.
- **Selectors as TEXT[] with GIN index, not a side table.**
  Selector cardinality per row is small (typically 4-10
  variants); a side table would multiply rows by ~5x for no
  read benefit. GIN's `&&` is the right operator for "any
  candidate matches any persisted selector" and runs in
  O(log n) over millions of rows.
- **`updated_at DESC LIMIT 1` for tie-break.** Mirrors
  `setdefault` first-key-wins in the file index but with
  recency semantics: when an ETL produces multiple rows for
  the same selectors, the newest payload wins. A unique
  constraint would force every ETL to delete-then-insert; the
  recency tie-break keeps writes idempotent.
- **Both case-as-given and lowercase selectors persisted.**
  GIN's `&&` is exact-match -- adding lowercase variants at
  write time keeps the read path's lowercase normalization
  cheap (no runtime LOWER() over the column, no functional
  index).
- **Empty-selectors rejection on save.** A row with no
  selectors is unreachable; surfacing it loudly catches
  upstream ETL bugs before they silently bloat the table.
- **Empty-selectors short-circuit on read.** No DB roundtrip
  when there's nothing to match -- avoids
  `selectors && '{}'::text[]` running a full predicate scan
  with no useful filter.
- **Env var rather than Pydantic settings.** Symmetric with
  PR #462's `ATLAS_CONTENT_OPS_REASONING_CONTEXT_PATH` and
  PR #453's `_content_ops_infrastructure.py`. The 5000+ LOC
  `config.py` refactor is a separate slice.
- **Lazy imports in factory bodies.** Same pattern as PR
  #462's `_default_loader`: the heavier
  `extracted_content_pipeline.campaign_reasoning_postgres`
  module loads only when the DB env var is set + a pool is
  available.
- **DI kwargs (`enabled_factory`, `pool_factory`,
  `repository_factory`, `db_factory`, `file_factory`).** Same
  testability pattern as the rest of the Content Ops wiring
  slices.
- **Pure INSERT writes (no upsert).** Hosts can run multiple
  ETLs against the same table without coordinating on a
  conflict shape. The recency tie-break in the read absorbs
  the multiplicity.

## Deferred

- Intervention-pipeline-backed provider that surfaces
  `intelligence/autonomous_narrative_architect` outputs as
  per-opportunity reasoning contexts. Slot in by registering
  a third factory in the chooser; no migration required.
- Per-mode filtering on read (`target_mode = $3 OR target_mode = ''`).
  The column is persisted; a future slice flips the read
  predicate.
- Pydantic settings integration folding both env vars into a
  `ContentOpsSettings` block.
- Operator admin UI for editing context rows. NocoDB
  auto-discovers the table on next start; a custom view is a
  follow-up.
- Upsert / conflict semantics on `save_context` (current
  behavior is pure INSERT; reads use `updated_at DESC` to
  pick the newest).
- Sweeper job to delete superseded rows once the recency
  tie-break is in production use.
- Cross-tenant route-layer isolation contract test (belongs
  in the upstream package's route-layer tests).

## Verification

- `pytest tests/test_extracted_campaign_reasoning_postgres.py
  tests/test_atlas_content_ops_reasoning.py` -- new tests
  pass; pre-existing 7 file-factory tests stay green.
- AST + ASCII gates clean.
- Smoke: `python -c "from atlas_brain._content_ops_reasoning
  import select_content_ops_reasoning_context_provider; print(
  select_content_ops_reasoning_context_provider())"` -> `None`
  (neither env var set; chooser falls through to file
  factory which also returns `None`).

## Estimated diff size

- `277_campaign_reasoning_contexts.sql`: ~50 LOC.
- `campaign_reasoning_postgres.py`: ~165 LOC.
- `_content_ops_reasoning.py` delta: ~115 LOC.
- `api/__init__.py` delta: ~2 LOC.
- `test_extracted_campaign_reasoning_postgres.py`: ~225 LOC.
- `test_atlas_content_ops_reasoning.py` delta: ~140 LOC.
- Plan doc: ~250 LOC.

Total: ~945 LOC. Over the 400 LOC soft cap; indivisible --
the migration, adapter, host factory, and chooser must land
together to be useful (a half-shipped slice would either
have an unwired adapter or a chooser pointing at a
non-existent factory). Production code is ~330 LOC; the
remainder is plan doc + tests + the deliberately-thorough
test inventory mirroring PR #458's blueprint adapter
coverage.
