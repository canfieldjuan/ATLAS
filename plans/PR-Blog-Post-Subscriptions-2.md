# PR: autonomous task tenant fanout (Subscriptions-2 of 2)

## Why this slice exists

PR-Subscriptions-1 (PR #460) landed the
`b2b_blog_post_subscriptions` table and the typed reader.
This slice closes the loop: when the autonomous blog-post
task (`atlas_brain/autonomous/tasks/b2b_blog_post_generation.py`)
generates a `PostBlueprint`, fan it out to every active
subscriber by writing a per-account copy into
`blog_blueprints` (PR #458's table). The Content Ops
`/api/v1/content-ops/execute` route then reads those
blueprints under each tenant's authenticated scope and
generates drafts on demand.

After this lands, the wired `blog_post` slot from PR #459
becomes functionally end-to-end: autonomous task ->
subscriber fanout -> `/execute` route -> draft.

## Scope (this PR)

Fanout helper + autonomous task hook + tests. No subscription
admin endpoints; no schema changes.

### Files touched

1. **`atlas_brain/_blog_blueprint_fanout.py`** (new, ~110
   LOC): the fanout helper. Pure async function
   `fanout_blueprint(pool, blueprint)` that:
   - Calls `list_active_blog_post_subscriptions(pool)` (PR
     #460's reader).
   - Filters to subscriptions whose
     `matches(topic_type, target_mode)` accepts the
     blueprint.
   - Builds a JSONB-safe payload from the host's
     `PostBlueprint` (sections + charts + tags +
     data_context + quotable_phrases + cta).
   - For each matching subscription, calls
     `PostgresBlogBlueprintRepository.save_blueprints` with
     `TenantScope(account_id=subscription.account_id)`.
   - Per-account `save` failures are logged but don't abort
     the remaining fanout -- the autonomous task's draft
     has already been generated and stored; partial fanout
     is strictly better than no fanout.
   - DI kwargs (`subscriptions_factory`, `repo_factory`)
     so tests stub host infrastructure without triggering
     asyncpg / heavy host init.
   - Lives at `atlas_brain/` root (not `services/`) for the
     same reason as PR #460's `_blog_post_subscriptions.py`
     and PR #453's `_content_ops_infrastructure.py`:
     importing `atlas_brain.services` triggers an eager
     torch / ollama load.

2. **`atlas_brain/autonomous/tasks/b2b_blog_post_generation.py`**
   (modified, ~15 LOC delta): import the fanout helper and
   call it at the two existing `_assemble_and_store`
   callsites (lines 3222 and 3462) AFTER the draft is
   stored. Wrapped in try/except so a fanout failure never
   regresses the existing autonomous flow; logged at WARN
   level.

3. **`tests/test_atlas_blog_blueprint_fanout.py`** (new,
   8 tests, ~225 LOC): pin the helper's empty-list
   short-circuit, no-matching-subscribers short-circuit,
   payload serialization, multi-subscriber count return,
   per-account save isolation (one failure doesn't abort
   the rest), `target_mode` default, and the matches()
   filter integration with both topic_types and
   target_modes constraints.

4. **`plans/PR-Blog-Post-Subscriptions-2.md`** (this file).

### What's NOT in this slice

- **Subscription management endpoints / admin UI.** Same as
  PR #460's deferred list -- subscriptions land via direct
  SQL or NocoDB until a follow-on slice ships an admin
  route.
- **Backfilling existing blog_posts into blog_blueprints.**
  The autonomous task's prior runs produced finished blog
  drafts in `blog_posts`; this slice only fans out
  blueprints generated AFTER it lands. Operators wanting
  retroactive fanout run a one-shot ETL.
- **Per-subscription cadence override.** Cadence is the
  autonomous task's existing cron; per-account override is
  out of scope.

## Mechanism

The helper is a pure async function with three tunable
seams (DI kwargs):

```python
async def fanout_blueprint(
    pool: Any,
    blueprint: Any,  # PostBlueprint duck-typed
    *,
    target_mode: str = DEFAULT_TARGET_MODE,
    subscriptions_factory: Callable | None = None,
    repo_factory: Callable | None = None,
) -> int:
    if subscriptions_factory is not None:
        subs = await subscriptions_factory()
    else:
        subs = await list_active_blog_post_subscriptions(pool)
    if not subs:
        return 0

    matching = [
        s for s in subs
        if s.matches(topic_type=blueprint.topic_type,
                     target_mode=target_mode)
    ]
    if not matching:
        return 0

    repo = (repo_factory or PostgresBlogBlueprintRepository)(pool=pool)
    payload = _serialize_payload(blueprint)

    saved = 0
    for sub in matching:
        record = BlogBlueprint(
            target_mode=target_mode,
            topic_type=blueprint.topic_type,
            slug=blueprint.slug,
            suggested_title=blueprint.suggested_title,
            payload=payload,
        )
        try:
            await repo.save_blueprints(
                [record],
                scope=TenantScope(account_id=sub.account_id),
            )
            saved += 1
        except Exception as exc:
            logger.warning(
                "Blog blueprint fanout failed for account_id=%s slug=%s: %s",
                sub.account_id, blueprint.slug, exc,
            )
    return saved
```

The autonomous task hook is a single new line at each of the
two `_assemble_and_store` callsites (in a try/except so a
fanout exception can't regress the existing flow):

```python
post_id = await _assemble_and_store(...)
if post_id:
    try:
        fanout_count = await _fanout_blog_blueprint(pool, blueprint)
        if fanout_count:
            logger.info(
                "Blog blueprint fanned out: slug=%s subscribers=%d",
                blueprint.slug, fanout_count,
            )
    except Exception as exc:
        logger.warning("Blog blueprint fanout exception: %s", exc)
```

## Intentional

- **Fixed `DEFAULT_TARGET_MODE = "b2b_blog_post"`.** The
  autonomous task generates one shape of B2B retention
  blog post regardless of topic_type; subscriptions filter
  by topic_type for variety. `target_mode` is the
  top-level grouping for future fan-in (e.g. a separate
  "executive brief" generator emitting
  `target_mode="executive_brief"`). Operators that want
  finer per-target-mode subscription control populate
  `b2b_blog_post_subscriptions.target_modes[]` accordingly.
- **Per-account `save` failures don't abort the fanout.**
  The autonomous task's draft has already been stored
  (post_id is non-empty); partial fanout is strictly better
  than no fanout. Errors logged at WARN.
- **Fanout happens AFTER `_assemble_and_store`** rather than
  before. If the draft generation fails mid-way, we don't
  want to land a blueprint in subscribers' `blog_blueprints`
  that the host's blog_posts table never produced anything
  from -- that would surface in `/execute` as a draft
  attempt that may then fail differently.
- **DI kwargs (`subscriptions_factory`, `repo_factory`) on
  the helper.** Same pattern as PRs #453 / #455 / #460 --
  tests pass stubs without triggering the host's full init
  chain.
- **Helper lives at `atlas_brain/` root with underscore
  prefix.** Matches `_blog_post_subscriptions.py` /
  `_content_ops_infrastructure.py` / `_content_ops_scope.py`
  -- avoids the heavy `atlas_brain.services` import chain
  in dev environments.
- **Two callsites patched, not one.** The autonomous task
  has two `_assemble_and_store` callsites (different topic
  generation phases); both produce blueprints that should
  fan out. Refactoring them into a single funnel is a
  separate refactor; the surgical insert keeps this slice
  focused.

## Deferred

- Subscription management admin UI / API endpoints.
- Backfilling pre-existing `blog_posts` rows into
  `blog_blueprints` for new subscribers.
- Per-subscription cadence overrides.
- `b2b_blog_post_subscriptions` audit columns
  (`created_by`, `updated_by`) -- carry no data without an
  admin route.
- Multi-pass reasoning provider host wiring.

## Verification

- `pytest tests/test_atlas_blog_blueprint_fanout.py`
  -- new tests pass.
- `pytest tests/test_atlas_blog_post_subscriptions.py`
  -- existing PR #460 tests stay green.
- AST + ASCII checks on the new module + test +
  modified autonomous task file.
- Smoke: `python -c "from atlas_brain._blog_blueprint_fanout
  import fanout_blueprint; print(fanout_blueprint)"` -> the
  function imports cleanly without triggering the heavy
  host init chain.

## Estimated diff size

- `_blog_blueprint_fanout.py`: ~115 LOC.
- Autonomous task patch: ~15 LOC delta (1 import + 2
  callsite hooks of ~7 LOC each).
- Tests: ~225 LOC (8 tests).
- Plan doc: ~190 LOC.

Total: ~545 LOC. Marginally over the 400 LOC soft cap.
Indivisible at the helper-plus-callsite seam: the helper
without callsites is dead code, the callsites without the
helper would inline the logic into the 9000-line
autonomous task file. Splitting into separate "helper PR"
+ "callsites PR" buys nothing -- the helper's tests
exercise the full contract. The autonomous task
modifications are mechanical and don't change behavior on
empty-subscription deployments.
