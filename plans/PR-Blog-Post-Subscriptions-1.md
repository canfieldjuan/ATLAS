# PR: blog post subscriptions schema + reader (Subscriptions-1 of 2)

## Why this slice exists

PR #459 wired `blog_post` into the Content Ops execution-services
bundle (last unwired slot). The wired generator reads from
`blog_blueprints` filtered by `scope.account_id` (Codex P1
cross-tenant isolation contract). But the host's autonomous
blog-post task (`atlas_brain/autonomous/tasks/b2b_blog_post_generation.py`)
runs as a single-host scheduled cron with no tenant context --
it generates `PostBlueprint`s in-memory and produces drafts
directly without persisting blueprints anywhere.

Result: the bundle advertises `blog_post`, but the table is
empty for every authenticated tenant. Calling
`/api/v1/content-ops/execute` with `outputs=["blog_post"]`
returns zero drafts.

To bridge, the autonomous task needs to fan out each generated
blueprint to every account that's subscribed to B2B blog posts.
This slice lands the storage + reader for that subscription
list. PR-Subscriptions-2 wires the autonomous task itself.

This shape mirrors `b2b_report_subscriptions`
(`atlas_brain/storage/migrations/265_b2b_report_subscriptions.sql`)
-- the established host pattern for per-tenant subscription
to scheduled content-generation outputs. Same FK shape
(`account_id REFERENCES saas_accounts(id) ON DELETE CASCADE`),
same enabled/disabled flag, same per-account uniqueness.

## Scope (this PR)

Storage + reader only. No autonomous task changes; no
end-to-end fanout.

### Files touched

1. **`atlas_brain/storage/migrations/324_b2b_blog_post_subscriptions.sql`**
   (new): the subscription table + indexes. Mirrors
   `b2b_report_subscriptions` shape minus the
   delivery-mechanic fields (recipient_emails, frequency,
   freshness_policy) since blueprint fan-out is a
   write-side push, not a read-side delivery. Carries
   `topic_types TEXT[]` + `target_modes TEXT[]` filters
   (empty arrays = subscribe to all values) so accounts can
   opt into specific blog topic_types or target_modes
   without subscribing to the firehose.

2. **`atlas_brain/_blog_post_subscriptions.py`** (new):
   `BlogPostSubscription` dataclass (frozen, account_id +
   topic_types + target_modes) + `matches()` predicate +
   `list_active_blog_post_subscriptions(pool)` async reader
   that returns enabled rows. The reader is the only thing
   PR-Subscriptions-2's autonomous task calls; the dataclass
   is the typed contract.

   Lives at the `atlas_brain/` root (not inside
   `services/`) for the same reason as the Content Ops
   LLM/Skill adapters from PR #453: importing
   `atlas_brain.services` triggers an eager torch / ollama
   load that panics in dependency-light dev envs. The
   underscore-prefix indicates a host-internal module --
   not a public re-export surface.

3. **`tests/test_atlas_blog_post_subscriptions.py`** (new):
   ~8 tests pinning the reader's enabled-only filter, the
   row-to-dataclass shape, and the `matches()` predicate's
   filter semantics (empty = accept-all, populated = subset
   match). Uses an asyncpg-shaped fake pool, same shape as
   `tests/test_extracted_blog_blueprint_postgres.py`.

4. **`plans/PR-Blog-Post-Subscriptions-1.md`** (this file).

### What's NOT in this slice

- **Autonomous task fanout.** Saved for PR-Subscriptions-2.
  That slice imports `list_active_blog_post_subscriptions`
  here, loops over each subscription, calls
  `subscription.matches(topic_type=..., target_mode=...)`,
  and writes a per-account copy of the blueprint via
  `PostgresBlogBlueprintRepository.save_blueprints`.
- **Subscription management endpoints / admin UI.**
  Subscriptions land via direct SQL or NocoDB until a
  follow-on slice ships an admin route. The B2B report
  subscription pattern accumulated its admin surface
  incrementally over many PRs (#266+); the same path here.
- **Per-subscription delivery configuration** (recipient
  emails, frequency, retention). The blueprint fanout is
  fire-and-forget into `blog_blueprints` -- the
  content-ops `/execute` route does the actual delivery on
  user request.

## Mechanism

The reader is a thin asyncpg call returning typed rows:

```python
@dataclass(frozen=True)
class BlogPostSubscription:
    account_id: str
    topic_types: tuple[str, ...] = ()
    target_modes: tuple[str, ...] = ()

    def matches(self, *, topic_type: str, target_mode: str) -> bool:
        if self.topic_types and topic_type not in self.topic_types:
            return False
        if self.target_modes and target_mode not in self.target_modes:
            return False
        return True


async def list_active_blog_post_subscriptions(
    pool: Any,
) -> list[BlogPostSubscription]:
    rows = await pool.fetch(
        """
        SELECT account_id, topic_types, target_modes
        FROM b2b_blog_post_subscriptions
        WHERE enabled = TRUE
        ORDER BY account_id
        """
    )
    return [
        BlogPostSubscription(
            account_id=str(row["account_id"]),
            topic_types=tuple(row["topic_types"] or ()),
            target_modes=tuple(row["target_modes"] or ()),
        )
        for row in rows
    ]
```

PR-Subscriptions-2 will call this once per autonomous run and
fan out blueprints to each subscriber whose `matches()`
predicate accepts the blueprint's topic_type / target_mode.

## Intentional

- **Mirror `b2b_report_subscriptions` shape**. Established
  host pattern for per-tenant subscription to scheduled
  content. Same FK, same enabled flag, same uniqueness on
  `account_id`. Operators familiar with the report
  subscription admin path apply the same mental model.
- **`TEXT[]` filters with empty-array = accept-all
  semantics.** Avoids a separate "subscribe to all" boolean
  toggle; an empty filter is the natural representation of
  "no restriction". The `matches()` predicate enforces the
  semantics so callers don't conditionalize.
- **Reader returns a list, not a generator / cursor.** The
  expected subscription count is small (per-tenant
  business-tier opt-in, not per-user); a list keeps the
  call site simple. If subscription counts grow large, the
  reader can switch to async iteration without a contract
  change.
- **`UNIQUE (account_id)` rather than `(account_id, topic)`.**
  Each account gets one subscription row carrying the full
  filter set; multiple-subscription-per-account would
  require merge semantics on fanout. Single row keeps the
  authoritative filter list visible.
- **No `created_by` / `updated_by` audit columns.** The
  report subscription mirror has them, but those exist to
  power the report subscription admin UI's "set up by:"
  display. Without an admin UI here yet, the columns would
  carry no data; defer until an admin route ships.

## Deferred

- Autonomous task fanout (PR-Subscriptions-2).
- Subscription management endpoints + admin UI.
- Per-subscription delivery cadence (the autonomous task's
  cron schedule is the cadence; per-account override is
  out of scope).
- Blueprint expiry / GC -- consumed_at + retention policy
  handled in PR #458's storage layer.

## Verification

- `pytest tests/test_atlas_blog_post_subscriptions.py`
  -- new tests pass.
- AST + ASCII checks on the new module + test.
- `bash scripts/check_ascii_python.sh` -- pass.

## Estimated diff size

- Migration: ~25 LOC.
- Reader module: ~80 LOC.
- Tests: ~180 LOC.
- Plan doc: ~165 LOC.

Total: ~450 LOC. Marginally over the 400 LOC soft cap.
Indivisible -- a migration without a reader has no
consumer, a reader without the migration won't pass tests
against a real pool. Plan doc dominates.
