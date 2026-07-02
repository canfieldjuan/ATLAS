# PR-Reddit-Listening-Hardening

## Why this slice exists

The #1934 arc ended with the operator's post-merge audits on #1940 (S5)
and #1941 (S6) parking six LOW-severity hardening items for a dedicated
follow-up PR ("fold ... into the existing hardening PR ... That list is
now 6 items"). The list deduplicates to five distinct fixes (the
`atlas_reddit/tracker.py` StopIteration guard was flagged on both audits). None broke the arc's
real flow, so per HARDENING.md discipline they were parked, not fixed
inline; this slice drains them. All five were re-verified against
current `main` before planning -- none has been fixed since.

## Scope (this PR)

Ownership lane: content-ops/reddit-listening/hardening
Slice phase: Production hardening

1. **User-Agent anchor trap** (#1940 item 4): `build_user_agent` uses
   `_USERNAME_RE.match()` whose `$` anchor accepts a trailing newline,
   letting a newline into the UA header (header-injection shape).
   `atlas_reddit/config.py` already uses `fullmatch` for the same trap on subreddit
   names. Fix: `fullmatch` + the trailing-newline negative probe.
2. **Intra-thread request burst** (#1940 item 1):
   `PrawHistorySource.fetch_thread_replies` runs refresh +
   `replace_more(limit=None)` per own comment with no pacing between
   them -- an unbounded burst the tracker's per-thread pace ceiling
   never sees. Fix: a named MoreComments budget (16, mirroring the
   existing top-level budget trade-off) and constructor-injected
   pacing applied between own-comment refreshes, wired from the CLI's
   `--pace-seconds`.
3. **Latent StopIteration** (#1940 item 2 / #1941 item 2):
   `track_once`'s per-thread re-read uses `next(...)` with no default;
   a row deleted underneath the pass (concurrent purge, future delete
   path) becomes an uncaught `StopIteration`. Fix: `None` default,
   record the error, skip dormancy for that thread, continue the pass.
4. **v1->v2 backfill NULL** (#1940 item 3): the migration backfills
   `last_activity` only from `replies`; a no-reply v1 thread migrates
   to `NULL` and is immediately marked dormant. Fix: `COALESCE` with
   the row's `last_checked`. In-place SQL fix is sufficient: no v1
   store has ever been migrated with the buggy SQL (credentials were
   never minted; no live stores exist).
5. **Irreversible purge on a false "missing"** (#1941 item 1): an item
   a silent-partial `info()` response fails to return is classified
   missing, purged, AND tombstoned -- so even if it was transient, the
   content can never re-ingest. Fix at the invariant level: the purge
   (fail-closed deletion) stands, but the tombstone -- the
   *irreversible* half -- now requires a confirmed deletion state
   (`[deleted]`/`[removed]` body or `removed_by_category`). A
   false-missing item re-ingests on its next listing appearance; a
   truly deleted item never reappears, so the wave-2 re-purge cycle
   stays closed. Schema v4 adds `purge_log.tombstone` (existing rows
   backfill to 1 -- conservative).

### Review Contract

- Acceptance criteria:
  - [ ] `build_user_agent("name\n")` raises; the existing valid case
        still passes.
  - [ ] Stub-praw probe: own-reply expansion is called with the bounded
        budget (never `limit=None`) and pacing sleeps run between
        own-comment refreshes (n-1 pattern, none before the first).
  - [ ] A tracked thread deleted mid-pass records an error and the pass
        completes (no StopIteration); remaining threads still evaluate.
  - [ ] v1 store with a no-reply thread migrates to
        `last_activity == last_checked` (not NULL); a thread with
        replies still backfills from `MAX(replies.created_utc)`.
  - [ ] Missing-classified purge deletes the row and logs, but the id
        re-ingests afterward; confirmed-deleted purge tombstones and
        re-ingestion stays refused. Both probed through the real store.
  - [ ] v3 store with existing purge_log rows opens at v4 with
        `tombstone=1` backfilled (still refuses re-ingestion).
- Affected surfaces: `atlas_reddit/reddit_client.py`, `atlas_reddit/tracker.py`,
  `atlas_reddit/store.py` (schema v4 + migration ladder),
  `atlas_reddit/purge.py`, `atlas_reddit/__main__.py` (pace wiring);
  tests.
- Risk areas: deletion/tombstone semantics (both directions probed);
  schema migration (ladder probed from v1 and v3).
- Reviewer rules triggered: R1, R2 (deletion-adjacent guard changes:
  both sides probed), R8 (replay/migration idempotence), R10, R11
  (zero new dependencies), R12 (tests run via the existing
  path-filtered glob workflow).
- Test-adapter posture (#1934 real-adapters rule): Reddit API faked at
  the source boundaries or stub-praw; real SQLite stores, real
  migration ladder, real CLI contracts.

### Files touched

- `atlas_reddit/__main__.py`
- `atlas_reddit/purge.py`
- `atlas_reddit/reddit_client.py`
- `atlas_reddit/store.py`
- `atlas_reddit/tracker.py`
- `plans/PR-Reddit-Listening-Hardening.md`
- `tests/test_atlas_reddit_poller.py`
- `tests/test_atlas_reddit_purge.py`
- `tests/test_atlas_reddit_tracker.py`

## Mechanism

The tombstone gate is the only design-level change. `fetch_gone_items`
already distinguishes confirmed deletion (content markers,
`removed_by_category`) from API absence; that distinction now travels
to the store: `purge_item(..., tombstone=...)` writes the flag,
`is_purged` consults only `tombstone=1` rows, and `atlas_reddit/purge.py` derives
the flag by comparing the reason against the shared
`MISSING_REASON` constant exported by `reddit_client`. Everything else
is a local guard: `fullmatch`, a bounded budget plus n-1 pacing inside
the history source, a `next(..., None)` default, and a `COALESCE` in
the v1->v2 backfill.

## Intentional

- **Missing items still purge.** Fail-closed local deletion is the
  compliance contract; only the *irreversibility* is now gated on
  confirmed evidence. This implements the audit's second suggested
  option; the first (returned-count < requested-count as inconclusive)
  is unusable because omission is exactly how `info()` reports
  deletion.
- **Bounded budget instead of `limit=0`**: the S5 docstring is right
  that `limit=0` silently discards direct children hidden behind
  placeholders; the budget (16, same named-trade-off pattern as the
  top-level scan) bounds the burst without that loss.
- **In-place v1->v2 SQL fix, no compensating v4 backfill**: no store
  was ever migrated with the buggy SQL (no credentials, no live data);
  a compensating update would be dead code guessing at data that
  cannot exist.
- **`thread` rows and own-content deletion stay out of scope** -- the
  S6 plan defers them; item 3's guard is what makes that future path
  safe to add.

## Deferred

- Producer-shape fixture assertions (the S6 P1 class) belong to the
  codification lane (#1947), not this slice.
- Scheduling and the LLM judge_fit pass remain operator decisions
  outside the arc (unchanged from the S6 plan).

Parked hardening: none.

## Verification

- pytest on `tests/test_atlas_reddit_purge.py`,
  `tests/test_atlas_reddit_tracker.py`, `tests/test_atlas_reddit_poller.py`,
  plus the untouched `tests/test_atlas_reddit_store.py`,
  `tests/test_atlas_reddit_digest.py`, `tests/test_atlas_reddit_config.py`,
  and `tests/test_atlas_reddit_scoring.py` -- full-suite count reported
  in the PR body (new probes: UA trailing-newline negative; bounded
  budget + n-1 pacing via stub praw; vanished-thread continuation;
  no-reply v1 backfill; missing-vs-confirmed tombstone both directions;
  v3->v4 backfill).
- ASCII byte-scan on the five changed Python files: clean.
- `python scripts/archive_plans.py index` run; INDEX unchanged (active
  plans are indexed on archive, not on creation).

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_reddit/__main__.py` | 4 |
| `atlas_reddit/purge.py` | 8 |
| `atlas_reddit/reddit_client.py` | 42 |
| `atlas_reddit/store.py` | 49 |
| `atlas_reddit/tracker.py` | 10 |
| `plans/PR-Reddit-Listening-Hardening.md` | 174 |
| `tests/test_atlas_reddit_poller.py` | 7 |
| `tests/test_atlas_reddit_purge.py` | 54 |
| `tests/test_atlas_reddit_tracker.py` | 124 |
| **Total** | **~472** |

Over the 400 soft cap: the plan doc plus mandated both-sides probes on
deletion-adjacent changes dominate; splitting five one-audit items into
two PRs would separate fixes from their probes for no review benefit.
The PR body carries the explicit diff-budget override.
