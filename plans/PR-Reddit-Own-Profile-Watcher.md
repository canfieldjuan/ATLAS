# PR-Reddit-Own-Profile-Watcher

## Why this slice exists

The operator asked for the Reddit tool to be pointed at their own profile
("home page") -- the one listing that already carries every post they made
and the subreddit each landed in -- so all their posts live in one local
place and can be read individually. Today the package cannot do that: the
radar (`poll`) watches watchlist subreddits for OTHER people's posts, and
the reply tracker (`track`) sees the operator's own submissions only as
skinny thread ids for reply discovery -- no title, no subreddit, no body.
Root cause: no producer method fetches the operator's own submission
listing with content fields, and no table holds it. This slice adds that
vertical: a read-only profile listing fetch (same identity/history/read
scopes the tracker already requires) -> a new `own_posts` table -> CLI
commands to sync the profile, list all own posts, and read one post
together with its tracked replies.

The write half of the operator's ask (replying to and editing those posts
via the API) is deliberately the NEXT slice: it reverses the package's
enforced read-only contract (scope ceiling, static no-write probe, package
docstring, runbook) and cannot function until the operator re-mints the
refresh token with write scopes -- so it is sequenced behind this read
slice, where each contract change is reviewable on its own.

Diff-budget overage (if any): the producer method, the table + migration,
the sync pass, and the CLI read surface are one indivisible vertical -- a
sync with no read surface delivers nothing user-visible, and a read
surface without the table reads nothing. Tests follow the #1947
producer-fidelity factory discipline rather than hand-seeded shapes.

## Scope (this PR)

Ownership lane: reddit-listening-tool
Slice phase: Vertical slice

1. `atlas_reddit/reddit_client.py`: `ProfileSource` protocol
   (`fetch_my_posts`) + implementation on `PrawHistorySource` -- the rich
   `ListingPost` mapping over `me.submissions.new()`, subreddit taken from
   the submission's own `subreddit.display_name` (a profile listing spans
   subreddits, unlike `fetch_new` where the caller names one).
2. `atlas_reddit/store.py`: schema v5 -- `own_posts` table (+ subreddit
   index), `OwnPost` row type, `upsert_own_post` / `get_own_post` /
   `list_own_posts`, and an additive v4 -> v5 migration on the existing
   ladder (main reached v4 via #1948's purge tombstone while this slice
   was in flight, so own_posts stacks as v5). Upsert is replay-safe the
   same two ways as candidates: `first_seen` preserved on conflict, and a
   stale (out-of-order) observation updates nothing.
3. `atlas_reddit/profile.py`: `sync_profile_once` -- fetch own posts,
   upsert each, report new/refreshed/error stats. No admission filters:
   unlike the radar, every own post belongs in "one place where all of it
   lives" (link posts included).
4. `atlas_reddit/__main__.py`: three commands -- `profile` (sync pass,
   network), `posts` (list from local state, offline; defaults to ALL
   synced posts, no silent newest-N cap, so "all my posts" is literal),
   `post <id>` (one post + its tracked replies, offline; accepts bare id
   or t3_ fullname).
5. `atlas_reddit/config.py`: `MAX_PROFILE_LIMIT` (1000, Reddit's listing
   ceiling) + a `profile_limit` setting defaulting to it, so the profile
   sync backfills all reachable own posts rather than the tracker's tight
   100-item history window (review finding: "all my posts" must not stop
   at 100).
6. `tests/atlas_reddit_fixtures.py`: `fake_submission` gains a
   `subreddit` attribute, `real_history_source` gains `own_submissions`
   (closing the deferred own-submission gap noted in its docstring), and
   `seed_own_posts` runs the real producer + sync pipeline.
7. `tests/test_atlas_reddit_profile.py`: producer-fidelity sync tests +
   store failure branches + migration + CLI wiring.
8. `docs/REDDIT_LISTENING_SETUP_RUNBOOK.md`: section 4 command list gains
   the three new commands, the profile listing-ceiling note, the
   own_posts purge-exclusion clarification, and a reply-coverage note
   (review findings) -- `post` shows tracked replies only for submissions
   the tracker has already discovered in its history window, since profile
   sync deliberately does not enqueue own posts into `tracked_threads`.

### Files touched

- `atlas_reddit/reddit_client.py`
- `atlas_reddit/store.py`
- `atlas_reddit/config.py`
- `atlas_reddit/profile.py` (new)
- `atlas_reddit/__main__.py`
- `tests/atlas_reddit_fixtures.py`
- `tests/test_atlas_reddit_profile.py`
- `tests/test_atlas_reddit_tracker.py`
- `tests/test_atlas_reddit_purge.py`
- `docs/REDDIT_LISTENING_SETUP_RUNBOOK.md`
- `plans/PR-Reddit-Own-Profile-Watcher.md`

The tracker-test and purge-test edits are consequences of the schema bump
landing on top of main's v4: each migration probe's terminal schema
version now follows SCHEMA_VERSION (the tracker probe also asserts the
own_posts rung created the table), and the tracker's HistorySource
public-surface probe gains the new profile fetch method.

### Review Contract

Acceptance criteria (reviewer checks one-by-one):

1. The new producer surface is read-only: `fetch_my_posts` only reads
   listings; the static no-write probe
   (`test_no_reddit_write_calls_anywhere`) and the public-surface probes
   still pass unchanged.
2. `upsert_own_post` is replay-safe: `first_seen` survives re-sync, and a
   stale `observed_at` (older than the stored `last_seen`) regresses
   nothing -- both covered by failure-branch tests, plus StoreError
   branches for malformed ids/ints.
3. A v3 store opens and walks the ladder to the current version (v4
   tombstone, then v5 own_posts; existing rows untouched, `own_posts`
   created); a NEWER schema version still fails closed. Covered by tests.
4. `posts` and `post` read only local state (no praw import on those
   paths); an unknown post id exits 2 naming the id.
5. Sync tests go through the REAL producer mapping via the fixture
   factory (praw-shaped stubs; no hand-seeded id shapes), per the #1947
   discipline.
6. `python -m pytest tests/test_atlas_reddit_profile.py
   tests/test_atlas_reddit_store.py tests/test_atlas_reddit_poller.py
   tests/test_atlas_reddit_tracker.py tests/test_atlas_reddit_fixture_fidelity.py -q`
   passes.

Affected surfaces: the atlas_reddit package + its tests + one runbook
section. No atlas_brain code, no gate scripts, no workflows.

Risk areas: schema migration on the shared SQLite state file (additive
only -- new table + index; the ladder pattern and fail-closed
newer-version branch are already established); the profile sync spends one
PRAW listing per run, which PRAW paginates in 100-item batches up to
`MAX_PROFILE_LIMIT` (1000) -- run occasionally, not per-subreddit, so it
is not bound by the radar's tight request budget.

Reviewer rules triggered: R10 (schema/persistence change ->
failure-branch fixtures per AGENTS.md 3h/3i), R14 (checked-out PR-head
verification).

## Mechanism

`me.submissions.new()` IS the profile page's submission feed: the
authenticated user's posts, newest first, each carrying the subreddit it
was posted in. `PrawHistorySource` already authenticates with exactly the
scopes this needs (identity to resolve `me`, history for own listings,
read), so pointing the tool at the "home page" is a new mapping over an
already-authorized listing -- no new credentials, no scope change.
`sync_profile_once` upserts the mapped posts into `own_posts`, keyed by
fullname like every other stored id in the package, so a post row joins
its replies (tracked by `track`, keyed by the same `t3_` thread fullname)
with a plain equality -- `post <id>` renders the post plus its stored
replies without touching the network. Reply collection itself stays the
tracker's job: single writer per table, and the tracker already handles
top-level comments on own submissions (`include_top_level=True`).

## Intentional

- **The subreddit radar stays.** `poll`, the watchlist, and scoring are
  untouched -- this slice repoints the tool's center of gravity to the
  operator's profile without deleting the listening radar. If the radar
  should retire once the profile surface proves out, that is its own
  slice (named in Deferred), not a silent removal here.
- **Own posts are admitted unfiltered.** The radar's is_self/freshness/
  score filters exist to triage strangers' posts; "all of the posts" from
  the operator's profile means link posts and old posts belong too.
- **Profile sync does not write `tracked_threads`.** The tracker already
  discovers own submissions from history and owns that table's lifecycle
  (dormancy, wake). Two writers would race the dormancy state machine.
- **Purge scope unchanged; own_posts retention documented.** The 48h
  deletion-compliance contract exists for third-party content (other
  people's posts and replies); `own_posts` rows are the operator's own
  words, so `purge` does not touch them and a post deleted on Reddit
  stays in the local mirror. The runbook now says this explicitly (review
  finding: the deletion-compliance paragraph must not imply it covers
  own posts).
- **The read-only contract is intact this slice.** ALLOWED_SCOPES, the
  static no-write probe, and the package docstring are unchanged.

## Deferred

- **Write surface (the operator's slice 2):** reply to comments on own
  posts and edit own post bodies via the API. Requires the operator to
  re-mint the refresh token with `submit` + `edit` scopes; in code it
  means an explicit write client behind its own scope floor, a widened
  ceiling, reworking the static no-write probe into an allowlist over the
  write module, and runbook + package-docstring updates. Deliberately its
  own reviewable slice.
- Radar retirement decision (delete `poll`/watchlist/scoring) once the
  profile surface is the proven center.
- Digest section for own posts (the digest currently renders radar
  candidates + warm replies only).
- Factory adoption in the remaining reddit test files (#1947 Deferred,
  narrowed here: `real_history_source` now covers own submissions).

Parked hardening: none new this slice.

## Verification

Commands run from the repo root:

- `python -m pytest tests/test_atlas_reddit_profile.py
  tests/test_atlas_reddit_store.py tests/test_atlas_reddit_poller.py
  tests/test_atlas_reddit_tracker.py tests/test_atlas_reddit_fixture_fidelity.py
  tests/test_atlas_reddit_digest.py tests/test_atlas_reddit_purge.py
  tests/test_atlas_reddit_config.py -q` -- pass count recorded in the PR
  body (the purge/config suites cover the schema-ladder terminal version
  and the profile-limit ceiling).
- `scripts/check_ascii_python.sh` (via bash) -- passes.
- `bash scripts/local_pr_review.sh --current-pr-body-file <pr-body.md>`
  -- all checks PASS.
- `python scripts/check_diff_budget.py --additions <n> --body-file
  <pr-body.md>` -- recorded in the PR body.
- Operator smoke (with creds in env): `python -m atlas_reddit profile`
  then `python -m atlas_reddit posts` and
  `python -m atlas_reddit post <id>` against the live profile.

## Estimated diff size

| File | LOC (added) |
|---|---:|
| `atlas_reddit/__main__.py` | 151 |
| `atlas_reddit/store.py` | 151 |
| `atlas_reddit/profile.py` | 70 |
| `atlas_reddit/reddit_client.py` | 37 |
| `atlas_reddit/config.py` | 15 |
| `tests/test_atlas_reddit_profile.py` | 339 |
| `tests/atlas_reddit_fixtures.py` | 40 |
| `tests/test_atlas_reddit_tracker.py` | 4 |
| `tests/test_atlas_reddit_purge.py` | 2 |
| `docs/REDDIT_LISTENING_SETUP_RUNBOOK.md` | 31 |
| `plans/PR-Reddit-Own-Profile-Watcher.md` | 222 |
| **Total** | **1062** |
