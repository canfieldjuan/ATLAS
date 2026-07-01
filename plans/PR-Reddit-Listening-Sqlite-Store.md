# PR-Reddit-Listening-Sqlite-Store

## Why this slice exists

Second slice (S2) of the approved #1934 arc (Fable 5 builder trial; product
epic #1931), continued under the approved arc-continuation protocol after S1
merged as 46de7ba45. S2 gives the tool its local, inspectable state: the
SQLite store and state model that S3 (digest) reads, S4 (poller) writes, S5
(reply tracker) drives, and S6 (purge) cleans. Deletion/purge fields ship
here, before any live ingestion exists, per the epic's compliance framing.
No network, no LLM.

This branch also carries one housekeeping commit: the merged S1 plan's
archive move (`plans/archive/`), folded into this slice because direct main
pushes are forbidden (#1934 comment 4860540364 step 3 authorizes exactly
this fold-in).

Diff-budget note: over the 400 LOC soft cap for the same structural reasons
as S1 -- the store is state-bearing and guard-shaped (status enums, replay
semantics, schema versioning), so the trial's adversarial-test rules require
per-branch negative fixtures, replay probes, and dual-layer enum enforcement
proofs. The code under test is ~430 LOC; the rest is mandated coverage.

## Scope (this PR)

Ownership lane: content-ops/reddit-listening/resolution-audit
Slice phase: Vertical slice

1. `atlas_reddit/store.py`: `ListeningStore` over stdlib sqlite3 with
   schema-versioned DDL (PRAGMA user_version), candidates / tracked
   threads / replies / purge-log tables, and fail-closed mutations.
2. Candidate lifecycle: replay-safe upsert (preserves `first_seen` and
   triage `status`), validated status enum (new/seen/dismissed/responded)
   enforced at both the API and a SQL CHECK constraint, ranked listing
   with status/min-score/limit filters.
3. Reply-tracker state: tracked threads with set-union comment-id merge and
   dormant flag preserved across replays; replies with INSERT OR IGNORE
   replay semantics and seen/unseen state.
4. Purge fields before live data: `purge_log` table + `record_purge`
   validated writer (the purge job itself is S6).
5. `db_path` typed setting (`ATLAS_REDDIT_DB_PATH`), defaulting under the
   gitignored data/ tree.
6. Housekeeping (separate first commit): archive the merged S1 plan doc and
   regenerate `plans/INDEX.md`.

### Review Contract

- Acceptance criteria:
  - [ ] Fresh store creates schema at version 1; reopening is idempotent;
        an unknown/newer user_version raises `StoreError` (fail closed)
        instead of misreading the file.
  - [ ] Candidate re-observation (replay) refreshes volatile fields and
        `last_seen` while preserving `first_seen` and `status`; no
        duplicate rows.
  - [ ] Invalid status values are rejected at the API (StoreError) AND by
        the CHECK constraint (IntegrityError on raw SQL) -- both layers
        probed.
  - [ ] Mutations on unknown ids (candidate status, thread dormant, reply
        seen) raise instead of silently no-oping.
  - [ ] Replayed reply inserts are ignored and report False; original row
        content wins.
  - [ ] Tracked-thread upserts merge comment ids as a stable-order set
        union and never resurrect a dormant thread.
  - [ ] Injection-shaped ids/titles (quotes, semicolons, SQL fragments) are
        stored literally; listing filters use an inclusive min-score
        boundary and a stable tiebreak.
  - [ ] New tests run in PR CI via the existing path-filtered glob workflow
        (tests/test_atlas_reddit_*.py) -- no workflow edit required.
- Affected surfaces: new store module + one settings field + tests; one
  plans/ housekeeping move. No existing Atlas surface touched.
- Risk areas: data-loss shape (replay overwriting triage state) and
  state-machine validity -- both covered by the replay/enum probes above.
  No network, no secrets, no migration of existing data (v1 creates from
  empty only).
- Reviewer rules triggered: R1, R2 (state store: failure branches, replay
  and wrong-transition fixtures), R4 (schema versioning fails closed; v1 is
  create-only, no destructive path), R8 (idempotent upserts + INSERT OR
  IGNORE replay semantics), R10, R11 (db_path typed field; zero new
  dependencies), R12 (CI enrollment via the existing glob).
- Test-adapter posture (#1934 real-adapters rule): zero mocks. Every test
  drives a real SQLite file under pytest tmp_path through the real
  `ListeningStore` API. The clock enters as explicit timestamp arguments,
  so no clock mocking exists either.

### Files touched

- `atlas_reddit/config.py`
- `atlas_reddit/store.py`
- `plans/INDEX.md`
- `plans/PR-Reddit-Listening-Sqlite-Store.md`
- `plans/archive/PR-Reddit-Listening-Config-Scoring.md`
- `tests/test_atlas_reddit_config.py`
- `tests/test_atlas_reddit_store.py`

## Mechanism

`ListeningStore(path)` connects, applies PRAGMA foreign_keys = ON, and runs
`_ensure_schema`: user_version 0 creates the four tables and stamps version
1; version 1 is a no-op; anything else raises `StoreError` so a file written
by a newer build is never misread. All statements are parameterized; rows
hydrate into frozen dataclasses (`Candidate`, `TrackedThread`, `Reply`,
`PurgeRecord`).

Candidates use INSERT ... ON CONFLICT(post_id) DO UPDATE that explicitly
excludes `status` and `first_seen` from the update set -- triage state
survives ingestion replays by construction. `set_candidate_status` validates
against the enum, and a zero UPDATE rowcount converts unknown ids into
errors. Listing orders by final_score DESC, created_utc DESC, post_id ASC
(deterministic tiebreak) with optional status / inclusive min-score / limit
filters.

Tracked threads merge `my_comment_ids` as JSON with order-stable dedupe
(dict.fromkeys) across upserts; `dormant` is deliberately not in the
conflict-update set. Replies use INSERT OR IGNORE and report whether the
row was inserted, so pollers can replay windows safely. `purge_log` accepts
only known item types and exists now so S4's live ingestion lands with the
compliance audit trail already in the schema.

## Intentional

- **Stdlib sqlite3, synchronous**: same rationale as S1 -- this is a local
  single-user CLI tool, not a brain service; the async-first convention
  targets `atlas_brain` I/O paths. Zero new dependencies.
- **JSON-in-TEXT for matched_topics / my_comment_ids** instead of join
  tables: single-user volumes, human-inspectable with the sqlite3 CLI, and
  the consumers (digest, reply tracker) always read the whole list. A join
  table is an abstraction larger than the problem (R10).
- **Permissive status transitions** (any valid value to any valid value):
  triage is human-driven -- dismiss then un-dismiss, or responded straight
  from new, are legitimate flows. A transition graph would fight the S3/S5
  flows for no safety gain; validity of the *values* is what the enum and
  CHECK constraint guarantee.
- **Timestamps as explicit arguments** rather than an injected clock object:
  the simplest honest seam. Tests pass literal ints; production callers (S4)
  will pass the wall clock at their boundary.
- **`_ensure_schema` runs in the constructor**, not lazily per call: one
  obvious failure point at open time, and the version check cannot be
  skipped by an unusual call order.
- **Purge job deferred to S6 by design** (arc decision): only the fields and
  the validated log writer land here, satisfying "purge fields exist before
  live ingestion" without pulling S6's scope forward.
- **Housekeeping commit in this branch**: authorized fold-in (#1934 comment
  4860540364) because direct main pushes are forbidden; kept as its own
  commit so the S2 code review is not polluted by the plans/ move.

Review-fix notes (Codex wave 1 on 9d84228ce; all three verified real and
fixed at root in this PR):

- **SQLite TEXT PRIMARY KEY does not imply NOT NULL** (P2): a None id could
  insert anonymous rows that replays cannot dedupe and lookups cannot find.
  Root cause: relying on the PK to imply NOT NULL (a SQLite-specific trap).
  Fixed in the DDL (explicit NOT NULL on all three id PKs) plus API-level
  non-empty-string id validation, which also rejects the unseen class case
  the schema cannot catch: the empty string. Schema stays version 1 -- it is
  unreleased (no store file exists outside test tmp dirs), so this is a
  pre-release DDL correction, not a migration.
- **INSERT OR IGNORE masked every integrity failure as a replay** (P1,
  silent data loss): a malformed reply (NULL created_utc, unknown thread)
  returned False as if it were a duplicate. Root cause: blanket conflict
  suppression where only duplicate-key suppression was intended. Fixed with
  a narrow ON CONFLICT(reply_id) DO NOTHING; both sides probed (true replay
  still returns False; integrity violations raise).
- **Replies could reference untracked threads** (P2): orphan rows drift
  outside the tracked-thread lifecycle that S5/S6 drive. Root cause: missing
  referential constraint. Fixed with a foreign key to
  tracked_threads(thread_id) (foreign_keys pragma already ON per
  connection); tests register threads before replies, mirroring the real S5
  discovery order.

## Deferred

- S3 Markdown digest + `python -m atlas_reddit` CLI (first consumer of
  `list_candidates`).
- S4 PRAW read-only poller (first producer; doc-verify auth path first;
  read/identity/history fail-closed; passes real timestamps and post ids
  into this store).
- S5 reply tracker read path (consumes tracked_threads/replies helpers).
- S6 deletion-compliance purge job (consumes purge_log; adds the actual
  row-removal path and the 48h window logic).
- WAL mode / performance pragmas: single-user daily volumes do not justify
  them yet; revisit only if S4 shows contention.

Parked hardening: none.

## Verification

- pytest on `tests/test_atlas_reddit_store.py`,
  `tests/test_atlas_reddit_config.py`, and
  `tests/test_atlas_reddit_scoring.py`: 148 passed (S1 suite unchanged plus
  the store suite: schema create/reopen/fail-closed-version, replay
  preservation probes, dual-layer enum enforcement, unknown-id fail-closed
  mutations, replayed reply ignores, dormant preservation, set-union merge,
  injection-shaped input, unicode roundtrip, inclusive boundary + stable
  tiebreak, purge-log validation both sides, db_path setting default and
  override; wave-1 class probes: None and empty-string ids across all four
  write paths, NULL-PK schema-layer rejection, malformed-reply-raises vs
  true-replay-False both sides, untracked-thread FK rejection). 148 passed
  after the wave-1 review fixes. This line is the single verification-count
  source; the PR body mirrors it.
- ASCII byte-scan on the four changed Python files: clean.
- python `scripts/sync_pr_plan.py` on this plan: Files touched + diff table
  regenerated from the real diff.
- Local review bundle runs via `scripts/push_pr.sh` before push.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_reddit/config.py` | 8 |
| `atlas_reddit/store.py` | 500 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Reddit-Listening-Sqlite-Store.md` | 216 |
| `plans/archive/PR-Reddit-Listening-Config-Scoring.md` | 0 |
| `tests/test_atlas_reddit_config.py` | 9 |
| `tests/test_atlas_reddit_store.py` | 483 |
| **Total** | **1219** |
