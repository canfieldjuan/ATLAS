# PR-Reddit-Listening-Purge

## Why this slice exists

Sixth and FINAL slice of the approved #1934 arc (Fable 5 builder trial;
product epic #1931), continued under the arc-continuation protocol after S5
merged as 1c8d11216. S6 closes the epic's compliance contract: stored
third-party content that is later deleted or removed on Reddit (or missing
entirely) must be purged locally within the 48-hour target and recorded.
S2 shipped the purge fields before any live ingestion existed; this slice
ships the purge JOB on top of them, completing the arc's read-only,
compliance-first posture. After this slice merges the approved arc is
complete and the builder stops for the operator's trial review.

This branch carries one housekeeping commit (merged S5 plan archive +
INDEX rebuild) per #1934 comment 4860540364 step 3.

Diff-budget note: modestly over the soft cap for the arc's standing
reasons -- the purge is a deletion guard, so the trial rules require
both-sides probes (live content survives, gone content is ACTUALLY gone),
replay idempotence, batching, and error containment.

## Scope (this PR)

Ownership lane: content-ops/reddit-listening/resolution-audit
Slice phase: Production hardening

1. `atlas_reddit/purge.py`: purge_once -- every stored candidate post and
   reply is re-checked through the DeletionSource boundary in batches of
   100 (one reddit info() read per batch), gone rows are deleted, and
   each purge is recorded in purge_log with the detection reason.
2. `atlas_reddit/reddit_client.py`: DeletionSource protocol +
   PrawDeletionSource (same constructor discipline as the other sources;
   requires only the read scope; batched info() with deleted/removed/
   missing classification).
3. `atlas_reddit/store.py`: purge_candidate / purge_reply row deletes
   (validated ids, rowcount semantics; the caller records the log entry).
4. CLI `python -m atlas_reddit purge` with the shared pace ceiling and the
   established operator-error contract; exit 1 on partial-pass errors.
5. Runbook cadence note: run purge at least every 48 hours while stored
   content exists.
6. Housekeeping (separate first commit): archive the merged S5 plan doc
   and regenerate `plans/INDEX.md`.

### Review Contract

- Acceptance criteria:
  - [ ] Both sides probed: live candidates and replies survive a purge
        pass; deleted/removed/missing items are ACTUALLY deleted from the
        database and logged with reason, type, and timestamps.
  - [ ] Purged content disappears from the digest end-to-end (real
        writer).
  - [ ] A second pass is idempotent: nothing left to check, no duplicate
        purge_log entries.
  - [ ] Batching respects the 100-fullname info() limit with n-1 pacing;
        a failed batch is recorded and the remaining batches still run.
  - [ ] Tracked-thread rows (ids only, no third-party content) are
        retained even when all their replies purge.
  - [ ] PrawDeletionSource requires only the read scope, fails closed on
        missing credentials, and exposes a read-only public surface; the
        package-wide forbidden-write static probe stays green.
  - [ ] CLI purge honors the operator-error contract and the shared pace
        ceiling.
  - [ ] New tests run in PR CI via the existing path-filtered glob
        workflow.
- Affected surfaces: new purge module, DeletionSource additions, two store
  delete methods, one CLI subcommand, runbook note; one plans/
  housekeeping move. No schema change (purge_log shipped in S2).
- Risk areas: data deletion (guard-shaped -- both error directions
  probed); no network writes (reddit info() is a read; deletion happens
  only in the local database).
- Reviewer rules triggered: R1, R2 (deletion guard: both sides + replay +
  containment fixtures), R3 (read-scope floor), R8 (idempotent replay of
  the purge pass), R10, R11 (zero new dependencies), R12 (CI enrollment
  via the existing glob).
- Test-adapter posture (#1934 real-adapters rule): the Reddit API is
  faked at the DeletionSource boundary; everything else is real (real
  SQLite stores seeded through real APIs, the real digest writer, real
  CLI main() in-process). Clock and sleep enter as arguments.

### Files touched

- `atlas_reddit/__main__.py`
- `atlas_reddit/purge.py`
- `atlas_reddit/reddit_client.py`
- `atlas_reddit/store.py`
- `docs/REDDIT_LISTENING_SETUP_RUNBOOK.md`
- `plans/INDEX.md`
- `plans/PR-Reddit-Listening-Purge.md`
- `plans/archive/PR-Reddit-Listening-Reply-Tracker.md`
- `tests/test_atlas_reddit_purge.py`

## Mechanism

purge_once collects every stored candidate post_id and reply reply_id,
sorts them for determinism, and walks them in batches of 100 (the
documented reddit info() ceiling) with the injectable n-1 pacing.
PrawDeletionSource starts from "everything in the batch is missing" and
removes items the API returns as present-and-live; an item whose author is
gone and whose body/selftext shows the deleted/removed markers, or that
carries removed_by_category, stays in the gone map with a specific reason.
For each gone item the store row is deleted (purge_candidate/purge_reply,
validated ids, rowcount semantics) and a purge_log entry records item,
type, reason, and both timestamps. A batch-level fetch failure is recorded
in stats and the remaining batches still run -- a partial compliance pass
beats none, and the failure stays visible via exit 1.

## Intentional

- **Rows are deleted, not blanked**: the compliance contract is "the
  content is gone", and a blanked row is content-shaped residue. The
  purge_log (ids + reason only, no content) is the audit trail.
- **Start-from-missing classification**: items the API does not return
  are treated as gone by default -- fail-closed in the compliance
  direction; the live path is the one that must prove itself.
- **Tracked-thread rows retained**: they hold only the thread id and the
  operator's own comment ids -- no third-party content to purge.
- **The 48h window is operational cadence, not code**: a manual-run tool
  cannot schedule itself; the runbook states the cadence, and the epic's
  acceptance (deletion cleanup exists before long-lived storage beyond
  proof usage) is satisfied by this job existing and being idempotent.
- **Read scope only** for the deletion source: checking liveness is a
  read; deletion happens exclusively in the local database (the static
  no-write probe covers the new module automatically).
- **Housekeeping commit in this branch**: authorized fold-in (#1934
  comment 4860540364).

Review-fix notes (Codex wave 1 on a6ad25736; all three verified real and
fixed at root in this PR):

- **P1: candidates stored bare submission ids while info() requires
  fullnames** -- on first real use every live candidate would have been
  classified missing and mass-purged. Root cause: inconsistent id
  formats across the pipeline (replies/threads stored fullnames,
  candidates did not), invisible to the purge tests because the fakes
  seeded fullname-shaped ids -- fixture-vs-producer drift. Fixed at the
  producer (the S4 poller mapping now stores submission.fullname; no
  live data exists, creds were never minted) PLUS a defensive purge-side
  shape guard: ids that are not valid fullnames are surfaced as errors
  and RETAINED -- fail-closed must never extend to "our own data shape
  is wrong". Probed at both layers, including a stub-praw test of the
  REAL producer mapping (the probe that would have caught this).
- **Removed comments with a surviving author classified live**: mod
  removal can keep the author object while the body shows [removed].
  Root cause: classification coupled to the author instead of the
  content state. Fixed: body/selftext markers and removed_by_category
  decide, author-independent; the second side is probed too
  (account-deleted author with intact content is NOT deleted content
  and stays).
- **Runbook ordered digest before purge**, so a digest could surface
  content deleted since the last pass. Fixed: purge runs before digest
  in the documented flow, with the reason stated inline.

Review-fix notes (Codex wave 2 on d10f32587; all four verified real and
fixed at root in this PR):

- **The shape guard admitted wrong-KIND fullnames** (t2_ user ids in
  candidates would still be purged as missing). Root cause: the wave-1
  guard validated "is a fullname" but not "is the right kind for its
  table". Fixed per table (candidates t3_, replies t1_); wrong-kind rows
  are retained and surfaced.
- **Delete and audit log were separate transactions**: an I/O error or
  interrupt between them deletes content with no record. Root cause:
  action and audit in different transactions. Fixed with an atomic
  store.purge_item (single transaction; the log entry exists IFF the row
  was deleted; probed from both directions).
- **Rendered digest files retained purged content indefinitely**: local
  compliance covers artifacts, not just rows. Fixed: a pass that purged
  content also removes existing digest files (regenerable projections of
  the now-clean store); a zero-purge pass -- which proves store and
  artifacts consistent -- leaves them. Both sides probed.
- **Purged ids could resurrect via re-ingestion**: Reddit keeps returning
  removed items in listings, and ingestion never consulted the log.
  Root cause: purge_log was audit-only. Fixed: it now tombstones --
  upsert_candidate and insert_reply refuse previously purged ids, ending
  the re-ingest/re-purge cycle.

Review-fix notes (Codex wave 3 on 34273cd75; both verified real and fixed
at root in this PR):

- **Union-level malformed subtraction shielded cross-table id twins**: a
  corrupt reply holding a t3_ id suppressed the legitimate candidate with
  the same id from ever being purged. Root cause: flattening ids into a
  union loses table scoping -- the same flattening behind the wave-2 kind
  bug, one level up. Valid sets now stay per-table end-to-end; probed
  with the twin scenario (candidate purges, corrupt reply retained).
- **Digest cleanup was keyed to a transient signal** (rows purged this
  pass): a failed unlink would never retry once the rows were committed.
  Root cause: artifact cleanup not derived from persisted state. Fixed:
  any digest file whose mtime predates the newest purge_log entry is
  removed; per-file failures are surfaced and retried naturally on the
  next pass; digests rendered after the latest purge are kept. Probed
  with a transient-failure-then-retry sequence and the fresh-digest
  survival case.

## Deferred

- Scheduling (cron/autonomous task) for poll/track/digest/purge: beyond
  the arc's manual-run boundary; the LLM judge_fit pass is beyond the
  approved arc entirely. Both are operator decisions after the trial
  review.
- Purge of the operator's own comment ids on request (account-owner
  deletion of their own data is a different contract from third-party
  deletion compliance; not in the epic's v1).

Parked hardening: none.

## Verification

- pytest on `tests/test_atlas_reddit_purge.py` plus the existing
  `tests/test_atlas_reddit_tracker.py`, `tests/test_atlas_reddit_poller.py`,
  `tests/test_atlas_reddit_digest.py`, `tests/test_atlas_reddit_store.py`,
  `tests/test_atlas_reddit_config.py`, and
  `tests/test_atlas_reddit_scoring.py`: 313 passed (both-sides deletion
  probes, digest disappearance end-to-end, replay idempotence with no
  duplicate log entries, 100-item batching with n-1 pacing, failed-batch
  containment with continued purging, tracked-thread retention,
  read-scope floor, read-only surface, CLI no-creds and pace-ceiling
  exits; wave-1 probes: malformed-id retained-not-missing, REAL producer
  mapping stores fullnames via stub praw, removed-body-with-author gone,
  author-deleted-content-intact live, absent-from-info missing; wave-2
  probes: wrong-kind-per-table retained, purge_item atomicity both
  directions, tombstoned re-ingestion refused on both write paths,
  digest artifacts removed-from-persisted-state/kept-when-clean; wave-3
  probes: cross-table twin not shielded, unlink-failure retry, fresh
  post-purge digest survival). This line is the single verification-count
  source; the PR body mirrors it.
- ASCII byte-scan on the five changed Python files: clean.
- python `scripts/sync_pr_plan.py` on this plan: tables regenerated from
  the real diff.
- Local review bundle runs via `scripts/push_pr.sh` before push.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_reddit/__main__.py` | 62 |
| `atlas_reddit/purge.py` | 132 |
| `atlas_reddit/reddit_client.py` | 76 |
| `atlas_reddit/store.py` | 58 |
| `docs/REDDIT_LISTENING_SETUP_RUNBOOK.md` | 9 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Reddit-Listening-Purge.md` | 246 |
| `plans/archive/PR-Reddit-Listening-Reply-Tracker.md` | 0 |
| `tests/test_atlas_reddit_purge.py` | 546 |
| **Total** | **1132** |
