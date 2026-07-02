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
  `tests/test_atlas_reddit_scoring.py`: 300 passed (both-sides deletion
  probes, digest disappearance end-to-end, replay idempotence with no
  duplicate log entries, 100-item batching with n-1 pacing, failed-batch
  containment with continued purging, tracked-thread retention,
  read-scope floor, read-only surface, CLI no-creds and pace-ceiling
  exits). This line is the single verification-count source; the PR body
  mirrors it.
- ASCII byte-scan on the five changed Python files: clean.
- python `scripts/sync_pr_plan.py` on this plan: tables regenerated from
  the real diff.
- Local review bundle runs via `scripts/push_pr.sh` before push.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_reddit/__main__.py` | 51 |
| `atlas_reddit/purge.py` | 84 |
| `atlas_reddit/reddit_client.py` | 68 |
| `atlas_reddit/store.py` | 23 |
| `docs/REDDIT_LISTENING_SETUP_RUNBOOK.md` | 6 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Reddit-Listening-Purge.md` | 154 |
| `plans/archive/PR-Reddit-Listening-Reply-Tracker.md` | 0 |
| `tests/test_atlas_reddit_purge.py` | 249 |
| **Total** | **638** |
