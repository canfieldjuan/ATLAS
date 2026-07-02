# PR-Reddit-Listening-Reply-Tracker

## Why this slice exists

Fifth slice (S5) of the approved #1934 arc (Fable 5 builder trial; product
epic #1931), continued under the arc-continuation protocol after S4 merged
as 8d269b8f0. S5 is the Reply Tracker: the tool's second job per the epic --
auto-discover the threads Juan participated in from his own read-only
history, store direct replies with unread state, and light up the digest's
warm-replies section so follow-up happens while threads are still warm.
No posting, no drafting; read-only scopes throughout, with the history
source requiring the full read/identity/history floor via the S4 guard's
required-scopes override.

This branch carries one housekeeping commit (merged S4 plan archive +
INDEX rebuild) per #1934 comment 4860540364 step 3.

Diff-budget note: over the 400 LOC soft cap for the arc's standing
reasons -- the dormancy lifecycle is state-machine-shaped, so the trial
rules require both-sides probes (quiet-sleeps, fresh-reply-keeps-awake,
stale-reply-does-not, dormant-not-polled, fresh-activity-wakes,
stale-rediscovery-does-not-wake). Code under test is ~330 LOC; the rest
is mandated coverage and this plan.

## Scope (this PR)

Ownership lane: content-ops/reddit-listening/resolution-audit
Slice phase: Vertical slice

1. `atlas_reddit/reddit_client.py` additions: OwnActivity/ThreadReply
   shapes, HistorySource protocol, and PrawHistorySource (lazy praw,
   check_for_updates=False, whole-open-path RedditAuthError wrap, scope
   floor read+identity+history, read-only public surface).
2. `atlas_reddit/tracker.py`: track_once -- discover own threads from
   recent comments/submissions (set-union comment-id merge), fetch
   replies on ACTIVE threads with pacing and per-thread error isolation,
   replay-safe reply inserts, and the dormancy lifecycle: a thread
   sleeps when its newest known activity is older than the quiet window
   (or when no activity timestamp is known at all -- exactly the case
   where engagement aged out of the history window); fresh own activity
   inside the window is the only wake signal; dormant threads are not
   polled.
3. CLI `python -m atlas_reddit track` (validated knobs, same
   exit-2-stderr contract, exit 1 on partial-pass errors) and
   `mark-read <reply_id>` (drops a reply from the digest).
4. Settings: `history_limit` and `dormant_after_hours` with shared MAX_*
   ceilings consumed by both the pydantic bounds and the CLI checks
   (the S4 two-entry-path lesson applied from the start).
5. Housekeeping (separate first commit): archive the merged S4 plan doc
   and regenerate `plans/INDEX.md`.

### Review Contract

- Acceptance criteria:
  - [ ] Threads auto-discovered from own comments AND submissions;
        rediscovery grows my_comment_ids as a set union without
        duplicate thread rows.
  - [ ] Replies insert replay-safe (second pass: zero new, counted as
        replayed); unseen replies-to-me appear in the S3 digest's
        warm-replies section with zero renderer changes.
  - [ ] Dormancy both sides: quiet thread sleeps; fresh reply keeps it
        awake; stale replies do not; dormant threads are not polled;
        fresh own activity wakes them; STALE rediscovery does not wake
        them.
  - [ ] PrawHistorySource requires the full read/identity/history floor
        (probed via the pure guard) and exposes a read-only public
        surface; the package-wide forbidden-write static probe stays
        green over the new modules.
  - [ ] CLI track/mark-read honor the operator-error contract; all knob
        ceilings enforced on the CLI path.
  - [ ] New tests run in PR CI via the existing path-filtered glob
        workflow.
- Affected surfaces: reddit_client additions, new tracker module, two CLI
  subcommands, two settings fields; one plans/ housekeeping move. No
  existing Atlas surface touched.
- Risk areas: auth floor for the wider scopes (probed); dormancy
  state-machine correctness (both-sides probes); no schema change (S2
  tables already carry tracked_threads/replies); live network behavior
  deferred to the operator runbook smoke.
- Reviewer rules triggered: R1, R2 (state-machine both-sides fixtures +
  replay probes), R3 (scope floor for wider grants; SecretStr unchanged),
  R8 (replay-safe inserts re-proven through the tracker path), R10, R11
  (two typed settings fields; zero new dependencies), R12 (CI enrollment
  via the existing glob).
- Test-adapter posture (#1934 real-adapters rule): the Reddit API is
  faked at the HistorySource protocol; everything else is real (real
  SQLite stores, the real digest writer proving the warm-replies flow,
  real CLI main() in-process). Clock and sleep enter as arguments.

### Files touched

- `atlas_reddit/__main__.py`
- `atlas_reddit/config.py`
- `atlas_reddit/reddit_client.py`
- `atlas_reddit/tracker.py`
- `plans/INDEX.md`
- `plans/PR-Reddit-Listening-Reply-Tracker.md`
- `plans/archive/PR-Reddit-Listening-Praw-Poller.md`
- `tests/test_atlas_reddit_tracker.py`

## Mechanism

track_once runs three phases. Discovery: recent own comments and
submissions come through the HistorySource protocol; activities group by
thread (comment fullnames collected per thread; submissions track the
thread itself) and upsert through the S2 store's set-union merge, so
known comment ids never drop. A thread that was dormant wakes only if
the rediscovered activity is inside the quiet window -- old activity
scrolling back into the history window cannot resurrect it. Reply fetch:
ACTIVE threads only, paced n-1 with the injectable sleep; each thread's
replies insert via the S2 replay-safe path (duplicate reply ids ignored
and counted, integrity violations raise); one failing thread lands in
stats.errors without aborting the pass. Dormancy: after fetching, the
newest known activity for a thread is max(own activity seen this pass,
newest stored reply); if that is older than the window -- or unknown
entirely -- the thread sleeps.

PrawHistorySource mirrors the S4 constructor discipline (credential
check before the lazy praw import, whole praw block wrapped into
RedditAuthError, scope validation immediately after) and adds the
identity lookup (reddit.user.me()) inside the same contract. Reply
extraction walks the submission's comment forest (replace_more(limit=0)),
keeps comments whose parent is one of the operator's comment fullnames
(direct replies) or the thread itself when the thread is the operator's
submission (top-level responses), and skips the operator's own comments.

## Intentional

- **Dormancy = stop polling until re-engagement**: a reply arriving on a
  dormant thread is not observed until the operator engages again
  (rediscovery wake). This is the deliberate quiet-period contract from
  the epic -- polling every historical thread forever is exactly the
  scaled-extraction shape the tool must not have. Named trade-off, not
  an oversight.
- **Wake requires IN-WINDOW activity**: without the freshness condition,
  any old comment scrolling back into the history fetch window would
  resurrect sleeping threads every pass, defeating dormancy entirely
  (probed by the stale-rediscovery test).
- **Submissions tracked with empty comment-id lists**: top-level replies
  on own posts are matched by parent == thread id at the source, so no
  synthetic comment ids are invented.
- **mark-read as a CLI command** (not a digest-side effect): the digest
  stays a pure projection; state changes are explicit operator actions.
- **Knob ceilings shared from the start** (MAX_HISTORY_LIMIT,
  MAX_DORMANT_AFTER_HOURS): the S4 wave-1/wave-2 two-entry-path class
  applied proactively rather than waiting for the bot to find it.
- **Housekeeping commit in this branch**: authorized fold-in (#1934
  comment 4860540364).

## Deferred

- S6 deletion-compliance purge job (last arc slice; purge_log ready).
- Live-credential smoke: same operator runbook posture as S4.
- Unread-count badges, reply threading depth, and any notification
  delivery: beyond the arc's manual-run boundary.

Parked hardening: none.

## Verification

- pytest on `tests/test_atlas_reddit_tracker.py` plus the existing
  `tests/test_atlas_reddit_poller.py`, `tests/test_atlas_reddit_digest.py`,
  `tests/test_atlas_reddit_store.py`, `tests/test_atlas_reddit_config.py`,
  and `tests/test_atlas_reddit_scoring.py`: 281 passed (discovery from
  comments+submissions, union-merge rediscovery, replay-safe replies,
  digest warm-section end-to-end through the real writer, per-thread
  error isolation, n-1 pacing, dormancy both sides incl. stale-reply and
  stale-rediscovery probes, dormant-not-polled, wake-then-poll, scope
  floor for the history source, read-only public surface, CLI
  no-creds/knob-ceilings/mark-read happy+unknown). This line is the
  single verification-count source; the PR body mirrors it.
- ASCII byte-scan on the five changed Python files: clean.
- python `scripts/sync_pr_plan.py` on this plan: tables regenerated from
  the real diff.
- Local review bundle runs via `scripts/push_pr.sh` before push.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_reddit/__main__.py` | 92 |
| `atlas_reddit/config.py` | 17 |
| `atlas_reddit/reddit_client.py` | 140 |
| `atlas_reddit/tracker.py` | 140 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Reddit-Listening-Reply-Tracker.md` | 174 |
| `plans/archive/PR-Reddit-Listening-Praw-Poller.md` | 0 |
| `tests/test_atlas_reddit_tracker.py` | 361 |
| **Total** | **927** |
