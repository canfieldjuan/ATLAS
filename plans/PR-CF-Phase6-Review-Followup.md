# PR-CF-Phase6-Review-Followup

## Why this slice exists

The content-factory store's per-job lock keyed its in-thread re-entrancy depth on
the job id alone, while the OS lock it guards lives at
`<root>/.locks/<job_id>.lock`. A nested acquisition for a **different** store
root therefore saw `depth[job_id] > 0`, took the re-entrant path, and skipped
that root's `flock` entirely.

This is reachable on every write, not theoretical: `write_artifact` acquires
`job_lock` (`atlas_brain/services/content_factory_store.py:189`) while
`run_stage` already holds it (`atlas_brain/services/content_factory_runner.py:701`),
so every persisted artifact goes through a nested acquisition.

The fix was authored on `codex/fix-codex-review-issues-in-pr-#2192` as PR #2208,
which targeted `feat/cf-phase6-repurposing`. That branch is fully merged into
`main` (#2192 merged at its round-9 head `d9015920f`, squashed as `9a07c88eb`),
so #2208 had no path to `main` and was closed. This slice carries the same
commit onto `main`, preserving authorship.

### Problem-derived contract

- Root cause: lock identity and re-entrancy identity disagree. The OS lock is
  scoped to `(root, job_id)`; the depth key was scoped to `job_id`.
- Correct fix must: make the depth key match the lock file's namespace exactly,
  including when two paths reach the same lock through a symlink, and prove the
  OS lock is genuinely taken rather than asserting an in-process flag.
- Must not change: cross-process exclusion via `flock`, the re-entrancy
  behaviour for the same `(root, job_id)`, or any artifact contract.

## Scope (this PR)

Ownership lane: content-factory/store-locking
Slice phase: Production hardening

1. `atlas_brain/services/content_factory_store.py`: key the thread-local depth
   map on `(resolved root, safe job id)`.
2. `tests/test_content_factory_runner.py`: regression test proving a nested
   acquisition for a second root still takes that root's OS lock.

### Review Contract

- Acceptance criteria:
  1. With `job_lock` held for `(root_a, "shared-id")`, a nested `job_lock` for
     `(root_b, "shared-id")` acquires `root_b`'s OS lock -- settled by
     `tests/test_content_factory_runner.py::test_job_lock_treats_same_job_id_in_different_roots_as_distinct`,
     which probes via a second file description rather than an in-process flag.
  2. Re-entrancy for the *same* `(root, job_id)` still short-circuits without
     deadlocking -- settled by the existing
     `test_job_lock_is_reentrant_within_a_thread`.
  3. Cross-process exclusion is unchanged -- settled by the existing
     `test_job_lock_excludes_a_second_process`.
  4. Reverting the key to `lock_key = safe` fails criterion 1's test, so the
     test detects the defect rather than coexisting with the fix.
- Reachability proof: `run_stage` is the real entrypoint; it holds `job_lock`
  across `_stamp_draft_fingerprint`, `_enforce_lineage` and `write_artifact`
  (`content_factory_runner.py:701-704`), and `write_artifact` nests a second
  acquisition (`content_factory_store.py:189`).
- Affected surfaces: content-factory artifact store locking only. No contract,
  schema, API or product surface changes.
- Risk areas: false re-entrancy, cross-process exclusion, deadlock.
- Reviewer rules triggered: R1, R2, R8, R10, R14.

### Files touched

- `atlas_brain/services/content_factory_store.py`
- `plans/PR-CF-Phase6-Review-Followup.md`
- `tests/test_content_factory_runner.py`

## Mechanism

`job_lock` keeps `flock` for cross-process exclusion and a thread-local depth map
for safe re-entrancy. The depth key becomes `(str(Path(root).resolve()), safe)`
so it names the same thing the lock file does.

`resolve()` is load-bearing rather than cosmetic: two different paths reaching
one inode through a symlink share a single `flock`, so resolving makes the depth
key agree with the OS's view instead of treating them as distinct and
deadlocking.

### Execution model (AGENTS.md 3k.4)

The invariant: **at most one holder mutates a given `(root, job_id)` at a time,
and a nested acquisition by that same holder does not block.**

- **What can interleave.** Any number of processes and threads calling
  `job_lock`. Cross-process exclusion is `flock` on `<root>/.locks/<job>.lock`;
  `flock` conflicts between separate open file descriptions, so a second
  *thread* in the same process that opens its own handle is excluded too. The
  thread-local depth map is only a re-entrancy short-circuit for the holder --
  it never grants entry to a non-holder, because a non-holder's thread-local has
  no entry for that key.
- **What can be cancelled.** The `finally` releases `flock` and closes the
  handle on any exception, and the nested path decrements its own depth, so an
  exception at any nesting level unwinds to a released lock.
- **Where a crash can land.** If the process dies holding the lock, the OS
  releases `flock` when the fd closes. A crash mid-`write_artifact` can
  therefore leave a partially written artifact with no lock held. That is
  **unchanged by this slice and explicitly out of scope**: `write_artifact`'s
  own atomicity is what bounds it, not the lock.
- **Duplicate / out-of-order.** `job_lock` has no retry or redelivery path, so
  R8's duplicate-execution case does not arise here. Callers that retry
  `run_stage` re-acquire from scratch.
- **Not covered, stated as an assumption:** NFS or other filesystems where
  `flock` is advisory-only or unsupported. The store assumes a local POSIX
  filesystem, as it already did before this change.

## Intentional

- Keep `flock` plus explicit re-entrancy rather than an in-memory mutex;
  readiness must serialize across worker processes, not only threads.
- The fix is the minimum that makes the two identities agree. Deleting the depth
  entry instead of zeroing it, and resolving the root once for both the key and
  the lock path, are noted below rather than bundled in.

## Deferred

- `depth[lock_key] = 0` leaves a zeroed entry per `(root, job_id)` ever locked,
  so the thread-local map grows without bound in a long-lived worker. Pre-existing;
  the composite key multiplies the entry count by the number of roots. `del` is
  equivalent and bounded.
- The depth key resolves the root while the lock *path* does not, so a store root
  that is itself a symlink created between two calls could key differently from
  its lock file. Exotic, and no failure path is demonstrated.

Parked hardening: none.

## Verification

    python -m pytest tests/test_content_factory_runner.py -k job_lock -q
    # -> 4 passed

    python -m pytest tests/test_content_factory_runner.py \
        tests/test_content_factory_schemas.py \
        tests/test_content_factory_store.py -q
    # -> 451 passed

Failure detection proven per 3i, not assumed: reverting the key to
`lock_key = safe` fails
`test_job_lock_treats_same_job_id_in_different_roots_as_distinct` and leaves the
other three lock tests passing; restored, 4 pass. The original PR recorded this
suite as unrunnable in its environment -- it runs here, so the evidence is
stronger than first filed.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/services/content_factory_store.py` | 14 |
| `plans/PR-CF-Phase6-Review-Followup.md` | 153 |
| `tests/test_content_factory_runner.py` | 14 |
| **Total** | **181** |
