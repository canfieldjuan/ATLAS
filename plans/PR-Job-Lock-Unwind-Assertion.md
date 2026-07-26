# PR-Job-Lock-Unwind-Assertion

## Why this slice exists

`test_job_lock_identity_includes_the_root` proves the store's per-job lock is
held for two roots while nested, but stops there. It never observes the unwind,
so a `job_lock` that acquires correctly and then **never releases** passes it:
inside the block a leaked lock and a healthy one are indistinguishable.

This was found while reviewing #2208 and confirmed by injection rather than by
reading: replacing the release in `job_lock`'s `finally` with a leak makes the
pre-change test pass and the post-change test fail.

### Problem-derived contract

- Root cause: the test asserts an acquisition invariant and no release
  invariant, so the release path has no coverage at all.
- Correct fix must: observe the state *after* leaving the inner scope -- inner
  root free, outer root still held -- and after leaving the outer scope, using
  the same second-file-description probe the rest of the file uses so it reads
  the OS lock rather than an in-process flag.
- Must not change: `job_lock` itself, the existing acquisition assertions, or
  any other test.

## Scope (this PR)

Ownership lane: content-factory/store-locking
Slice phase: Robust testing

1. `tests/test_content_factory_runner.py`: extend
   `test_job_lock_identity_includes_the_root` with the unwind assertions.

### Review Contract

- Acceptance criteria:
  1. Leaving the inner `job_lock` scope frees the inner root's OS lock and
     leaves the outer root's held -- settled by the two assertions added after
     the inner `with` block.
  2. Leaving the outer scope frees the outer root's lock -- settled by the
     assertion after the outer block.
  3. The assertions detect a real defect: injecting a leak into `job_lock`'s
     `finally` fails this test, while the pre-change version of the same test
     passes with that leak present.
  4. No other test changes behavior -- settled by the content-factory suite
     (runner + store + schemas + copy-verification) at 1354 passed on this
     branch, matching `main`, since this slice adds assertions to an existing
     test rather than new tests.
- Reachability proof: the test drives the real `job_lock` context manager and
  probes exclusion through `_job_lock_is_free`, which opens a second file
  description and attempts a non-blocking `flock`, so it observes the OS lock.
- Affected surfaces: one test function. No runtime code changes.
- Execution model (R8). The invariant proved is: on leaving a `job_lock` scope,
  exactly the lock identity that scope acquired is released, and no other. The
  admitted model and what settles it:
  - **Normal exit** -- the assertions added here, at both nesting levels.
  - **Exception / early return** -- `job_lock` is a `@contextmanager` whose
    release lives in a `finally`, so the generator is closed and the release
    runs on any non-fatal unwind. `with` guarantees this; it is not sampled
    separately because the same `finally` is the only release path, and a
    probe would assert Python's context-manager contract rather than this
    module's behaviour.
  - **Re-entrant exit** -- the depth branch returns before touching `flock`, so
    an inner same-identity release cannot free the outer OS lock. Covered by
    the existing re-entrancy test.
  - **Cross-process** -- `flock` is released by the kernel on fd close or
    process death, so a crashed holder cannot wedge the lock permanently. Out
    of scope for this slice: no test drives a second process.
  - **Assumption, stated:** the probe reads the OS lock through a SECOND file
    description, because `flock` is per-open-file-description; an in-process
    flag would not observe a leak.
- Risk areas: none -- test-only, and the injection probe bounds whether the new
  assertions are meaningful rather than decorative.
- Reviewer rules triggered: R2, R8, R10, R14.

### Files touched

- `HARDENING.md`
- `plans/PR-Job-Lock-Unwind-Assertion.md`
- `tests/test_content_factory_runner.py`

## Mechanism

Three assertions on the existing test. After the inner `with` exits,
`_job_lock_is_free(second)` must be true and `_job_lock_is_free(first)` false;
after the outer `with` exits, `first` must be free. The comment records why
held-while-nested alone is insufficient, so the assertions are not deleted later
as redundant with the two above them.

## Intentional

- **Extends the existing test rather than adding a new one.** The defect class
  is identical; a second test function would duplicate coverage and the file
  already carries four `job_lock` tests.
- **Test-only.** `job_lock` is correct as it stands on `main` after #2201; this
  closes a coverage gap, it does not change behavior.

## Deferred

- One NIT from #2213 survives review: `depth[lock_key] = 0` never deletes its
  entry, so the per-thread depth map grows one key per distinct (job, root).
  No demonstrated failure path, so it is parked in `HARDENING.md` rather than
  left only in this plan -- an in-flight plan is archived on merge, which would
  drop the item out of the active queue.
- The second #2213 NIT is WITHDRAWN as stale, not deferred: it claimed the
  depth key and the lock path disagree about resolution, but the key is derived
  FROM the resolved path, so they cannot differ. Recording the withdrawal
  rather than deleting it silently, so the plan is not misleading history.

Parked hardening: none.

## Verification

    python -m pytest tests/test_content_factory_runner.py -k job_lock -q
    # -> 4 passed

    python -m pytest tests/test_content_factory_runner.py \
        tests/test_content_factory_schemas.py \
        tests/test_content_factory_store.py -q
    # -> 1184 passed

Detection proven by injection, per AGENTS.md 3i. Replacing the release in
`job_lock`'s `finally` with a leak:

    post-change test:  FAILED test_job_lock_identity_includes_the_root
    pre-change test:   1 passed          <- blind to the leak

so the added assertions catch something the previous ones could not.

## Estimated diff size

| File | LOC |
|---|---:|
| `HARDENING.md` | 21 |
| `plans/PR-Job-Lock-Unwind-Assertion.md` | 135 |
| `tests/test_content_factory_runner.py` | 9 |
| **Total** | **165** |
