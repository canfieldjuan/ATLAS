# PR-CF-Phase6-Review-Followup

## Why this slice exists

Codex review of #2192 identified two blockers: Phase 6 readiness could be invalidated between lineage validation and persistence, and the draft fingerprint had been added to the already-published `editorial_audit.v2` wire contract. The task checkout already contains the root fixes (a lock spanning validation through commit and a frozen v2 plus fingerprint-bearing v3); this follow-up closes a remaining lock-identity edge so independent roots cannot accidentally share re-entrancy state.

### Problem-derived contract

- Root cause: readiness depends on mutable per-job draft/audit state, so validation and commit must be one serialized transaction; additionally, fingerprint metadata cannot be appended to a strict, published v2 schema without breaking rollback readers. The lock's in-thread re-entrancy identity must include both storage root and job id, because those together identify a job.
- Correct fix must touch/change: retain the v3 audit contract and runner-wide lock already present, key re-entrant lock state by normalized root plus job id, and prove nested same-job locks in different roots remain independently exclusive.
- Must not change: Phase 6 artifact payload semantics, frozen editorial audit v1/v2 shapes, classifier behavior, worker protocol, or user-facing product shape.

## Scope (this PR)

Ownership lane: content-factory
Slice phase: Production hardening

1. Make the per-thread job-lock identity match the store's actual `(root, job_id)` namespace.
2. Add a concurrency regression test for identical job ids under different roots.

### Review Contract

- Acceptance criteria: v2 remains frozen and v3 owns the fingerprint; `run_stage` retains one lock across lineage reads and artifact commit; nested locks for the same id in different roots acquire distinct OS locks; focused tests pass.
- Reachability proof: `run_stage` is the real entrypoint exercised by the existing readiness/write-boundary test; the new test observes exclusion through the actual `job_lock` context manager.
- Affected surfaces: content-factory artifact store locking and runner regression tests.
- Risk areas: deadlock, false re-entrancy, cross-process exclusion, backward-compatible audit parsing.
- Reviewer rules triggered: R1, R2, R5, R8, R10, R14.

### Files touched

- `atlas_brain/services/content_factory_store.py`
- `plans/PR-CF-Phase6-Review-Followup.md`
- `tests/test_content_factory_runner.py`

## Mechanism

The store continues to use `flock` for process-wide exclusion and thread-local depth for safe re-entrancy. The depth map key becomes `(normalized absolute root, safe job id)` instead of only the job id, matching the filesystem lock namespace and preventing a nested operation on another store root from bypassing its lock.

## Intentional

- Keep `flock` plus explicit re-entrancy rather than replacing it with an in-memory mutex; readiness must serialize across worker processes as well as threads.
- Preserve the already-landed `editorial_audit.v3` design rather than modifying frozen v2 again.

## Deferred

- None.

Parked hardening: none.

## Verification

- Pytest focused runner suite — not run: environment lacks `pytest_asyncio` during conftest import.
- Python byte-compilation of the changed runtime and related runner/schema modules — passed.
- Root-scoped `job_lock` Python probe — passed.
- Git whitespace validation — passed.
- Local PR review bundle — not run: checkout has no `origin/main` ref.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/services/content_factory_store.py` | 14 |
| `plans/PR-CF-Phase6-Review-Followup.md` | 65 |
| `tests/test_content_factory_runner.py` | 14 |
| **Total** | **93** |
