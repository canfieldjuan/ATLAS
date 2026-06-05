# PR-Content-Ops-File-Cross-Process-Hardening-Register

## Why this slice exists

The uploaded-file import path now has an in-process admission gate and a
same-process load runner that proves shared-pool behavior. That closes the
original database-pool pressure issue for one hosted router process.

One deployment risk remains: a multi-worker or multi-process host can still
create one admission window per process. That is not solved by the in-process
gate, and it should stay visible as future production hardening instead of
being hidden by the closed `FILECONCURRENCY-1` item.

## Scope (this PR)

Ownership lane: content-ops/backend-file-ingestion-validation

1. Add a narrower `FILECONCURRENCY-2` root hardening item for cross-process /
   multi-worker uploaded-file import admission.
2. Leave the merged `PR-Content-Ops-File-Concurrency-Hardening-Close` plan doc
   unchanged.

### Files touched

- `HARDENING.md`
- `plans/PR-Content-Ops-File-Cross-Process-Hardening-Register.md`

## Mechanism

This is a register-only slice. `FILECONCURRENCY-2` records that the current
route-level gate is process-local and names the production topology that still
needs a distributed queue, shared semaphore, or durable job boundary before
high-concurrency multi-worker uploads can be called fully hardened.

## Intentional

- No product code changes. The current synchronous route still functions, and
  the remaining concern depends on deployment topology.
- No claim that durable imports are implemented. This PR records the remaining
  production-hardening gap so it is owned explicitly.

## Deferred

- Cross-process import admission, durable import jobs, and queue visibility
  remain future production-hardening work.
- Parked hardening: `FILECONCURRENCY-2`.

## Verification

- Hardening register spot-check:
  - Confirmed `HARDENING.md` contains `FILECONCURRENCY-2`.
- Plan/code consistency and local review:
  - `bash scripts/local_pr_review.sh --allow-dirty`
  - Passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 62 |
| Hardening register | 11 |
| Total | 73 |
