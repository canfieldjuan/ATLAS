# PR-Content-Ops-File-Concurrency-Hardening-Close

## Why this slice exists

`HARDENING.md` still lists `FILECONCURRENCY-1` even though the source fix and
the hosted-shape proof are now merged:

1. `PR-Content-Ops-File-Import-Admission-Gate` added the in-process route
   admission gate before import requests resolve or acquire the database pool.
2. `PR-Content-Ops-File-Import-Inprocess-Load` proved concurrent uploaded-file
   imports use one shared pool and return deterministic 429 admission responses
   when the gate is full.

Leaving the item parked makes the active hardening queue look stale and can
send the next session back into work that is already closed.

## Scope (this PR)

Ownership lane: content-ops/backend-file-ingestion-validation

1. Remove the closed `FILECONCURRENCY-1` item from root `HARDENING.md`.
2. Keep the blog/deep-dive hardening pointer intact.

### Files touched

- `HARDENING.md`
- `plans/PR-Content-Ops-File-Concurrency-Hardening-Close.md`

## Mechanism

This is a register-only update. The code-level mitigation remains in
`extracted_content_pipeline/api/control_surfaces.py`, and the reusable proof
remains in `scripts/smoke_content_ops_ingestion_file_route_inprocess_load.py`.
The hardening queue is updated to reflect that the parked item has been drained
by the already-merged implementation and validation slices.

## Intentional

- This PR does not remove the older process-based smoke runner. It still helps
  discover multi-process pressure, but it is not the hosted shared-pool shape.
- This PR does not claim cross-process or durable queueing is done. Those are
  separate production-hardening follow-ups named in the merged plan docs.

## Deferred

- Cross-process backpressure, durable import jobs, and shared queue visibility
  remain deferred by `PR-Content-Ops-File-Import-Admission-Gate` and
  `PR-Content-Ops-File-Import-Inprocess-Load`.
- Parked hardening: none.

## Verification

- Register spot-check:
  - `git show origin/main:HARDENING.md | sed -n '1,220p'`
  - Confirmed `FILECONCURRENCY-1` was the only root parked item in this
    ownership lane.
- Merged mitigation/proof spot-check:
  - `git log --oneline --decorate -8 origin/main`
  - Confirmed `PR-Content-Ops-File-Import-Admission-Gate` and
    `PR-Content-Ops-File-Import-Inprocess-Load` are both on `origin/main`.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 68 |
| Hardening register | 11 |
| Total | 79 |
