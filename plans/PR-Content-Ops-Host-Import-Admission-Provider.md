# PR-Content-Ops-Host-Import-Admission-Provider

## Why this slice exists
`FILECONCURRENCY-2` tracks the remaining Content Ops uploaded-file import risk:
the extracted route now exposes a host admission provider seam, but Atlas still
mounts the router without a shared provider. In a multi-worker deployment that
means each process can still admit its own local import window.

This slice wires the Atlas host to a Postgres advisory-lock-backed admission
gate so non-dry-run file imports share one bounded capacity window across app
processes.

## Scope (this PR)
Ownership lane: content-ops/backend-file-ingestion-validation

1. Add a host-owned Content Ops import admission gate backed by Postgres
   session advisory locks.
2. Wire that provider into the Atlas-mounted Content Ops control-surface
   router.
3. Add focused unit tests for admission success, capacity denial, release, and
   invalid capacity.
4. Narrow the parked hardening item to the remaining durable-job/load-proofing
   work after the shared provider is wired.

### Files touched

- `atlas_brain/_content_ops_import_admission.py`
- `atlas_brain/api/__init__.py`
- `tests/test_atlas_content_ops_import_admission.py`
- `HARDENING.md`
- `plans/PR-Content-Ops-Host-Import-Admission-Provider.md`

## Mechanism
`ContentOpsPostgresImportAdmissionGate.acquire()` obtains one host database
connection and tries `pg_try_advisory_lock(namespace, slot)` for slots from `0`
through `max_concurrency - 1`.

The first successful slot is held on the same connection until `release()`
calls `pg_advisory_unlock(namespace, slot)` and returns the connection to the
pool. If no slot is available, the gate releases the connection immediately and
returns `False`, letting the existing extracted route return its 429
`content_ops_ingestion_import_at_capacity` response.

## Intentional
- The gate uses Postgres advisory locks rather than adding Redis or a durable
  queue because Atlas already requires the database for write-mode imports.
- The lock is session-scoped and intentionally holds one pool connection while
  the import runs; advisory locks require the same connection for unlock.
- The provider returns a fresh gate per request so a held connection/slot never
  leaks between unrelated requests.

## Deferred
- Durable background import jobs and queue visibility remain parked; this slice
  bounds concurrent writers but does not add async job orchestration.
- A live multi-process smoke that exercises the Atlas-mounted provider remains a
  follow-up. Existing route-level load probes exercise the extracted route seam,
  not the host provider.
- Parked hardening: `FILECONCURRENCY-2 - Uploaded-file imports need hosted
  multiprocess proof`.

## Verification
- Focused host admission tests: `python -m pytest tests/test_atlas_content_ops_import_admission.py -q` (`7 passed`)
- Import-route seam regression: `python -m pytest tests/test_extracted_content_control_surface_api.py -q` (`92 passed`)
- Local PR review: `bash scripts/local_pr_review.sh --allow-dirty` (`passed`)

## Estimated diff size
| Area | Estimated LOC |
|---|---:|
| Total | 397 |
