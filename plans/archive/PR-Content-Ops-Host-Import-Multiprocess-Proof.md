# PR-Content-Ops-Host-Import-Multiprocess-Proof

## Why this slice exists
PR-Content-Ops-Host-Import-Admission-Provider wired Atlas-host imports to a
Postgres advisory-lock admission gate, but the existing multiprocess load probe
still exercises the extracted router's process-local gate. `FILECONCURRENCY-2`
remains parked until we can prove the host provider path across child
processes.

This slice makes that proof repeatable without adding a new load-test framework.

## Scope (this PR)
Ownership lane: content-ops/backend-file-ingestion-validation

Slice phase: robust testing.

1. Add an opt-in Postgres host-admission mode to the in-process file import
   load runner.
2. Thread that mode through the existing multiprocess wrapper.
3. Add focused tests proving the child command and router wiring use the host
   provider when requested.
4. Record the real local DB multiprocess proof and close the corresponding
   hardening item.

### Files touched
- `scripts/smoke_content_ops_ingestion_file_route_inprocess_load.py`
- `scripts/smoke_content_ops_ingestion_file_route_multiprocess_load.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `.github/workflows/extracted_pipeline_checks.yml`
- `tests/test_smoke_content_ops_ingestion_file_route_inprocess_load.py`
- `tests/test_smoke_content_ops_ingestion_file_route_multiprocess_load.py`
- `docs/extraction/validation/content_ops_host_import_multiprocess_proof_2026-05-23.md`
- `HARDENING.md`
- `plans/PR-Content-Ops-Host-Import-Multiprocess-Proof.md`

## Mechanism
The existing runners keep their default `local` behavior. A new
`--admission-provider postgres` flag injects
`build_content_ops_import_admission_gate(max_concurrency=...)` into the router
via the existing `ingestion_import_admission_provider` seam. The gate uses the
same asyncpg pool as the import write path, so separate child processes compete
for the same Postgres advisory-lock slots.

The multiprocess wrapper passes the flag to children. A second opt-in flag,
`--allow-capacity-only-children`, lets the parent accept child processes that
only hit admission capacity when a global gate has fewer slots than child
processes.

The mocked load-runner tests are enrolled in the extracted pipeline CI wrapper,
and the workflow path filter now includes both load-runner scripts and tests so
future edits trigger that gate.

The host Postgres admission builder is imported lazily so the mocked CI tests
can import the runner without requiring the optional `asyncpg` runtime package.

## Intentional
- No production route behavior changes. PR #895 already wired the hosted Atlas
  router; this slice only makes the smoke tooling exercise that same provider
  shape.
- CI-sized coverage stays mocked and enrolled in extracted pipeline CI. The real
  local DB proof is recorded in the validation note and uses the existing
  1,000-row CFPB fixture.
- The Postgres admission provider import stays lazy. That keeps mocked CI
  coverage independent of optional database-driver installation while preserving
  the real provider path when `--admission-provider postgres` is used.
- No durable job queue work. This is the proof harness for the current
  synchronous admission design.

## Deferred
- Parked hardening: none.
- Background jobs and queue visibility remain future product hardening if the
  product chooses to accept imports beyond the current synchronous caps.

## Verification
- Focused load-runner tests:
  `python -m pytest tests/test_smoke_content_ops_ingestion_file_route_inprocess_load.py tests/test_smoke_content_ops_ingestion_file_route_multiprocess_load.py -q`
- Python compile check for both load-runner scripts and focused tests.
- CI enrollment check:
  `scripts/run_extracted_pipeline_checks.sh` includes both load-runner test
  files, and `.github/workflows/extracted_pipeline_checks.yml` includes both
  load-runner scripts and tests in the `pull_request` and `push` path filters.
- Real local DB multiprocess proof:
  `python scripts/smoke_content_ops_ingestion_file_route_multiprocess_load.py tmp/content_ops_faq_1000/cfpb_1000_source_rows.jsonl --source-format jsonl --source cfpb-route-host-postgres-admission-20260523 --min-source-rows 1000 --default-field company_name=CFPB --default-field vendor_name=CFPB --default-field contact_email=cfpb-public-archive@example.invalid --account-id acct-host-import-admission-multiprocess-20260523 --user-id user-host-import-admission --processes 3 --child-concurrency 2 --child-import-max-concurrency 1 --child-min-successes 1 --child-expect-at-capacity-min 1 --min-total-successes 1 --expect-total-at-capacity-min 5 --allow-capacity-only-children --admission-provider postgres --output-dir tmp/content_ops_host_import_admission_multiprocess_20260523 --output-result tmp/content_ops_host_import_admission_multiprocess_20260523/result.json --json`
  (`ok=true`, `successes=1`, `at_capacity=5`, `failed_processes=0`)
- Local PR review:
  `bash scripts/local_pr_review.sh --allow-dirty`

## Estimated diff size
| Area | Estimated LOC |
|---|---:|
| Total | 432 |
