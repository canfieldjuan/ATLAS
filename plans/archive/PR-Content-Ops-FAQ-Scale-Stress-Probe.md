# PR-Content-Ops-FAQ-Scale-Stress-Probe

## Why this slice exists

The FAQ generator and database lifecycle now have real 500-row and 1,000-row
proofs. The next survivability question is where the deterministic path starts
to bend under larger customer uploads.

This slice intentionally pushes larger real CFPB-derived support-ticket files
through the existing generator and database lifecycle smokes, then adds
concurrent lifecycle probes to find the first operational failure mode. It
records the observed limits, failures, timing, memory, and issue status.

## Scope (this PR)

Ownership lane: content-ops/faq-generator-validation

1. Generate larger local JSONL source-row fixtures from the local CFPB archive.
2. Run existing deterministic FAQ generator smokes at increasing row counts.
3. Run existing Postgres-backed FAQ lifecycle smokes at increasing row counts
   when the generator run passes.
4. Run concurrent Postgres-backed FAQ lifecycle probes to pressure the shared
   database ceiling.
5. Add a validation note with the commands, observed results, and any surfaced
   issues.

### Files touched

- `plans/PR-Content-Ops-FAQ-Scale-Stress-Probe.md`
- `docs/extraction/validation/content_ops_faq_scale_stress_probe_2026-05-23.md`
- `HARDENING.md`

## Mechanism

The probe uses the existing scripts only:

- CFPB row conversion reuses `cfpb_row_to_source_row`.
- FAQ generation uses `scripts/smoke_content_ops_faq_scale_run.py`.
- DB lifecycle uses `scripts/smoke_content_ops_faq_lifecycle.py`.
- Concurrent probes launch the same lifecycle smoke with unique account IDs and
  temp-only result paths.

The validation note records exact row counts, concurrency levels, elapsed time,
peak RSS when available, output checks, warning counts, and each issue that
appears.

## Intentional

- Documentation/validation record only; no generator, repository, migration, or
  lifecycle code changes.
- Temporary large fixtures and result artifacts stay under `tmp/` and are not
  checked in.
- This slice does not depend on the unmerged artifact-runner PR.
- The concurrency harness is temp-only. This PR records the observed behavior
  without adding a reusable load-test runner.

## Deferred

- Hosted limits, async job execution, and bounded DB concurrency are deferred to
  hardening slices. This PR is the evidence slice that identifies the limits.
- The lifecycle smoke visibility fix for pool-creation failures is deferred to
  the next slice because the probe itself completed and the failure is recorded.

## Verification

- Passed: generated 2,000, 5,000, 10,000, 25,000, and 50,000 row
  CFPB-derived JSONL fixtures from the local archive.
- Passed: deterministic FAQ scale smokes at 2k, 5k, 10k, 25k, and 50k rows;
  all exited `0` with output checks passing.
- Passed: Postgres-backed lifecycle smokes at 2k, 5k, 10k, 25k, and 50k rows;
  all exited `0`, saved one FAQ, exported draft/reviewed rows, and reported
  `error_count=0`.
- Passed: concurrent lifecycle probes at 5x5k, 10x10k, 20x10k, and 50x10k.
- Failed as expected under saturation: 100x5k returned 97 successes and 3
  `asyncpg.exceptions.TooManyConnectionsError` failures.
- Parked: 50k uploads are batch-safe but not request/response-safe without
  hosted limits or an async job boundary.
- Parked: 100-way lifecycle pressure can exhaust DB connections and failed
  pool creation does not write the requested result artifact.
- Passed: `bash scripts/local_pr_review.sh --allow-dirty`.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~89 |
| Validation note | ~200 |
| HARDENING entries | ~22 |
| **Total** | **~311** |
