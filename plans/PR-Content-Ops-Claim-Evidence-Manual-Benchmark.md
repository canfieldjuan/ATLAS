# PR-Content-Ops-Claim-Evidence-Manual-Benchmark

## Why this slice exists

#1435 now has the deterministic pieces for the manual claim/evidence benchmark
path, but the operator still has to stitch returned prompt rows through multiple
commands before seeing the go/no-go artifact. #1530 closed the response-import
bridge; this slice makes the manual benchmark path a single deterministic
operator command after out-of-band model responses are collected.

This is still a vertical slice, not the live-provider runner. It proves the real
workflow shape from labeled fixture plus prompt packets plus returned responses
to recorded responses and result artifacts, while preserving the reliability
gate rule that no verifier or MCP inclusion happens before benchmark thresholds
pass.

The estimated diff is above the 400 LOC soft target because the slice adds a new
operator command, same-slice negative fixtures for each wrapper-owned guard, and
CI enrollment. Splitting the tests from the command would leave a new benchmark
orchestrator without CI-proven failure detection.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Vertical slice

1. Add a deterministic manual benchmark runner CLI that imports returned
   prompt-packet responses into the existing recorded-response JSON shape and
   then writes the existing claim/evidence result artifact bundle.
2. Fail closed before artifact writing when response import fails, result
   artifact files already exist in the output directory, output paths are
   unsafe, or the shared artifact builder reports write/benchmark errors.
3. Add focused CLI tests and enroll them in the extracted pipeline checks.

### Files touched

- `.github/workflows/extracted_pipeline_checks.yml`
- `plans/PR-Content-Ops-Claim-Evidence-Manual-Benchmark.md`
- `scripts/run_content_ops_claim_evidence_manual_benchmark.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_content_ops_claim_evidence_manual_benchmark_cli.py`

### Review Contract

Acceptance criteria:

- The CLI writes a recorded-response JSON file and the existing result artifact
  JSON/Markdown files from a valid fixture, packet export, and returned response
  file.
- The CLI does not call providers, read credentials, mutate registries, expose
  MCP tools, or change verifier rubric behavior.
- Import failures stop before final artifact files are written, and the wrapper
  refuses a directory that already contains result artifacts so a failed rerun
  cannot leave stale go/no-go files beside the failed envelope.
- Unsafe output directories or recorded-response paths fail closed without
  following symlinks.
- Tests are included in the extracted pipeline local runner and workflow path
  filters.

Affected surfaces:

- Deterministic #1435 claim/evidence benchmark scripts and tests only.

Risk areas:

- Wrapper path safety around an output directory that contains both intermediate
  and final artifact files.
- Stale result artifacts from previous runs when a later import attempt fails.
- Exit-code separation between option/I/O failures and benchmark/content
  failures.
- CI enrollment for a new script and test file.

Reviewer rules triggered: R1, R2, R10, R12, R14

## Mechanism

The runner reuses the two already-reviewed CLI service boundaries instead of
duplicating benchmark logic:

1. It calls the prompt-response importer with an explicit recorded-response
   target inside the operator output directory, unless the operator supplies a
   separate recorded-response output path.
2. Before import, it rejects output directories that already contain benchmark
   result files, plus symlinked output directories or recorded-response targets.
3. If import succeeds, it calls the existing artifact builder with the fixture
   and recorded-response file, writing the existing result JSON and Markdown
   bundle.
4. It prints one JSON envelope that includes the import payload, artifact
   payload, recorded-response path, output directory, errors, and go/no-go
   status.

The wrapper owns only orchestration and path safety. Prompt/schema construction,
returned-row validation, scoring, result rendering, and artifact writing remain
in the existing modules.

## Intentional

- No live provider adapter in this slice. The benchmark can still be run with
  out-of-band returned rows while provider credentials and model-specific
  transports remain deferred.
- The runner does not re-export prompt packets. Packet export remains a separate
  operator step so the packet file used for model witnesses is the same packet
  file used for response import.
- The wrapper refuses existing result-artifact filenames instead of deleting
  them. That avoids destructive cleanup and makes the operator choose a fresh
  run directory before a new benchmark run.
- The wrapper reports nested import and artifact payloads rather than inventing
  a new result schema. That keeps this slice thin and makes failures traceable to
  the existing deterministic components.

## Deferred

- Concrete provider adapters and live prompt execution for Claude/GPT/optional
  third model.
- Batch orchestration over provider credentials and stability reruns.
- Real benchmark result go/no-go writeup from the final operator-labeled 40-row
  set.
- Verifier rubric inclusion and MCP exposure, only after #1435 thresholds pass.

Parked hardening: none.

## Verification

- Python compile check for the manual benchmark runner script passed.
- Focused manual benchmark CLI tests:
  8 passed.
- Claim/evidence focused sweep:
  117 passed.
- Full extracted pipeline wrapper: 4088 passed, 10 skipped.
- Body-aware local PR review passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/extracted_pipeline_checks.yml` | 4 |
| `plans/PR-Content-Ops-Claim-Evidence-Manual-Benchmark.md` | 141 |
| `scripts/run_content_ops_claim_evidence_manual_benchmark.py` | 218 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_content_ops_claim_evidence_manual_benchmark_cli.py` | 316 |
| **Total** | **680** |
