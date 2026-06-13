# PR-Content-Ops-Claim-Evidence-Artifact-Writer

## Why this slice exists

Issue #1435 now has a deterministic benchmark result artifact, Markdown
renderer, and package-owned JSON/Markdown bundle. The next operator handoff gap
is writing that bundle into an artifact directory without letting a future CLI
or transport layer reinvent filenames, content types, or failure presentation.
This slice lands the narrow write boundary over an already-built artifact so the
later live-run CLI can focus on provider execution and orchestration.

This slice exceeds the normal 400 LOC target after review because PR #1508
surfaced a MAJOR symlink-follow defect at the write boundary. The added scope is
limited to rejecting symlinked result paths and pinning the defect class with
multiple same-class probes.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Functional validation

1. Add a deterministic artifact-directory writer for the existing
   claim/evidence JSON + Markdown bundle.
2. Return a small write result with success state, written file metadata, and
   errors.
3. Create the output directory when needed and reject an output path that is an
   existing file.
4. Preserve the existing fail-closed artifact behavior: malformed artifact input
   still writes a no-go JSON/Markdown bundle.
5. Leave artifact JSON parsing, provider adapters, live prompt execution, and
   batch model-run CLI orchestration out of this slice.

### Files touched

- `extracted_content_pipeline/claim_evidence_benchmark.py`
- `plans/PR-Content-Ops-Claim-Evidence-Artifact-Writer.md`
- `tests/test_extracted_content_claim_evidence_benchmark.py`

### Review Contract

Acceptance criteria:
- A valid result artifact writes the stable JSON and Markdown bundle into the
  requested output directory.
- The returned write result identifies both written files, content types, and
  byte counts.
- Missing directories are created.
- An existing non-directory output path fails closed with no write attempt.
- Malformed artifact input writes no-go JSON and Markdown using the same
  fail-closed artifact error as the bundle helper.

Affected surfaces: extracted benchmark helper and focused unit tests.

Risk areas: accidental file writes outside the requested directory, false-green
artifact presentation, and future CLI compatibility.

Reviewer rules triggered: R1, R2, R6, R10, R14.

## Mechanism

The benchmark module gains immutable write-result value types and a helper that
calls `claim_evidence_result_artifact_files`, creates the requested output
directory, writes each fixed bundle entry with UTF-8 encoding, and returns
metadata for the files it wrote. It does not accept caller-controlled filenames;
the only file names are the two package-owned bundle names from #1506.

Invalid output-directory input or an existing file at the output path returns a
failed result with an error string. A malformed artifact object is not an output
path error: it follows the already-established fail-closed artifact path and
writes no-go JSON/Markdown so operators can inspect the failure.

## Intentional

- This introduces file I/O only at the explicit artifact-writer boundary. The
  benchmark runner, scorer, renderer, and bundle helper remain deterministic and
  provider-free.
- No CLI is added yet. The helper gives the next CLI slice one package-owned
  write primitive instead of duplicating write behavior in a script.
- The writer reports relative bundle names and concrete file-system paths; it
  does not make an opinionated archive layout beyond the requested directory.
- Symlink rejection uses an explicit pre-write path check. A kernel-level
  no-follow write is deferred until this helper becomes a concurrent automation
  boundary.

## Deferred

- CLI wrapper that loads/constructs benchmark artifacts and calls this writer.
- Artifact JSON parsing back into result dataclasses.
- Concrete provider adapters and live prompt execution.
- Batch model-run CLI for fixture-file model runs.
- Verifier rubric inclusion and MCP exposure only if benchmark thresholds pass.
- Parked hardening: none.

## Verification

- pytest tests/test_extracted_content_claim_evidence_benchmark.py: 59 passed.
- bash scripts/validate_extracted_content_pipeline.sh: passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline: passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt: Atlas runtime import findings: 0.
- bash scripts/check_ascii_python.sh: passed.
- bash scripts/run_extracted_pipeline_checks.sh: 3903 passed, 10 skipped.
- git diff --check: passed.
- bash scripts/local_pr_review.sh --current-pr-body-file tmp/content_ops_claim_evidence_artifact_writer_pr_body.md: passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/claim_evidence_benchmark.py` | 118 |
| `plans/PR-Content-Ops-Claim-Evidence-Artifact-Writer.md` | 111 |
| `tests/test_extracted_content_claim_evidence_benchmark.py` | 223 |
| **Total** | **452** |
