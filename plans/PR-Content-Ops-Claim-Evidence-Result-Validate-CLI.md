# PR-Content-Ops-Claim-Evidence-Result-Validate-CLI

## Why this slice exists

#1435 now has the deterministic benchmark artifact path through saved
claim evidence result JSON output and the #1512 parser that can rehydrate a
saved result artifact fail-closed. That made the JSON readable by package code,
but operators still do not have a file-level command to validate a saved result
artifact or re-render the Markdown report without rerunning model witnesses.

This slice closes that deterministic handoff: take a saved result JSON file,
parse it through the package-owned artifact contract, report the parsed
go/no-go state in a machine-readable envelope, and optionally write the
existing Markdown rendering. It keeps the next live-provider and verifier/MCP
steps gated behind real benchmark data.

This slice is over the normal 400 LOC target because the operator CLI, its
malformed/read/write failure fixtures, and same-PR CI enrollment need to ship
together; splitting those would either ship an unprotected command or an
unenrolled test.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Functional validation

1. Add a deterministic operator CLI for saved claim/evidence result artifacts.
2. Read an existing claim evidence result JSON file and delegate validation
   to the package-owned parser from #1512.
3. Emit a JSON status envelope with parsed `ok`, `go_no_go`, artifact errors,
   and verdict failures.
4. Optionally write the package-owned Markdown rendering to an operator-selected
   path.
5. Keep provider calls, fixture/model-run execution, verifier rubric inclusion,
   MCP transport, DB writes, and registry mutation out of scope.

### Files touched

- `.github/workflows/extracted_pipeline_checks.yml`
- `plans/PR-Content-Ops-Claim-Evidence-Result-Validate-CLI.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `scripts/validate_content_ops_claim_evidence_result.py`
- `tests/test_content_ops_claim_evidence_result_validate_cli.py`

### Review Contract

Acceptance criteria:
- A valid saved result JSON parses through the existing artifact parser and
  returns a successful JSON envelope without rerunning benchmark witnesses.
- Malformed, missing, unreadable, or contradictory result JSON returns a
  no-go envelope and non-zero exit without raising.
- Markdown output is optional, uses the existing renderer, creates parent
  directories, and refuses to overwrite a directory path.
- The CLI does not call providers, execute prompts, mutate the registry, expose
  MCP behavior, or read fixture/model-response inputs.

Affected surfaces: one operator script plus focused CLI tests for the existing
extracted claim/evidence result artifact contract.

Risk areas: false-green saved benchmark evidence, ambiguous file errors,
unexpected overwrites, and scope creep into live benchmark execution.

Reviewer rules triggered: R1, R2, R10, R12, R14.

## Mechanism

The script reads a saved result artifact as UTF-8 text, calls the existing
package loader, builds a small deterministic status payload from the parsed
artifact, and prints that payload as sorted JSON. Exit codes distinguish
successful parsed artifacts, parsed no-go artifacts, and file/write failures.

When a Markdown output path is provided, the script renders the parsed artifact
with the existing Markdown renderer. It creates missing parent directories and
fails closed if the target path is a directory or cannot be written.

## Intentional

- This slice validates a saved result file only. It does not rebuild the result
  from fixture and recorded-response files; the #1509 artifact CLI already owns
  that path.
- Non-passing benchmark artifacts are valid saved artifacts, so they return a
  parsed no-go envelope rather than a file/write error.
- The Markdown writer is a narrow convenience for saved result artifacts, not a
  second artifact-directory writer.

## Deferred

- Concrete provider adapters and live prompt execution.
- Batch model-run orchestration that calls real models and records response
  files.
- Operator labeling workflow for the final 40-row fixture.
- Verifier rubric inclusion and MCP exposure only if benchmark thresholds pass.

Parked hardening: none.

## Verification

- py_compile for the result validation script and focused CLI test: passed.
- focused CLI pytest: 7 passed.
- claim/evidence focused pytest sweep: 85 passed.
- extracted content pipeline validation script: passed.
- reasoning import-boundary audit for extracted_content_pipeline: passed.
- standalone debt audit: passed.
- ASCII Python check: passed.
- full extracted pipeline wrapper: 3999 passed, 10 skipped.
- git diff whitespace check: passed.
- body-aware local PR review: passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/extracted_pipeline_checks.yml` | 2 |
| `plans/PR-Content-Ops-Claim-Evidence-Result-Validate-CLI.md` | 118 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `scripts/validate_content_ops_claim_evidence_result.py` | 129 |
| `tests/test_content_ops_claim_evidence_result_validate_cli.py` | 245 |
| **Total** | **495** |
