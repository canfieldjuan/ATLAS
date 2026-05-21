# PR-Content-Ops-FAQ-Archive-Runbook

## Why this slice exists

The CFPB FAQ smoke now reports source-profile counts, but the operator docs
still show only a tiny three-row demo and do not explain how to read those
counts during larger runs. That leaves the same confidence gap the FAQ scale
work is meant to close: a weak run should show whether the source archive was
sparse or whether FAQ generation failed after enough usable rows were loaded.

This slice documents the 1,000-row CFPB FAQ smoke and the `source_profile`
fields that identify source-density issues.

## Scope (this PR)

1. Update the extracted package README with a 1,000-row CFPB FAQ smoke command.
2. Document how to interpret `source_profile` counts and stop reasons.
3. Mirror the same operator guidance in the host install runbook.
4. Keep this docs-only; no runtime behavior changes.

### Files touched

- `plans/PR-Content-Ops-FAQ-Archive-Runbook.md`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/docs/host_install_runbook.md`

## Mechanism

The docs will show the large-run command with `--limit 1000`,
`--max-rows-scanned 5000`, `--output-source-rows`, `--output-markdown`, and
`--json` so operators get the FAQ artifact, source rows, and summary payload in
one run.

The interpretation notes will map source-profile fields to next actions:
`usable_source_count` below the requested limit means source prep failed early,
`missing_narrative_count` and `missing_complaint_id_count` point at skipped CFPB
rows, and `stop_reason="max_rows_scanned"` means the scan cap, not the FAQ
generator, limited the run.

## Intentional

- No code changes. PR #725 already added the CFPB source-profile payload.
- No live CFPB command is run as verification; docs changes should not depend on
  external API availability.
- The command writes local artifacts instead of implying hosted storage.

## Deferred

- Hosted UI surfacing for source-profile counts remains a separate portfolio or
  host-dashboard slice.
- A local static CFPB archive reader remains separate if operators need fully
  offline CFPB extraction beyond generic source-file scale smoke runs.

## Verification

- `bash scripts/local_pr_review.sh origin/main` passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | 65 |
| README | 24 |
| Host runbook | 23 |
| **Total** | **112** |
