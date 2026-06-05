# PR-Support-Ticket-Package-CLI-Scale-Validation

## Why this slice exists

PR-Support-Ticket-Provider-Package-Scale-Cap proved the package cap with a
direct inline Python call and explicitly deferred rerunning the same real-file
validation through the package-smoke CLI once #934 landed. #934 is now merged,
so this slice closes that deferred proof with the shipped operator-facing smoke
command.

This stays in the support-ticket provider lane. It does not change package
logic, hosted upload behavior, FAQ generation, database code, or LLM routing.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-input-provider
Slice phase: Robust testing

1. Run `scripts/smoke_content_ops_support_ticket_package.py` against the existing
   local CFPB-derived 1,000-row and 10,000-row support-ticket-shaped artifacts.
2. Verify the CLI summary matches the direct package proof: 1,000 rows package
   fully, 10,000 rows retain honest total counts while truncating generation
   inputs to 1,000 rows.
3. Record commands, summary counts, truncation warning, timing, and memory in
   the extraction validation trail.

### Files touched

- `docs/extraction/validation/support_ticket_package_cli_scale_validation_2026-05-24.md`
- `plans/PR-Support-Ticket-Package-CLI-Scale-Validation.md`

## Mechanism

The validation uses the CLI merged in #934:

```bash
python scripts/smoke_content_ops_support_ticket_package.py <source.jsonl> \
  --pretty \
  --require-included-rows
```

`--require-included-rows` proves the file did not merely load; at least one
usable support-ticket row survived packaging. The committed doc records the
result summaries instead of committing large ignored JSONL source artifacts or
generated summary files.

## Intentional

- No code changes. This is a robust-testing proof of the shipped CLI.
- No checked-in 1,000/10,000 row source or output artifacts. They remain ignored
  local validation inputs/outputs; the committed doc records the reproducible
  command and observed result.
- No hosted route assertion. Hosted upload/intake behavior remains a separate
  owner lane.

## Deferred

- Future PR: hosted upload/intake can surface the package CLI's truncation
  summary to users or operators.
- Future PR: choose product policy for customer-visible handling of files above
  the synchronous 1,000-row package cap.
- Parked hardening: none.

## Verification

- CLI package smoke for the 1,000-row local CFPB-derived JSONL file - passed;
  1,000 included rows, 0 truncated rows, 0 warnings.
- CLI package smoke for the 10,000-row local CFPB-derived JSONL file - passed;
  1,000 included rows, 9,000 truncated rows, 1 truncation warning.
- git diff whitespace check - passed.
- local PR review - passed after commit.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~75 |
| Validation doc | ~105 |
| **Total** | **~180** |
