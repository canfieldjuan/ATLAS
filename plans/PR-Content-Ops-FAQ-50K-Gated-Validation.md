# PR-Content-Ops-FAQ-50K-Gated-Validation

## Why this slice exists

PR-Content-Ops-FAQ-Scale-Gates added enforceable scale gates to the offline FAQ
scale smoke. The next robust-testing step is to run those gates against the
existing real CFPB-derived 50,000-row source artifact and record the proof in
the validation trail.

This keeps the testing progression concrete: the generator does not just claim
50,000-row support from a manually inspected JSON file; the smoke command now
fails unless the raw row count, accepted ticket-source count, and rendered
ticket-source coverage all meet the requested 50,000-row bar.

## Scope (this PR)

Ownership lane: content-ops/faq-generator

Slice phase: Robust testing.

1. Run the gated FAQ scale smoke against the local 50,000-row CFPB source
   artifact.
2. Record the command, artifact paths, scale-gate result, timing, memory, and
   generated-output summary in a validation doc.
3. Update the existing scale-stress probe doc to point at the gated validation
   follow-up.

### Files touched

- `plans/PR-Content-Ops-FAQ-50K-Gated-Validation.md`
- `docs/extraction/validation/content_ops_faq_50k_gated_validation_2026-05-23.md`
- `docs/extraction/validation/content_ops_faq_scale_stress_probe_2026-05-23.md`

## Mechanism

The validation uses the committed scale-smoke script with the local
`tmp/faq_scale_stress_20260523/cfpb_50000_source_rows.jsonl` artifact and these
new gates:

- minimum raw source rows: 50,000
- minimum accepted ticket source rows: 50,000
- require all accepted ticket sources to be represented in generated FAQ items

The generated artifacts stay under `tmp/`; only the reproducible command and
summary metrics are committed.

## Intentional

- No runtime code changes. This is a robust-testing proof slice.
- No checked-in 50,000-row artifact. The source and generated output are large
  local validation artifacts, not repository fixtures.
- No hosted route or database dependency. #912 is covering host route proof in a
  separate lane.

## Deferred

- Hosted gated generation proof can follow after the support-ticket route lane
  settles.
- Background-job and production concurrency hardening remain covered by the
  existing stress closeout path.
- Parked hardening: none; the gated run surfaced no new correctness or
  visibility issue.

## Verification

- Gated 50,000-row smoke passed:
  - exit code 0
  - raw source rows 50,000
  - accepted ticket source rows 50,000
  - rendered ticket source rows 50,000
  - output checks 3/3 passed
  - wall time 1:41.86
  - max RSS 592,764 KB
- Local PR review passed: bash scripts/local_pr_review.sh --allow-dirty.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 82 |
| Validation doc | 135 |
| Stress probe pointer | 4 |
| **Total** | 221 |
