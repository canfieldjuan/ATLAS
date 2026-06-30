# PR-Deflection-Landing-Demo-Example

## Why this slice exists

The landing demo contract-derivation arc is now tracked in ATLAS #1835 and
atlas-portfolio #386. The root cause is that the public landing demo still
depends on hand-authored, parallel fixtures while production treats the free
Snapshot as a projection of the report artifact. ATLAS already has a synthetic
deflection report example and a derived snapshot example, but the generator's
`--check` gate only guards the snapshot file. That leaves the report side of
the demo handoff outside the producer -> generated artifact -> CI drift gate
arc.

This slice fixes the root for the ATLAS half by making the existing synthetic
producer example generator own both committed public demo artifacts: the report
artifact and the snapshot derived from that same artifact.

The diff is slightly above the 400 LOC soft cap because the generator now owns
two artifacts and the tests prove both stale and missing failure branches for
each artifact. Splitting the report check from the snapshot check would leave
the drift gate only half-proven.

## Scope (this PR)

Ownership lane: deflection/landing-demo-contract-derive
Slice phase: Workflow/process

1. Extend the existing synthetic deflection example generator so `--check`
   validates both the report artifact and its derived snapshot projection.
2. Add generator tests for report-output writes, report-output stale/missing
   failures, and the report -> snapshot projection invariant.
3. Enroll the generated report example in the extracted-pipeline workflow path
   filter so changes to the ATLAS-owned demo artifact trigger the check.

### Files touched

- `.github/workflows/extracted_pipeline_checks.yml`
- `docs/frontend/content_ops_faq_deflection_checkout_contract.md`
- `docs/frontend/content_ops_faq_report_contract.md`
- `plans/PR-Deflection-Landing-Demo-Example.md`
- `scripts/generate_deflection_snapshot_example.py`
- `tests/test_content_ops_faq_deflection_snapshot_example_generator.py`

### Review Contract

- Acceptance criteria: the committed deflection report example is generated
  from the same synthetic producer path as the snapshot, `--check` fails if
  either committed artifact drifts, the legacy `--output` / `--snapshot-output`
  scratch path remains snapshot-only unless `--report-output` is also provided,
  and the snapshot is proven to derive from the generated report artifact.
- Affected surfaces: frontend/docs deflection example artifacts and extracted
  pipeline CI drift checks.
- Risk areas: public-demo PII boundary and accidental divergence between the
  report artifact and free Snapshot projection.
- Triggered reviewer rules: R1, R2, R3, R8, R9, R14.

## Mechanism

`scripts/generate_deflection_snapshot_example.py` keeps the existing frozen
synthetic support-ticket records and producer path. It now renders two expected
payloads from that one source:

- `docs/frontend/content_ops_faq_deflection_report_example.json` from
  `build_deflection_report_artifact(...)`.
- `docs/frontend/content_ops_faq_deflection_snapshot_example.json` from
  `build_deflection_snapshot(report_payload, ...)`.

The CLI checks or writes both committed files by default. Tests use temporary
report and snapshot outputs to prove write and check behavior without mutating
the committed artifacts, while the normal no-arg `--check` mode is the
CI/load-bearing drift gate.

For backward compatibility, the legacy `--output` alias remains snapshot-only
when used without `--report-output`. That keeps scratch snapshot generation from
checking or rewriting the tracked report example as a side effect.

## Intentional

- This does not create a new demo-specific JSON path; it promotes the existing
  frontend deflection report/snapshot examples into the generated public demo
  source of truth so atlas-portfolio #386 can consume them without another
  parallel artifact.
- The legacy `--output` alias is intentionally snapshot-only unless the caller
  explicitly opts into two-artifact mode with `--report-output`.
- The synthetic fixture remains compact. Portfolio can still render a curated
  subset in the locked preview; the invariant here is derivation, not section
  count or marketing density.

## Deferred

- atlas-portfolio #386 will vendor/consume these generated artifacts, delete
  its independent hand-authored demo fixtures, and add the landing projection
  equality test.

Parked hardening: none.

## Verification

- Command passed: python -m py_compile scripts/generate_deflection_snapshot_example.py tests/test_content_ops_faq_deflection_snapshot_example_generator.py.
- Command passed: python scripts/generate_deflection_snapshot_example.py --check; report and snapshot examples current.
- Command passed: python scripts/generate_deflection_snapshot_example.py --output /tmp/atlas-pr1836-snapshot-only.json; wrote only the snapshot scratch file and did not dirty tracked artifacts.
- Command passed: python scripts/generate_deflection_snapshot_example.py --check --output /tmp/atlas-pr1836-snapshot-only.json; snapshot scratch file current.
- Command passed: python -m pytest tests/test_content_ops_faq_deflection_snapshot_example_generator.py tests/test_content_ops_faq_report_contract_docs.py -q -- 16 passed.
- Command passed: bash scripts/run_extracted_pipeline_checks.sh -- 4958 passed, 15 skipped.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/extracted_pipeline_checks.yml` | 2 |
| `docs/frontend/content_ops_faq_deflection_checkout_contract.md` | 4 |
| `docs/frontend/content_ops_faq_report_contract.md` | 4 |
| `plans/PR-Deflection-Landing-Demo-Example.md` | 116 |
| `scripts/generate_deflection_snapshot_example.py` | 194 |
| `tests/test_content_ops_faq_deflection_snapshot_example_generator.py` | 213 |
| **Total** | **533** |
