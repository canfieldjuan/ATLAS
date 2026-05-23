# PR-Content-Ops-FAQ-Scale-Gates

## Why this slice exists

The FAQ generator has been manually stress-tested through 50,000 real
CFPB-derived support-ticket rows, and the scale smoke already captures the
right diagnostics in its run summary. The remaining testing gap is that an
operator cannot ask the smoke to fail if the loaded row volume or rendered
ticket-source coverage is below the promised scale.

This slice turns those manual JSON checks into explicit scale gates so future
500, 1,000, 10,000, and 50,000 row generation proofs can be demonstrated by the
script exit code and summary artifact.

## Scope (this PR)

Ownership lane: content-ops/faq-generator

Slice phase: Robust testing.

1. Add optional minimum raw-row and FAQ ticket-source gates to
   `scripts/smoke_content_ops_faq_scale_run.py`.
2. Add an optional gate requiring every accepted ticket source to be represented
   in generated FAQ items.
3. Include scale-gate results in the run summary and make gate failures return a
   non-zero smoke exit.
4. Add focused tests for passing and failing scale-gate runs.

### Files touched

- `plans/PR-Content-Ops-FAQ-Scale-Gates.md`
- `scripts/smoke_content_ops_faq_scale_run.py`
- `tests/test_smoke_content_ops_faq_scale_run.py`

## Mechanism

The scale smoke still runs the existing FAQ CLI and reads the compact FAQ result
payload. After the CLI exits, the wrapper evaluates optional
scale gates against:

- `input_profile.raw_row_count`
- `result.ticket_source_count`
- `result.diagnostics.rendered_ticket_source_count`

When a gate fails, the smoke writes the same artifact set, marks the summary as
failed with failure.type=scale_gates, and exits 1. CLI failures and FAQ
output-check failures keep their existing behavior.

## Intentional

- No FAQ generation behavior changes. This is a robust-testing harness slice,
  not a ranking or rendering change.
- No checked-in large fixture. The existing synthetic test rows prove the gate
  logic; real 50,000-row CFPB artifacts stay local operator artifacts.
- No new hosted route or database dependency. The scale smoke remains an
  offline deterministic proof.

## Deferred

- Production hardening for async/background execution remains covered by the
  existing stress validation and closeout docs; this slice only strengthens the
  repeatable test harness.
- A hosted route smoke can reuse these gate semantics once #909 or a later
  host-input slice exposes the full support-ticket source flow.
- Parked hardening: none.

## Verification

- Focused pytest command passed: python -m pytest
  tests/test_smoke_content_ops_faq_scale_run.py -q. Result: 25 passed.
- Py compile command passed for the scale-smoke script and focused test file.
- Scale-gated smoke command passed against
  extracted_content_pipeline/examples/support_ticket_sources.csv with minimum
  raw rows 4, minimum ticket rows 4, and all-ticket-source rendering required.
  The run summary reported scale_gates.passed=true.
- Local PR review command passed: bash scripts/local_pr_review.sh --allow-dirty.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 85 |
| Scale smoke gates | 142 |
| Tests | 156 |
| **Total** | 383 |
