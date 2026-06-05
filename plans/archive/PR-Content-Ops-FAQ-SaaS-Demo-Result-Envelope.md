# PR-Content-Ops-FAQ-SaaS-Demo-Result-Envelope

## Why this slice exists

PR #999 added a cleanup path for the SaaS FAQ demo seeder, but review caught a
result-contract drift: seed mode reports failures through an `errors` list while
cleanup mode reports a scalar `error`. Operators and scripts should be able to
read one fail-closed result envelope regardless of mode.

## Scope (this PR)

Ownership lane: content-ops/faq-generator
Slice phase: Production hardening

1. Normalize the cleanup result payload to use the same `errors: list[str]`
   convention as seed mode.
2. Update the human-readable cleanup summary to report the shared error count.
3. Pin success and failure tests for the cleanup envelope.

### Files touched

- `plans/PR-Content-Ops-FAQ-SaaS-Demo-Result-Envelope.md`
- `scripts/seed_content_ops_faq_saas_demo.py`
- `tests/test_content_ops_faq_saas_demo_corpus.py`

## Mechanism

`cleanup_saas_demo_faq` now builds an `errors` list from the delete-status
parser and derives `ok` from whether that list is empty. `_print_result` reads
the same `errors` key that seed mode already exposes. The focused tests assert
the exact success envelope and the bad-delete branches.

## Intentional

- No live database run in this slice; this only normalizes the result contract
  around an already-tested fake pool boundary.
- No seed-path behavior change beyond preserving the existing shared
  `errors` convention.
- No route or search-projection changes; this is operator-script output
  hardening.

## Deferred

Parked hardening: none. `HARDENING.md` has no active content-ops FAQ entries
touching this script.

Live database verification remains deferred until a real database URL is
available in the local environment.

## Verification

- `python -m pytest tests/test_content_ops_faq_saas_demo_corpus.py -q` (12
  passed)
- `python -m py_compile` with `scripts/seed_content_ops_faq_saas_demo.py`
  and `tests/test_content_ops_faq_saas_demo_corpus.py`
- `python` `scripts/audit_plan_code_consistency.py` with
  `plans/PR-Content-Ops-FAQ-SaaS-Demo-Result-Envelope.md`
- `python scripts/audit_extracted_pipeline_ci_enrollment.py .` (121 matching
  tests enrolled)
- `git diff --check`
- `bash` `scripts/validate_extracted_content_pipeline.sh`
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py
  extracted_content_pipeline`
- `python scripts/audit_extracted_standalone.py --fail-on-debt`
- `bash` `scripts/check_ascii_python.sh`
- `bash` `scripts/run_extracted_pipeline_checks.sh` (2456 passed, 6 skipped)

## Estimated diff size

| Area | LOC |
| --- | ---: |
| Plan doc | ~70 |
| Script/test changes | ~20 |
| **Total** | **90** |
