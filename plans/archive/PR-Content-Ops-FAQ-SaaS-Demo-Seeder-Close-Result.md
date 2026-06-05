# PR-Content-Ops-FAQ-SaaS-Demo-Seeder-Close-Result

## Why this slice exists

PR-Content-Ops-FAQ-SaaS-Demo-Seeder-Runtime-Result made runtime failures write a
safe result artifact, but review correctly parked one narrower edge: if a SaaS
demo seed or cleanup succeeds and `pool.close()` then raises, the seeder can
replace the successful operation payload with a generic runtime failure. That
makes it harder for operators to know whether the demo data was actually seeded
or cleaned up.

This slice drains that `HARDENING.md` item by preserving the primary operation
payload and reporting close failures as lifecycle metadata.

## Scope (this PR)

Ownership lane: content-ops/faq-generator
Slice phase: Production hardening

1. Add `pool_close` lifecycle metadata to SaaS demo seeder result payloads.
2. Preserve successful seed and cleanup payloads when close fails, while marking
   the overall result failed.
3. Keep post-validation runtime exception handling for pool creation and
   operation failures unchanged.
4. Add focused seed and cleanup tests for close-failure result preservation.
5. Remove the drained `HARDENING.md` pool-close item.

### Files touched

| File | Purpose |
|---|---|
| `plans/PR-Content-Ops-FAQ-SaaS-Demo-Seeder-Close-Result.md` | Plan contract for this hardening slice. |
| `HARDENING.md` | Remove the drained pool-close result preservation item. |
| `scripts/seed_content_ops_faq_saas_demo.py` | Preserve primary seeder payloads and attach pool-close metadata. |
| `tests/test_content_ops_faq_saas_demo_corpus.py` | Cover seed and cleanup close-failure result preservation. |

## Mechanism

`_run(...)` now builds the primary seed/cleanup payload before closing the pool.
The close call is wrapped separately and converted into:

```python
{"ok": false, "attempted": true, "error": {"type": "...", "message": "..."}}
```

`_with_pool_close_result(...)` attaches that lifecycle object to the primary
payload and combines `ok = payload["ok"] and pool_close["ok"]`. If seeding or
cleanup itself raises, the existing `main()` runtime handler still writes the
runtime failure artifact.

Close-failure messages use the same configured database URL redaction helper as
runtime failures before the lifecycle metadata is written to disk or stdout.

## Intentional

- No generator, database write, cleanup SQL, search projection, or route
  behavior changes.
- This preserves successful seed/cleanup details but does not add retry or
  reconnect behavior.
- The existing runtime exception artifact remains responsible for pool creation
  and operation failures.

## Deferred

- Parked hardening: none. The matching `HARDENING.md` item is removed by this
  slice.

## Verification

- `python -m pytest tests/test_content_ops_faq_saas_demo_corpus.py -q` — 18 passed.
- `python -m py_compile scripts/seed_content_ops_faq_saas_demo.py tests/test_content_ops_faq_saas_demo_corpus.py` — passed.
- `python scripts/audit_plan_code_consistency.py plans/PR-Content-Ops-FAQ-SaaS-Demo-Seeder-Close-Result.md` — passed.
- `git diff --check` — passed.
- `python scripts/audit_extracted_pipeline_ci_enrollment.py .` — 122 matching tests enrolled.
- `bash scripts/validate_extracted_content_pipeline.sh` — passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` — passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` — passed.
- `bash scripts/check_ascii_python.sh` — passed.
- `bash scripts/run_extracted_pipeline_checks.sh` — 2503 passed, 7 skipped, 1 warning.

## Estimated diff size

| Area | LOC |
|---|---:|
| Plan doc | 89 |
| Hardening note removal | 11 |
| Script | 71 |
| Tests | 119 |
| **Total** | **290** |
