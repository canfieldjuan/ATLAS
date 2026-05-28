# PR-FAQ-Live-Run-Blocker-Preflight

## Why this slice exists

Issue #1075 calls out that the FAQ hosted route e2e harness is ready, but the
live run has stayed deferred because the blocking hosted inputs are not explicit
enough. The next useful slice is not another result-envelope field; it is a
safe preflight that answers whether the deployed host inputs are present before
an operator attempts the DB write plus hosted route smoke.

## Scope (this PR)

Ownership lane: content-ops/faq-search
Slice phase: Functional validation.

1. Add a `--preflight-only` mode to the SaaS demo route e2e wrapper.
2. Emit a redacted required-input status summary for the DB URL, API base URL,
   bearer token, and FAQ account id.
3. Prove preflight-only success exits 0 without invoking seed, route, or cleanup
   subprocesses.
4. Update the runbook so issue #1075 has an explicit blocker-classification
   command before the live run.
5. Pin the documented preflight command and disambiguate the existing
   one-command smoke parser pin so multiple e2e wrapper commands can coexist in
   the runbook.

### Files touched

- `plans/PR-FAQ-Live-Run-Blocker-Preflight.md`
- `scripts/smoke_content_ops_faq_saas_demo_route_e2e.py`
- `tests/test_content_ops_faq_saas_demo_corpus.py`
- `tests/test_smoke_content_ops_faq_saas_demo_route_e2e.py`
- `docs/extraction/validation/content_ops_faq_saas_demo_route_case_runbook.md`

## Mechanism

The existing `_validate_args(...)` already knows the required hosted inputs.
This slice adds `_required_input_status(...)`, which reports only booleans:

```python
{"database_url": {"present": True}, "token": {"present": False}}
```

`--preflight-only` runs validation, writes the same result file path when
requested, prints JSON when `--json` is set, and exits before `run(args)`.
Missing inputs still exit 2; ready inputs exit 0 with seed/route/cleanup marked
as skipped for `preflight_only`.

## Intentional

- No live hosted route behavior changes.
- No secret values are printed or written to the result artifact.
- No DB or HTTP connectivity probe is added here; this is only the blocker
  classification step before the live smoke.
- Runbook commands now use the script-supported `ATLAS_ACCOUNT_ID` fallback so
  preflight classification does not false-negative when only the generic account
  variable is configured.

## Deferred

- Parked hardening: none.
- The actual hosted run remains the next operator step once the preflight shows
  all required inputs are present.

## Verification

- `python -m py_compile scripts/smoke_content_ops_faq_saas_demo_route_e2e.py tests/test_smoke_content_ops_faq_saas_demo_route_e2e.py tests/test_content_ops_faq_saas_demo_corpus.py`
  - passed.
- `python -m pytest tests/test_content_ops_faq_saas_demo_corpus.py tests/test_smoke_content_ops_faq_saas_demo_route_e2e.py -q`
  - 36 passed.
- `python scripts/smoke_content_ops_faq_saas_demo_route_e2e.py --database-url postgresql://example/atlas --base-url https://atlas.example.com --token token-123 --account-id acct-1 --preflight-only --json --output-result /tmp/faq-preflight-proof.json`
  - passed; result had `ok: true`, `phase: preflight`, all required inputs present, and seed/route/cleanup skipped for `preflight_only`.
- `git diff --check`
  - passed.
- `python scripts/audit_plan_doc.py plans/PR-FAQ-Live-Run-Blocker-Preflight.md`
  - passed.
- `python scripts/audit_plan_code_consistency.py plans/PR-FAQ-Live-Run-Blocker-Preflight.md`
  - passed.
- `bash scripts/check_ascii_python.sh`
  - passed.
- `ATLAS_CURRENT_PR_BODY_FILE=/home/juan-canfield/Desktop/atlas-pr-bodies/faq-live-run-blocker-preflight.md bash scripts/local_pr_review.sh`
  - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 93 |
| E2E wrapper | 40 |
| Corpus/runbook tests | 31 |
| Wrapper tests | 36 |
| Runbook | 34 |
| **Total** | **234** |
