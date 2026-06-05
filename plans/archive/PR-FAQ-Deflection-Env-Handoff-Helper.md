# PR-FAQ-Deflection-Env-Handoff-Helper

## Why this slice exists

PR-FAQ-Deflection-Hosted-Submit-Handoff shipped the deployed submit smoke, but
the handoff is still blocked on operator setup: `ATLAS_API_BASE_URL`, a
B2B-Growth bearer token, and the matching `account_id` are not committed and
should not be pasted into chat. The operator also asked where to get those
values, which means the next validation step needs a repeatable local helper
instead of tribal knowledge.

This slice adds the smallest operator path that logs into the deployed ATLAS
API, verifies the returned account is eligible for the FAQ deflection hosted
proof, and writes the smoke's local `.env` values without printing secrets.

## Scope (this PR)

Ownership lane: content-ops/faq-deflection
Slice phase: Functional validation

1. Add an operator script that accepts a deployed ATLAS API URL and login
   credentials, calls `/api/v1/auth/login`, verifies `/api/v1/auth/me`, and
   prepares the local env keys needed by the hosted submit smoke.
2. Fail closed unless the API URL is hosted HTTPS, login returns an access
   token, `/auth/me` returns a usable account id, and the account is B2B Growth
   or better.
3. Preserve unrelated `.env` contents, refuse to overwrite existing ATLAS keys
   unless `--force` is passed, and ensure created env files are private.
4. Document the operator flow in the submit handoff runbook and enroll the new
   test in the extracted pipeline checks.

### Files touched

- `plans/PR-FAQ-Deflection-Env-Handoff-Helper.md`
- `.github/workflows/extracted_pipeline_checks.yml`
- `docs/extraction/validation/content_ops_faq_deflection_submit_handoff_runbook.md`
- `scripts/prepare_content_ops_deflection_env.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_prepare_content_ops_deflection_env.py`

## Mechanism

The helper uses stdlib `urllib` to post credentials to:

```text
POST /api/v1/auth/login
GET /api/v1/auth/me
```

It validates the login response envelope before using the access token, then
validates `/auth/me` against the deployed account gate used by the deflection
routes: B2B product and at least `b2b_growth` plan rank. On success it writes:

```dotenv
ATLAS_API_BASE_URL=https://...
ATLAS_B2B_JWT=<access token>
ATLAS_ACCOUNT_ID=<account id from /auth/me>
```

to `.env` by default. The script redacts the token from stdout and only prints
the host, account id, product, plan, and env path.

## Intentional

- The helper rejects localhost and non-HTTPS API URLs because it prepares the
  hosted handoff proof, not local route development.
- The helper writes the `account_id` returned by `/auth/me`; it does not accept
  an operator-provided account id because the submit smoke needs the id that
  matches the bearer token.
- The helper refuses existing key replacement without `--force` to avoid
  accidentally overwriting a working smoke configuration.
- The helper does not collect the signed blob URL, company name, or contact
  email. Those remain portfolio/demo-specific inputs for the hosted submit
  smoke.

## Deferred

- Parked hardening: `FAQ deflection blob submit DNS-rebinding TOCTOU` remains
  parked in `HARDENING.md`; this slice does not fetch customer blobs or touch
  the blob reader.
- Actual hosted submit proof remains deferred until the operator also provides
  `ATLAS_DEFLECTION_SUBMIT_BLOB_URL`, `ATLAS_DEFLECTION_COMPANY_NAME`, and
  `ATLAS_DEFLECTION_CONTACT_EMAIL`.
- Stripe webhook paid-unlock E2E remains deferred until after the unpaid hosted
  submit/snapshot/artifact handoff is run against the deployed API.

## Verification

- `python -m py_compile scripts/prepare_content_ops_deflection_env.py tests/test_prepare_content_ops_deflection_env.py` - passed.
- `python -m pytest tests/test_prepare_content_ops_deflection_env.py -q` - 14 passed.
- `python -m pytest tests/test_prepare_content_ops_deflection_env.py tests/test_smoke_content_ops_deflection_submit_handoff.py -q` - 24 passed.
- `python scripts/audit_extracted_pipeline_ci_enrollment.py` - passed, 137 matching tests enrolled.
- `bash scripts/check_ascii_python.sh` - passed.
- `bash scripts/run_extracted_pipeline_checks.sh` - passed, 2834 passed, 10 skipped, 1 warning.
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/faq-deflection-env-handoff-helper-pr-body.md` - passed.
- Live deployed login: not run in this checkout because the operator has not
  provided deployed credentials locally.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 111 |
| Env helper script | 348 |
| Tests | 313 |
| Runbook/check enrollment | 28 |
| **Total** | **852** |

The diff exceeds the 400 LOC target because this is a secret-handling
validation helper. The negative fixtures for URL safety, response-envelope
drift, plan/product gates, overwrite refusal, env preservation, and output
redaction need to ship with the script so it cannot false-green or leak tokens.
