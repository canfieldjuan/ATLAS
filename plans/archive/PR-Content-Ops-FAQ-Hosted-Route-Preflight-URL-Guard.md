# Content Ops FAQ Hosted Route Preflight URL Guard

## Why this slice exists

PR #1091 recorded that the SaaS demo hosted-route proof preflight passed even
though `ATLAS_API_BASE_URL` pointed at `http://127.0.0.1:8000` with no server
listening. The proof then seeded and cleaned the database successfully but
failed all route requests with connection refused, which means the preflight was
only checking input presence and not whether the route target could represent a
deployed-host proof.

This promotes the parked hardening item `FAQ hosted route proof preflight
accepts local API URLs` from `HARDENING.md`.

## Scope (this PR)

Ownership lane: content-ops/faq-search

Slice phase: Production hardening

1. Make the SaaS demo hosted-route proof fail preflight when `--base-url` is a
   local host such as `localhost`, `127.*`, `0.0.0.0`, or `::1`.
2. Make malformed or scheme-less route base URLs fail preflight before any seed
   or route command runs.
3. Keep the existing preflight result artifact behavior so operators still get
   a JSON explanation before exit.
4. Update the SaaS demo route runbook to state that hosted proof mode requires a
   deployed HTTP(S) API host.
5. Remove the resolved `HARDENING.md` entry; leave unrelated parked items
   parked.

### Files touched

- `scripts/smoke_content_ops_faq_saas_demo_route_e2e.py` - hosted URL
  preflight validation.
- `tests/test_smoke_content_ops_faq_saas_demo_route_e2e.py` - fail-closed
  local/malformed URL fixtures and artifact assertion.
- `tests/test_content_ops_faq_saas_demo_corpus.py` - runbook parser
  integration expectation after shell-placeholder expansion.
- `docs/extraction/validation/content_ops_faq_saas_demo_route_case_runbook.md` -
  hosted base URL requirement.
- `HARDENING.md` - remove the resolved URL-preflight parked item.
- `plans/PR-Content-Ops-FAQ-Hosted-Route-Preflight-URL-Guard.md` - this plan.

## Mechanism

`_validate_args` keeps the existing required-input checks, then parses
`args.base_url` with `urllib.parse.urlparse`. If the URL lacks an `http` or
`https` scheme, lacks a host, or targets a local host, validation returns a
preflight error. `main()` already writes `_preflight_summary` and returns exit
code `2` before child commands run, so the slice uses that existing fail-closed
path instead of adding a second control flow.

## Intentional

- This does not probe network liveness in preflight. A liveness check would
  require tokened route behavior and can become flaky; this slice only blocks
  URL shapes that can never prove deployed-host behavior.
- This does not remove the separate parked top-level query-summary issue from
  `HARDENING.md`; that item is unrelated to local URL preflight.
- Non-local `http` URLs remain allowed. Some internal deployed environments use
  HTTP behind trusted networking, so the guard is scoped to local/non-absolute
  targets rather than enforcing HTTPS.

## Deferred

- Parked hardening resolved by this slice: `FAQ hosted route proof preflight
  accepts local API URLs`.
- Parked hardening considered but left parked: `FAQ route concurrency result
  top-level query can disagree with case-file query`; it affects compact result
  readability, not hosted target readiness.

## Verification

- Command: python -m py_compile scripts/smoke_content_ops_faq_saas_demo_route_e2e.py tests/test_smoke_content_ops_faq_saas_demo_route_e2e.py
  - Passed.
- Command: python -m pytest tests/test_smoke_content_ops_faq_saas_demo_route_e2e.py -q
  - Passed, 17 tests.
- Command: python -m pytest tests/test_content_ops_faq_saas_demo_corpus.py::test_saas_demo_route_case_runbook_e2e_command_matches_parser tests/test_content_ops_faq_saas_demo_corpus.py::test_saas_demo_route_case_runbook_preflight_command_matches_parser tests/test_smoke_content_ops_faq_saas_demo_route_e2e.py -q
  - Passed, 19 tests.
- Review follow-up command: python -m py_compile scripts/smoke_content_ops_faq_saas_demo_route_e2e.py tests/test_smoke_content_ops_faq_saas_demo_route_e2e.py && python -m pytest tests/test_smoke_content_ops_faq_saas_demo_route_e2e.py tests/test_content_ops_faq_saas_demo_corpus.py::test_saas_demo_route_case_runbook_e2e_command_matches_parser tests/test_content_ops_faq_saas_demo_corpus.py::test_saas_demo_route_case_runbook_preflight_command_matches_parser -q
  - Passed, 21 tests.
- Command: bash scripts/run_extracted_pipeline_checks.sh
  - First run failed on stale runbook parser expectations that validated the
    literal `$ATLAS_API_BASE_URL` placeholder as a URL.
  - Rerun passed with 2,627 passed, 8 skipped, 1 warning.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| URL preflight validation | ~30 |
| Tests | ~90 |
| Runbook | ~10 |
| Hardening cleanup | ~10 |
| Plan doc | ~90 |
| Total | ~230 |
