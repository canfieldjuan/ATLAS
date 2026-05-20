# Content Ops CFPB Live Fetch Compatibility

## Why this slice exists

The CFPB source exporter is the current real public support-ticket-like data
path for AI Content Ops, but a live run failed with HTTP 403 because CFPB
rejects the script's custom Atlas User-Agent. A browser-compatible User-Agent
plus normal CSV accept/referer headers returns HTTP 200. This blocks using CFPB
as a real-public validation dataset for FAQ and source-row output checks.

## Scope (this PR)

1. Make CFPB HTTP request headers explicit, configurable, and
   browser-compatible by default.
2. Add a query-level narrative filter so the exporter asks CFPB for rows that
   can actually become source evidence.
3. Keep output as generic Content Ops source rows; do not add CFPB-specific
   logic to FAQ generation or other consumers.
4. Update tests for request headers, narrative query filtering, and CLI
   argument threading.
5. Remove the merged FAQ language coordination row and claim this CFPB slice.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-CFPB-Live-Fetch-Compat.md` | Plan doc for this slice. |
| `docs/extraction/coordination/inflight.md` | Replace stale FAQ row with this CFPB slice claim. |
| `scripts/export_content_ops_cfpb_sources.py` | Add configurable request headers and narrative query filter. |
| `tests/test_export_content_ops_cfpb_sources.py` | Regression coverage for request shape and CLI threading. |

## Mechanism

The exporter builds request headers through a small helper. The default
User-Agent is browser-compatible while still identifying Atlas Content Ops in
the token. The fetcher accepts optional user-agent and referer values so hosts
can override them without editing code.

The query builder adds the CFPB API's narrative filter by default. This is not a
FAQ coupling: the exporter already drops rows without complaint narratives, so
the query now matches the existing source-row contract and avoids scanning
irrelevant rows.

## Intentional

- No FAQ renderer changes. CFPB remains one source-row producer among many.
- No downloaded fixture checked into the repo. Live fetch is operator-run
  because CFPB is an external public service.
- No hard-coded company/search choice. Search filters remain CLI parameters.

## Deferred

- A reusable HTTP fetch abstraction for all future public source connectors can
  wait until a second connector needs the same header/timeout behavior.

## Verification

- Live probe before coding: CFPB returned 403 for the old custom User-Agent and
  200 for browser-compatible request headers.
- Focused pytest for the CFPB exporter: 11 passed.
- Python compile for the CFPB exporter and tests: passed.
- Live CFPB export smoke with a small row limit: passed, 3 JSONL rows exported.
- CFPB JSONL rows into the existing FAQ builder: passed; exposed a separate
  generic FAQ action-classifier quality issue for billing/account narratives.
- Local PR review: pending.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Content-Ops-CFPB-Live-Fetch-Compat.md` | 65 |
| `docs/extraction/coordination/inflight.md` | 4 |
| `scripts/export_content_ops_cfpb_sources.py` | 45 |
| `tests/test_export_content_ops_cfpb_sources.py` | 75 |
| **Total** | **189** |
