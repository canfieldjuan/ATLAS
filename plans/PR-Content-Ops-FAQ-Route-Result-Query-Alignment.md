# PR-Content-Ops-FAQ-Route-Result-Query-Alignment

## Why this slice exists

The FAQ route concurrency smoke can execute the correct query from a case file while still writing the default/env query into the compact result artifact. That was parked in `HARDENING.md` during hosted route proof work because the per-case execution path was correct, but the summary artifact could mislead operators skimming a failed run.

## Scope (this PR)

Ownership lane: content-ops/faq-search
Slice phase: Production hardening

1. Derive the result artifact's top-level query from the active cases instead of raw CLI defaults.
2. Make mixed case-file query runs explicit instead of pretending one default query describes the whole run.
3. Remove the resolved `HARDENING.md` entry.

### Files touched

- `scripts/smoke_content_ops_faq_search_route_concurrency.py`
- `tests/test_smoke_content_ops_faq_search_route_concurrency.py`
- `HARDENING.md`
- `plans/PR-Content-Ops-FAQ-Route-Result-Query-Alignment.md`

## Mechanism

`_summary_payload` already receives the active case list. This slice adds a small helper that summarizes the unique case queries and uses that summary for the top-level artifact fields:

```python
query_summary = _query_summary(active_cases, from_case_file=args.case_file is not None)
```

Single-query runs keep a concrete query string. Multi-query case-file runs report a mixed query mode and query count so the compact artifact cannot imply the default query was used for every request.

## Intentional

- The request execution path is unchanged; `_run_one` already uses the active case query.
- The top-level `query` field remains a string for compatibility with existing JSON consumers.
- This slice does not add new route assertions beyond the artifact summary truthfulness issue being drained.

## Deferred

Parked hardening: none. This slice removes the relevant `HARDENING.md` entry rather than adding new debt.

## Verification

Passed locally:

```bash
python -m py_compile scripts/smoke_content_ops_faq_search_route_concurrency.py tests/test_smoke_content_ops_faq_search_route_concurrency.py
python -m pytest tests/test_smoke_content_ops_faq_search_route_concurrency.py -q
75 passed
```

To be run before PR:

```bash
bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/content-ops-faq-route-result-query-alignment-pr-body.md
```

## Estimated diff size

| Area | LOC |
|---|---:|
| Plan doc | 67 |
| Result summary helper | 39 |
| Tests | 74 |
| Hardening cleanup | 9 |
| **Total** | **189** |
