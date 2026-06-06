# PR-Content-Ops-FAQ-Search-Contract-Result

## Why this slice exists
The hosted FAQ search contract checker verifies the deployed route envelope, but
it only prints a console summary. The demo and review handoff need a deterministic
JSON artifact that records pass/fail, query metadata, and contract errors without
including bearer tokens.

## Scope (this PR)
Ownership lane: content-ops/faq-search

Slice phase: Vertical slice.

1. Add `--output-result` to the hosted FAQ search route contract checker.
2. Write a compact JSON artifact on success, contract failure, route failure, and
   local preflight failure.
3. Keep bearer tokens out of the artifact and console output.
4. Add focused tests for success and failure artifacts.

### Files touched

- `plans/PR-Content-Ops-FAQ-Search-Contract-Result.md`
- `scripts/check_content_ops_faq_search_route_contract.py`
- `tests/test_check_content_ops_faq_search_route_contract.py`

## Mechanism
The checker builds a result payload with `ok`, `phase`, route/query metadata,
optional `count`, and any validation or request errors. `--output-result` writes
that payload as deterministic JSON after the run. Missing local inputs use the
same writer before returning exit code `2`; route and contract failures return
exit code `1`; success returns `0`.

## Intentional
- This is not a new live HTTP smoke. It improves the existing deployed-route
  checker so external runs can be archived.
- The artifact does not include the bearer token or request headers.
- Invalid argparse usage can still exit before writing a result, consistent with
  the existing CLI behavior.

## Deferred
- Running this checker against a real deployed host remains environment-owned
  until a backend URL and bearer token are provided.
- Latency timing and HTTP status capture are deferred; this slice only records
  contract outcome and validation details.

## Verification
- `pytest tests/test_check_content_ops_faq_search_route_contract.py -q` passed with 28 tests.
- Python compile check for the route contract checker and focused tests passed.
- Preflight CLI proof with blank `--token` exited `2` and wrote a JSON result with `ok=false`, `phase=preflight`, and no bearer token.

## Estimated diff size
| Area | Estimated LOC |
|---|---:|
| Plan doc | 57 |
| Checker | 146 |
| Tests | 112 |
| **Total** | **315** |
