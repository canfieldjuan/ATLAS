# PR-Deflection-Process-Contract-Live-Runner-Preflight

## Why this slice exists

#1684 added the durable process-contract endpoint and checker after #1682 proved
that stale hosted API processes can make the paid full-report proof fail late,
after the operator has already created a fresh request, paid/unlocked it, and
fetched artifacts.

Root cause: the live full-report QA runner still did not invoke that checker.
The detector existed, but the paid proof path could bypass it and only discover
stale process drift at the `/report-model` or `/artifact` fetch stage.

This change fixes the root at the runner boundary by making the process-contract
check a required preflight inside `run_deflection_full_report_qa_live_runner.py`
before it fetches paid report-model/artifact endpoints. It does not attempt to
solve deployment restart automation or the separate atlas-portfolio buyer page
proof.

## Scope (this PR)

Ownership lane: content-ops/deflection-full-report-qa
Slice phase: Production hardening

1. Add a process-contract preflight to the ATLAS live full-report QA runner,
   using the checker from #1684 and the same hosted base URL/token inputs.
2. Stop before paid artifact fetches when the hosted process contract is stale,
   malformed, missing, or shape-drifted.
3. Keep local validation/PDF leak failures before network calls, so bad local
   proof inputs still fail without touching hosted endpoints.
4. Add focused tests proving success order, stale-process fail-closed behavior,
   and no paid endpoint fetch when the process preflight fails.

### Review Contract
- Acceptance criteria:
  - [ ] The live runner fetches `/deflection-reports/process-contract` before
        `/report-model` and `/artifact` on successful proof runs.
  - [ ] A stale same-version report-model shape fails the runner with a
        sanitized process-contract error.
  - [ ] When process-contract preflight fails, the runner does not fetch paid
        `/report-model` or `/artifact` endpoints.
  - [ ] Existing local input/PDF leak failures still fail before any network
        call.
  - [ ] Runner output remains sanitized: no bearer token, raw request ID, local
        path, endpoint URL, source IDs, evidence rows, Stripe IDs, or raw PDF
        text.
- Affected surfaces: ATLAS full-report QA live runner and its CI test file.
- Risk areas: false-green live proofs, output sanitization, network call order,
  checker reuse.
- Reviewer rules triggered: R1, R2, R5, R10, R12, R14.

### Files touched

- `plans/PR-Deflection-Process-Contract-Live-Runner-Preflight.md`
- `scripts/run_deflection_full_report_qa_live_runner.py`
- `tests/test_run_deflection_full_report_qa_live_runner.py`

## Mechanism

The live runner imports `check_process_contract` from the #1684 checker and
adds `--process-contract-path`, defaulting to the checker path. After argument
validation and local PDF/text/leak checks pass, `_process_contract_preflight()`
builds a checker namespace from the runner's base URL, token, timeout, and path.

If the checker returns any error, the runner emits a sanitized failure payload:

```json
{
  "ok": false,
  "fetches": {"process_contract": {"status": 200, "ok": false}},
  "errors": ["process contract preflight failed: ..."]
}
```

Only a green process contract allows the existing live report-model/artifact
fetches and downstream PDF/export scorecard to run. The process-contract summary
is included in successful `fetches` output so proof artifacts show that the
structural preflight ran.

## Intentional

- The runner still performs local PDF/read/leak validation before any network
  request. A bad local proof bundle should fail locally instead of contacting
  the hosted API.
- The process-contract preflight does not replace the paid proof. It proves the
  hosted process advertises the current structural contract; the existing paid
  artifact/PDF/export checks still prove value-level output correctness.
- No deployment restart/supervision automation. This slice makes stale process
  drift visible before paid artifact fetches; operations automation remains a
  separate hardening concern.
- No atlas-portfolio buyer hosted-result smoke. #1612 keeps that in the
  `atlas-portfolio/web` lane.

## Deferred

- Deployment restart/supervision automation remains a separate operations
  hardening slice if stale long-running processes recur.
- atlas-portfolio buyer hosted-result proof remains outside this repo/lane.
- Submit-smoke teaser-path alignment remains outside this runner slice.

Parked hardening: none.

## Verification

- `pytest tests/test_run_deflection_full_report_qa_live_runner.py -q`
  - 17 passed.
- Python compile check for the live runner and test file.
  - passed.
- Extracted pipeline CI enrollment audit.
  - OK: 185 matching tests are enrolled.
- Extracted pipeline check bundle.
  - extracted reasoning core: 295 passed.
  - extracted content pipeline: 4630 passed, 10 skipped, 1 existing torch
    warning.
- Pending before push: local PR review.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Deflection-Process-Contract-Live-Runner-Preflight.md` | 124 |
| `scripts/run_deflection_full_report_qa_live_runner.py` | 51 |
| `tests/test_run_deflection_full_report_qa_live_runner.py` | 89 |
| **Total** | **264** |
