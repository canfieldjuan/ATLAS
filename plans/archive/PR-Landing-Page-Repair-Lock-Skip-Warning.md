# PR-Landing-Page-Repair-Lock-Skip-Warning

## Why this slice exists

`PR-Landing-Page-Repair-Cost-Guard` added an advisory-lock guard around saved
landing-page repair. The extracted router still supports host pools that do not
expose `acquire()`, but that fallback silently skips duplicate-spend
protection. That was parked in `HARDENING.md`.

This slice drains that parked item with the thinnest real end-to-end behavior:
when repair continues without a lock because the pool has no `acquire()`, the
router emits an operator-visible warning.

## Scope (this PR)

Ownership lane: content-ops/landing-page-repair-lock-skip-warning

1. Log a warning when the landing-page repair lock fallback skips advisory
   locking because the injected pool has no `acquire()`.
2. Include the account id and landing-page id in the log record.
3. Add route-level coverage proving repair still works and the warning is
   emitted.
4. Remove the drained parked item from `HARDENING.md`.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Landing-Page-Repair-Lock-Skip-Warning.md` | Plan doc for this hardening-drain slice. |
| `HARDENING.md` | Remove the drained skipped-lock visibility item. |
| `extracted_content_pipeline/api/generated_assets.py` | Emit a warning when repair locking is skipped. |
| `tests/test_extracted_content_asset_api.py` | Cover the fallback warning through the repair route. |

## Mechanism

`_landing_page_repair_lock` already detects pools without `acquire()` and
continues without locking so lightweight extracted hosts do not need to adopt a
new pool port. This slice keeps that behavior, but logs a warning before the
fallback yields.

The focused test uses the existing no-`acquire()` editable pool and calls the
real repair route. It asserts the repair succeeds, the LLM is called once, and
the warning includes the account id and landing-page id.

## Intentional

- No new hard pool requirement for extracted hosts.
- No quota or lock redesign.
- No attempt to drain the remaining `hashtext()` or connection-hold parked
  items in this slice.

## Deferred

- `HARDENING.md` still tracks the theoretical `hashtext()` collision risk.
- `HARDENING.md` still tracks repair lock connection hold time across LLM
  latency.

## Parked hardening

- None added. This slice drains the skipped-lock visibility item and leaves the
  unrelated parked items in place.

## Verification

- Python compile for `extracted_content_pipeline/api/generated_assets.py` ->
  passed.
- Focused pytest for `tests/test_extracted_content_asset_api.py` -> 52 passed.
- Extracted content pipeline validation -> passed.
- Extracted reasoning import guard -> passed.
- Extracted standalone audit -> passed with 0 findings.
- ASCII Python policy -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan and hardening docs | ~70 |
| API warning | ~10 |
| API test | ~35 |
| Total | ~115 |
