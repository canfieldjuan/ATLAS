# PR-Landing-Page-Repair-Lock-Hash-Key

## Why this slice exists

The saved landing-page repair lock used PostgreSQL advisory locks with
`hashtext()`. That hashes the account/draft lock key to 32 bits, which leaves a
small but avoidable chance that unrelated draft repairs collide and return a
false `409`.

This slice drains that parked `HARDENING.md` item by adding a 64-bit advisory
lock hash while preserving rolling-deploy compatibility with the legacy lock.

## Scope (this PR)

Ownership lane: content-ops/landing-page-repair-lock-hash-key

1. Fold the repair-lock namespace into the account/draft lock key string.
2. Use PostgreSQL `hashtextextended($1, $2)` for the widened advisory lock and
   unlock SQL.
3. Keep acquiring the legacy `hashtext()` lock during rollout so old and new
   app instances still contend for the same repair.
4. Add focused API tests that pin both the legacy compatibility lock and the
   widened lock SQL/arguments.
5. Remove the drained hash-key item from `HARDENING.md`.
6. Add `Owner/session` markers so other sessions know the remaining parked work
   belongs to the landing-page repair thread.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Landing-Page-Repair-Lock-Hash-Key.md` | Plan doc for this hardening-drain slice. |
| `HARDENING.md` | Remove the drained hash-key item and mark remaining repair items owner/session. |
| `extracted_content_pipeline/api/generated_assets.py` | Use a namespaced 64-bit advisory lock hash with legacy rollout compatibility. |
| `tests/test_extracted_content_asset_api.py` | Cover the widened lock SQL, legacy lock SQL, and account/draft lock key. |

## Mechanism

`_landing_page_repair_lock_key` now returns a namespaced account/draft key:
`content-assets:landing-page-repair:account=<account>:landing_page=<id>`.

The lock and unlock SQL now call PostgreSQL `hashtextextended($1, $2)` with a
stable seed, producing the bigint key expected by the single-argument advisory
lock functions. The same key and seed are used for unlock.

For rolling deploys, the new path still acquires the legacy
`hashtext(namespace, account/draft)` lock before the widened lock. That keeps
new instances mutually exclusive with old instances until all running processes
use the widened lock path. Both locks are released in reverse order.

## Intentional

- No behavior change to the repair endpoint.
- No change to the endpoint's lock acquisition/release lifecycle.
- No quota or audit trail work.

## Deferred

- `HARDENING.md` still tracks repair lock connection hold time across LLM
  latency. It is marked `Owner/session: landing-page repair session`; after
  this slice, this session will wait rather than continue draining parked items.
- `HARDENING.md` now tracks removal of the legacy compatibility lock after the
  widened-lock release is fully rolled out.

## Parked hardening

- Added: `Remove landing-page repair legacy lock after rollout`.
- Drained: `Consider a wider advisory-lock hash key`.

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
| Plan and hardening docs | ~90 |
| API lock SQL | ~30 |
| API tests | ~25 |
| Total | ~145 |
