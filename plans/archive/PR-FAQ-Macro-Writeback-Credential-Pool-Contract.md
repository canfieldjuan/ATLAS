# PR-FAQ-Macro-Writeback-Credential-Pool-Contract

## Why this slice exists

Issue #1606 tracks the one hardening gap found during the live FAQ macro
writeback proof. The product proof succeeded, but setup exposed that
`upsert_zendesk_credentials` only knew the Atlas `DatabasePool` wrapper
transaction shape. The live setup path used a raw asyncpg-style pool and failed
before writing tenant Zendesk credentials because that pool has `acquire()` and
connection-level `transaction()`, not a wrapper-level `transaction()` helper.

The repo's default credential API route still uses the Atlas wrapper. However,
the service helpers already mostly behave like a database-bound utility:
`list_zendesk_credentials`, `lookup_zendesk_credentials`, and
`revoke_zendesk_credentials` call `fetch*`/`execute` directly and work with the
wrapper or raw asyncpg pool shapes. The outlier is upsert because it needs one
transaction spanning account lock, revoke, and insert.

This slice makes that transaction boundary explicit and tested. It keeps the
wrapper-backed API route working while adding coverage for the raw asyncpg pool
shape that failed during setup.

## Scope (this PR)

Ownership lane: content-ops/faq-macro-writeback
Slice phase: Production hardening

1. Add a small internal transaction adapter for Zendesk credential writes.
2. Preserve the Atlas wrapper contract: if the object exposes
   `transaction()`, use it exactly as today.
3. Add raw asyncpg pool support for setup tooling: if the object exposes a
   context-manager `acquire()`, acquire a connection and open
   connection-level `transaction()`.
4. Fail closed with a typed credential setup error when neither supported pool
   shape is present.
5. Add regression coverage for the raw asyncpg-style pool shape that failed in
   the live setup path.
6. Keep macro publish, Zendesk network, API auth, migrations, and live proof
   behavior out of scope.

### Files touched

- `atlas_brain/_content_ops_zendesk_credentials.py`
- `plans/PR-FAQ-Macro-Writeback-Credential-Pool-Contract.md`
- `tests/test_content_ops_zendesk_credentials.py`

### Review Contract

- Acceptance criteria:
  - [ ] Wrapper-shaped `transaction()` upsert behavior remains covered.
  - [ ] Raw asyncpg-shaped `acquire()` plus connection `transaction()` upsert
        behavior is covered.
  - [ ] Unsupported pool shapes fail closed before account lookup/revoke/insert.
  - [ ] Upsert still encrypts the token and returns a display-safe record.
  - [ ] Existing lookup/list/revoke behavior is unchanged.
  - [ ] No account ids, FAQ ids, Zendesk macro ids, or token values from the
        live proof are added to fixtures or docs.
- Affected surfaces: tenant Zendesk credential service and its unit tests.
- Risk areas: credential write transaction atomicity, silently bypassing
  encryption, and widening database contract beyond the intended setup path.
- Reviewer rules triggered: R1, R2, R3, R8, R10, R14.

## Mechanism

`upsert_zendesk_credentials` delegates the account lock, revoke, and insert
block to an internal async context manager. The context manager first checks for
the Atlas wrapper's `transaction()` method and yields that transaction's
connection. If that method is absent, it uses the raw asyncpg pool pattern:
acquire a connection through the pool's async context manager, then open the
connection-level transaction.

Tests add a raw asyncpg-shaped fake whose pool has no `transaction()` method,
whose `acquire()` returns an async context manager, and whose connection owns
the query methods. That reproduces the live setup failure class without a real
database or secrets.

## Intentional

- No route or UI changes. The default API path already passes the Atlas wrapper
  through `get_db_pool()`.
- No live database integration test in this slice. The raw asyncpg shape is
  reproduced with a focused fake so CI can run it without secrets.
- No changes to lookup/list/revoke. They did not require a transaction and
  already use query methods compatible with both shapes.

## Deferred

- If setup tooling becomes a first-class operator command, add that command as
  a separate slice with dry-run behavior and no secret output.

Parked hardening: none.

## Verification

- python -m pytest tests/test_content_ops_zendesk_credentials.py -q
  - 8 passed.
- python -m pytest tests/test_content_ops_zendesk_credentials_api.py tests/test_atlas_content_ops_macro_writeback.py tests/test_content_ops_faq_macro_writeback_flow.py -q
  - 24 passed.
- python -m py_compile atlas_brain/_content_ops_zendesk_credentials.py tests/test_content_ops_zendesk_credentials.py
- python scripts/audit_extracted_pipeline_ci_enrollment.py
  - OK: 183 matching tests are enrolled.
- bash scripts/check_ascii_python.sh
  - ASCII check passed for extracted_content_pipeline Python files.
- Pending before push: local PR review via `scripts/push_pr.sh`.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/_content_ops_zendesk_credentials.py` | 61 |
| `plans/PR-FAQ-Macro-Writeback-Credential-Pool-Contract.md` | 113 |
| `tests/test_content_ops_zendesk_credentials.py` | 118 |
| **Total** | **292** |
