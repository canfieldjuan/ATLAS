# PR-FAQ-Macro-Writeback-Credential-Setup-CLI

## Why this slice exists

The live FAQ macro writeback proof succeeded, but the setup path still required
manual credential insertion to provision the tenant Zendesk credential row. PR
#1607 fixed the underlying pool contract so the credential service can accept
the raw asyncpg pool shape used by operator scripts. The remaining upstream gap
is delivery: future live proofs should not rely on ad hoc SQL for encrypted
tenant credential setup.

This slice turns the fixed service seam into a small operator command. It is
production hardening because it removes a manual secret-handling step from the
live-proof workflow without changing macro publish behavior.

The diff is over the 400 LOC soft target because the command is
security-sensitive: the slice includes direct negative coverage for dry-run
no-DB behavior, missing config, endpoint guard mismatch, missing write
confirmation, pool-open failure, validation failure, generic write failure, and
lazy asyncpg import. Review fixes added same-class coverage for short-token
redaction, close-failure cleanup after success and handled failure, and
encryption setup diagnostics. Splitting those tests out would make the operator
command look safer than the PR proves.

## Scope (this PR)

Ownership lane: content-ops/faq-macro-writeback
Slice phase: Production hardening

1. Add an operator CLI that provisions one tenant-scoped Content Ops Zendesk
   credential by calling the existing encrypted credential service.
2. Read Zendesk credentials through the typed Atlas config path by default, not
   raw environment access, and keep the database DSN as an explicit operator
   argument consistent with the live proof scripts.
3. Add a dry-run mode that validates account id and configured Zendesk endpoint
   without opening a database pool or writing credentials.
4. Require `--confirm-write` for actual credential writes.
5. Emit display-safe JSON only: account id, endpoint, label, token prefix, and
   status; never print the API token, ciphertext, encryption key id, or DSN.
6. Add focused tests for dry-run, missing config, successful upsert delegation,
   and sanitized error output.
7. Suppress complete short-token prefixes at the shared credential service
   boundary so API and CLI display paths cannot expose a whole token.
8. Treat pool-close failures as cleanup warnings so they cannot mask a
   successful write or a handled safe failure payload.
9. Distinguish encryption setup failures from invalid Zendesk credentials.

### Files touched

- `.github/workflows/atlas_content_ops_generated_assets_checks.yml`
- `atlas_brain/_content_ops_zendesk_credentials.py`
- `plans/PR-FAQ-Macro-Writeback-Credential-Setup-CLI.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `scripts/setup_content_ops_zendesk_credentials.py`
- `tests/test_content_ops_zendesk_credentials.py`
- `tests/test_setup_content_ops_zendesk_credentials.py`

### Review Contract
- Acceptance criteria:
  - [ ] The setup CLI can validate config in dry-run mode without DB access.
  - [ ] The setup CLI delegates actual writes to the existing encrypted
        credential service and never hand-builds SQL.
  - [ ] Actual writes require an explicit confirmation flag.
  - [ ] Output is display-safe and tests prove token, ciphertext, encryption
        key id, DSN, and operator email values are not emitted on success or
        failure.
  - [ ] Short tokens are not emitted in full as display prefixes.
  - [ ] Pool-close failures cannot override a successful write or handled
        write failure.
  - [ ] Encryption setup failures return a distinct non-secret diagnostic.
  - [ ] Missing/incomplete Zendesk config fails before DB access with a
        non-secret diagnostic.
  - [ ] New tests are enrolled in the extracted pipeline wrapper and generated
        assets workflow path/test lists.
- Affected surfaces: scripts, config consumption, credential setup, CI.
- Risk areas: security, tenant isolation, operator error, CI enrollment.
- Reviewer rules triggered: R1, R2, R3, R6, R10, R11, R12, R14.

## Mechanism

The command accepts `--database-url`, `--account-id`, optional `--label`, and an
optional exact `--expected-zendesk-base-url` guard. Zendesk email/token/endpoint
come from `settings.b2b_campaign` through
`zendesk_macro_credentials_from_config`, matching the existing macro writeback
host path.

With `--dry-run`, the command validates the account id and config-derived
endpoint, applies the optional expected-endpoint guard, then emits a safe
preview and exits before importing asyncpg or opening a pool. Without
`--dry-run`, it still refuses side effects until `--confirm-write` is present.
With confirmation, it creates a raw asyncpg pool from the explicit DSN and
delegates the write to `upsert_zendesk_credentials`. The returned display
record is the only source for success output.

Errors are mapped to concise reason codes and sanitized messages. The command
does not log or print raw tokens, ciphertext, encryption key ids, or the DSN.
Pool close is best-effort after the write attempt: failures become a
`database_pool_close_failed` warning on the already-determined payload instead
of replacing the payload with a traceback.

The token-prefix helper suppresses prefixes when a token is not longer than the
display prefix length. That fix lives in the shared credential service as well
as the CLI preview path so the hosted API cannot display a complete short token
either.

## Intentional

- No API route or frontend credential form changes. The hosted route already
  exists; this slice is the operator path for live proofs and setup.
- No live Zendesk network validation. The macro publish proof already validates
  the stored credential when it talks to Zendesk; this command only stores the
  tenant credential.
- No raw environment reads. Existing config aliases remain in
  `atlas_brain/config.py`; the script consumes typed settings only.
- No migration changes. The credential table and encrypted storage semantics
  are already present.

## Deferred

- A product UI for customer self-service credential setup remains future work.
- A live operator run with real credentials remains manual and should be
  recorded in #1338 only when the operator chooses to run it.

Parked hardening: none.

## Verification

- python -m pytest tests/test_setup_content_ops_zendesk_credentials.py tests/test_content_ops_zendesk_credentials.py -q -
  21 passed.
- python -m pytest tests/test_content_ops_zendesk_credentials_api.py tests/test_atlas_content_ops_macro_writeback.py -q -
  23 passed.
- python -m py_compile scripts/setup_content_ops_zendesk_credentials.py atlas_brain/_content_ops_zendesk_credentials.py -
  passed.
- python scripts/audit_extracted_pipeline_ci_enrollment.py - OK: 183
  matching tests are enrolled.
- Explicit ASCII check for touched Python files - passed.
- Pending before push: bash scripts/local_pr_review.sh.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_content_ops_generated_assets_checks.yml` | 5 |
| `atlas_brain/_content_ops_zendesk_credentials.py` | 9 |
| `plans/PR-FAQ-Macro-Writeback-Credential-Setup-CLI.md` | 148 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `scripts/setup_content_ops_zendesk_credentials.py` | 293 |
| `tests/test_content_ops_zendesk_credentials.py` | 25 |
| `tests/test_setup_content_ops_zendesk_credentials.py` | 402 |
| **Total** | **883** |
