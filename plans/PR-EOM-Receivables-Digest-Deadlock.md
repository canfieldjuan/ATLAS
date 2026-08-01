# PR-EOM-Receivables-Digest-Deadlock

Issue: #2255

## Why this slice exists

The full Atlas app cannot enable the EOM receivables API with the new
digest-only secret posture because one imported auth stack rejects raw token
material while the legacy invoicing stack still requires it. That deadlock made
`RECEIVABLES_API_ENABLED=true` incompatible with full-app startup and, because
lead intake is mounted by the same app, took down public lead intake when the
app failed to boot.

### Problem-derived contract

Root cause: the full Atlas app still validates the EOM receivables service API
with the legacy raw-token contract, while the newer EOM API stack rejects that
same raw token and requires only the SHA-256 digest on the Atlas side. Because
the full app imports the EOM API stack and also runs the legacy validator during
startup, no enabled configuration can satisfy both validators.

A correct fix must:

- Make the full-app invoicing settings capable of reading
  `ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN_SHA256`.
- Make the legacy `api.invoicing` auth boundary use the same digest-only Atlas
  storage model as `eom_api` receivables auth.
- Reject raw `ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN` material with a clear
  message that points operators to the digest variable.
- Continue authenticating callers that present the generated raw bearer by
  hashing the presented token and comparing it to the stored digest.
- Prove the full app starts with receivables enabled and digest-only config,
  and prove the public lead-intake route remains mounted in that state.
- Prove the raw-token-present and missing-digest failure modes no longer fall
  back to the old contradictory raw-token-required error.

Must not change:

- Receivables payment, invoice, allocation, deposit, or MCP semantics.
- EOM funnel enablement or `ATLAS_EOM_FUNNEL_*` configuration.
- Render service declarations or production env values.
- Public website lead-intake implementation.
- Stripe/SaaS billing.

## Scope (this PR)

Ownership lane: eom/receivables-auth
Slice phase: Production hardening

This PR is limited to the full-app EOM receivables auth/config migration from
raw-token storage to digest-only storage, plus regression tests for the exact
startup modes in #2255.

### Review Contract

- Acceptance criteria: full-app startup accepts
  `ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN_SHA256` without raw token
  material; raw `ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN` fails with one
  operator-facing digest error; missing digest fails with one digest-required
  error; caller raw bearer authentication hashes the presented generated token
  and compares it to the stored digest; `/api/v1/leads/intake` remains mounted
  when receivables is enabled with digest-only config.
- Reachability proof: subprocess imports `atlas_brain.main`, starts
  `TestClient(main.app)`, probes the security metadata endpoint, and inspects
  the full-app route table for `/api/v1/leads/intake`.
- Affected surfaces: full-app `InvoicingConfig`, legacy
  `api.invoicing.auth` receivables service-token validation, receivables route
  bearer authentication, and render/full-app regression tests.
- Risk areas: startup env validation, secret material exposure, legacy/new EOM
  auth contract parity, full-app route mounting, and existing receivables API
  auth behavior.
- Reviewer rules triggered: R1, R2, R5, R11, R12.

### Boundary-change enumeration

- Boundary path/seam: `atlas_brain.api.invoicing.auth` receivables service
  token validation and bearer authentication.
- Replaced-path behaviors: replace legacy Atlas-side raw-token storage and
  raw string comparison with digest-only storage and digest comparison.
- Guard-relevant fields:
  `ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN`,
  `ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN_SHA256`, and
  `ATLAS_INVOICING_RECEIVABLES_API_ENABLED`.
- Caller x input shape: generated raw bearer supplied in the HTTP
  `Authorization: Bearer ...` header hashes to the configured digest; raw token
  material in Atlas env is rejected.

### Deployed-config probing

- Deployed/default config values: full app starts with receivables enabled only
  when the digest env is present and raw token env is absent.
- Explicit value probe: subprocess full-app test sets a generated token digest
  and verifies startup plus route mounting.
- Absent value probe: subprocess full-app test enables receivables without a
  digest and verifies the single digest-required error.
- Default-session/default-context probe: subprocess tests import
  `atlas_brain.main` from a clean env snapshot with only the test overrides.
- Side-effect ordering: startup validation occurs before serving requests; the
  passing startup probe then proves lead-intake remains mounted in the same
  app instance.

### Files touched

- `atlas_brain/api/invoicing/auth.py`
- `atlas_brain/config.py`
- `plans/PR-EOM-Receivables-Digest-Deadlock.md`
- `tests/test_eom_render_profile.py`
- `tests/test_receivables.py`

## Mechanism

The broad invoicing settings model now carries the same
`receivables_service_token_sha256` field that the newer EOM API stack already
uses. The legacy invoicing auth boundary rejects configured raw token material,
validates the stored digest with the shared generated-token digest validator,
and authenticates callers by hashing the presented generated bearer before
performing constant-time digest comparison. The regression tests cover both the
legacy receivables route behavior and the full app startup path that imports
the EOM API stack and mounts public lead intake.

## Intentional

- `InvoicingConfig` keeps the deprecated raw token field for compatibility with
  settings shape, but adds `receivables_service_token_sha256` as the Atlas-side
  secret representation.
- `api.invoicing.auth.validate_receivables_api_config` now rejects raw token
  material and validates the configured digest with the shared generated-token
  digest validator.
- `api.invoicing.auth.require_receivables_api` now hashes the caller-provided
  generated bearer through the shared receivables token helper and compares the
  digest with `hmac.compare_digest`.
- Existing legacy-router receivables tests now use generated caller tokens plus
  stored digests.
- Full-app subprocess regression tests cover digest-only startup, raw-token
  rejection, missing-digest rejection, and the lead-intake route remaining
  mounted while receivables is enabled.

## Deferred

- Production systemd or Render environment changes. Operators still need to set
  the digest and remove the temporary disabled workaround outside this code PR.
- Reducing startup blast radius so a receivables config error cannot kill
  unrelated public lead intake. This PR proves the immediate deadlock is gone;
  subsystem isolation is a separate architectural slice.
- Removing the deprecated raw token field from the broad `InvoicingConfig`
  settings model after all callers and docs have migrated.

## Cold diff reconstruction

- Changed `atlas_brain/config.py` to add
  `receivables_service_token_sha256` to the full-app invoicing settings while
  marking `receivables_service_token` as deprecated caller-side material.
- Changed `atlas_brain/api/invoicing/auth.py` so the legacy full-app
  receivables routes reject raw token material, require a generated-token
  digest when enabled, and compare caller bearer digests instead of raw values.
- Changed `tests/test_receivables.py` to validate raw-token rejection, bad
  digest rejection, digest-backed bearer authentication, and the existing HTTP
  receivables entrypoint under the new auth contract.
- Changed `tests/test_eom_render_profile.py` to add full-app subprocess probes
  for digest-only startup, raw-token failure, missing-digest failure, and
  lead-intake route reachability while receivables is enabled.

Contract match:

- Every runtime change traces to replacing the legacy raw-token validator with
  the digest-only Atlas-side representation required by #2255.
- The tests prove the full-app startup path, not just `main_eom.py`.
- No payment model, funnel model, Render declaration, Stripe billing, or
  public lead intake implementation changed.

Gaps: none in the immediate validator-deadlock scope. Startup blast-radius
isolation remains intentionally deferred.

## Verification

- `python -m py_compile atlas_brain/config.py atlas_brain/api/invoicing/auth.py tests/test_receivables.py tests/test_eom_render_profile.py`
- `python -m pytest tests/test_receivables.py::test_enabled_receivables_api_rejects_raw_missing_placeholder_and_bad_digests tests/test_receivables.py::test_receivables_api_is_fail_closed tests/test_eom_render_profile.py::test_full_app_starts_with_receivables_digest_only tests/test_eom_render_profile.py::test_full_app_rejects_raw_receivables_token_with_single_digest_message tests/test_eom_render_profile.py::test_full_app_rejects_enabled_receivables_without_digest_once -q`
- `python -m pytest tests/test_eom_render_profile.py tests/test_receivables.py -q` — 94 passed, 6 skipped, 1 warning from `torch`/`pynvml`.
- `git diff --check`

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/api/invoicing/auth.py` | 53 |
| `atlas_brain/config.py` | 12 |
| `plans/PR-EOM-Receivables-Digest-Deadlock.md` | 190 |
| `tests/test_eom_render_profile.py` | 115 |
| `tests/test_receivables.py` | 28 |
| **Total** | **398** |
