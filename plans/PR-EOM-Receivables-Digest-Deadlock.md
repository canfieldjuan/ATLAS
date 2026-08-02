# PR-EOM-Receivables-Digest-Deadlock

Issue: #2255

## Why this slice exists

The full Atlas app cannot enable the EOM receivables API with the new
digest-only secret posture because one imported auth stack rejects raw token
material while the legacy invoicing stack still requires it. That deadlock made
`RECEIVABLES_API_ENABLED=true` incompatible with full-app startup and, because
lead intake is mounted by the same app, took down public lead intake when the
app failed to boot.

Diff-budget override: this slice exceeds the 400 LOC target because the runtime
auth/config change, operator rollout instructions, full-app startup regression,
legacy HTTP-boundary auth regressions, plan contract evidence, and the known
unit-gate baseline shrink must land atomically. Splitting any one of those
pieces would either leave the production deadlock unfixed, leave operators with
stale secret instructions, or leave CI/review without proof that the exact
startup/request boundaries in #2255 are covered.

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
  and compares it to the stored digest for both legacy invoice action routes
  and receivables ledger routes; `/api/v1/leads/intake` remains mounted when
  receivables is enabled with digest-only config; operator docs define the
  maintenance-window migration from raw-token deployments.
- Reachability proof: subprocess imports `atlas_brain.main`, starts
  `TestClient(main.app)`, probes the security metadata endpoint, and inspects
  the full-app route table for `/api/v1/leads/intake`.
- Affected surfaces: full-app `InvoicingConfig`, legacy
  `api.invoicing.auth` receivables service-token validation,
  `api.invoicing.actions` invoice view/send/reminder routes,
  `api.invoicing.receivables` receivables ledger routes, EOM operator setup
  docs, and render/full-app regression tests.
- Risk areas: startup env validation, secret material exposure, legacy/new EOM
  auth contract parity, full-app route mounting, and existing receivables API
  auth behavior.
- Reviewer rules triggered: R1, R2, R3, R5, R11, R12.

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
  material in Atlas env is rejected; missing or malformed bearer receives
  401/503 instead of an open route.
- Boundary path/seam: `atlas_brain.api.invoicing.actions` router dependency
  for invoice view/send/reminder routes.
- Caller x input shape: callers reach `GET /api/v1/invoicing/{invoice_id}`,
  `POST /api/v1/invoicing/{invoice_id}/send`, and
  `POST /api/v1/invoicing/{invoice_id}/send-reminder` only with a generated
  raw bearer whose digest matches
  `ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN_SHA256`.
- Boundary path/seam: `atlas_brain.api.invoicing.receivables` router
  dependency for receivables readiness, open-invoice, allocation, payment,
  return/void, and deposit-batch routes.
- Caller x input shape: callers reach `/api/v1/receivables/*` only with the
  same generated raw bearer digest match plus route-specific actor headers
  where required.

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
  app instance. For legacy raw-token deployments, the enforced rollout order is
  maintenance-window only: disable the receivables API, remove raw Atlas token
  material, deploy the digest-only code to all Atlas API processes, provision
  `ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN_SHA256`, then re-enable the API
  after an authenticated readiness probe passes.

### Files touched

- `CLAUDE.md`
- `HARDENING.md`
- `atlas_brain/api/invoicing/auth.py`
- `atlas_brain/config.py`
- `plans/PR-EOM-Receivables-Digest-Deadlock.md`
- `tests/test_eom_render_profile.py`
- `tests/test_receivables.py`
- `tests/unit_gate_baseline.txt`

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
- Operator docs cover digest provisioning and the maintenance-window sequence
  required when old raw-token Atlas processes could overlap with digest-only
  Atlas processes.

## Deferred

- Production systemd or Render environment changes. Operators still need to set
  the digest and remove the temporary disabled workaround outside this code PR.
- Reducing startup blast radius so a receivables config error cannot kill
  unrelated public lead intake. This PR proves the immediate deadlock is gone;
  subsystem isolation is a separate architectural slice.
- Removing the deprecated raw token field from the broad `InvoicingConfig`
  settings model after all callers and docs have migrated.

Parking predicate: this validator slice parks hardening outside the full-app
receivables auth/config deadlock, generated-token digest admission,
legacy-invoicing caller authentication parity, operator digest provisioning
instructions, and regressions for those startup/request boundaries.

Parked hardening:

- `HARDENING.md` entry: "Isolate receivables config failures from unrelated
  public routes" parks startup blast-radius isolation so a future receivables
  config error cannot prevent unrelated public lead intake from serving.

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
- Changed `CLAUDE.md` to replace the stale raw-token Atlas setup with
  generated-token digest provisioning and maintenance-window rollout steps.
- Changed `HARDENING.md` to register the deferred startup blast-radius
  isolation item named by this plan.

Contract match:

- Every runtime change traces to replacing the legacy raw-token validator with
  the digest-only Atlas-side representation required by #2255.
- The tests prove the full-app startup path, not just `main_eom.py`.
- No payment model, funnel model, Render declaration, Stripe billing, or
  public lead intake implementation changed.

Gaps: none in the immediate validator-deadlock scope. Startup blast-radius
isolation remains intentionally deferred.

## Verification

- Existing CI failure log before this follow-up showed one regression:
  `tests/test_eom_render_profile.py::test_full_app_starts_with_receivables_digest_only`
  reported `lead_intake_route_mounted=False` while startup, security metadata,
  enabled projection, and digest projection all succeeded. The same log showed
  three stale unit-gate baseline entries in
  `tests/test_reasoning_graph_routing.py`; this follow-up removes those
  entries from `tests/unit_gate_baseline.txt`.
- `python3 -m py_compile atlas_brain/config.py atlas_brain/api/invoicing/auth.py tests/test_receivables.py tests/test_eom_render_profile.py`
  — exit 0.
- `/tmp/atlas-pr2259-venv/bin/python -m pytest tests/test_eom_render_profile.py::test_route_path_probe_follows_included_router_context -q`
  — 1 passed.
- Local `/tmp/atlas-pr2259-venv/bin/python -m pytest tests/test_receivables.py::test_receivables_api_is_fail_closed tests/test_receivables.py::test_legacy_invoicing_route_rejects_well_formed_mismatched_bearer -q`
  — not runnable in that venv because collection imports `torch`, which is not
  installed there; rerun on the Office PC worktree below.
- Office PC `/tmp/atlas-pr2259-test` worktree with this follow-up patch:
  `~/Desktop/Atlas/.venv/bin/python -m pytest tests/test_receivables.py::test_receivables_api_is_fail_closed tests/test_receivables.py::test_legacy_invoicing_route_rejects_well_formed_mismatched_bearer -q`
  — 2 passed in 3.87s.
- Office PC `/tmp/atlas-pr2259-test` worktree with the earlier follow-up patch:
  `~/Desktop/Atlas/.venv/bin/python -m pytest tests/test_eom_render_profile.py::test_full_app_starts_with_receivables_digest_only -q`
  — 1 passed in 35.33s.
- Office PC `/tmp/atlas-pr2259-test` worktree with the earlier follow-up patch:
  `~/Desktop/Atlas/.venv/bin/python -m pytest tests/test_receivables.py::test_enabled_receivables_api_rejects_raw_missing_placeholder_and_bad_digests tests/test_receivables.py::test_receivables_api_is_fail_closed tests/test_eom_render_profile.py::test_route_path_probe_follows_included_router_context -q`
  — 3 passed in 2.69s.
- Office PC `/tmp/atlas-pr2259-test` worktree with this follow-up patch:
  `~/Desktop/Atlas/.venv/bin/python -m pytest tests/test_eom_render_profile.py tests/test_receivables.py -q`
  — 96 passed, 6 skipped in 42.10s.
- Office PC `/tmp/atlas-pr2259-test` worktree with this follow-up patch:
  `git diff --check` and
  `~/Desktop/Atlas/.venv/bin/python -m py_compile atlas_brain/config.py atlas_brain/api/invoicing/auth.py tests/test_receivables.py tests/test_eom_render_profile.py`
  — exit 0.
- `python3 scripts/audit_plan_doc.py plans/PR-EOM-Receivables-Digest-Deadlock.md`
  — OK for required sections and review contract.
- `git diff --check` — exit 0.

## Estimated diff size

| File | LOC |
|---|---:|
| `CLAUDE.md` | 19 |
| `HARDENING.md` | 19 |
| `atlas_brain/api/invoicing/auth.py` | 53 |
| `atlas_brain/config.py` | 12 |
| `plans/PR-EOM-Receivables-Digest-Deadlock.md` | 273 |
| `tests/test_eom_render_profile.py` | 154 |
| `tests/test_receivables.py` | 53 |
| `tests/unit_gate_baseline.txt` | 3 |
| **Total** | **586** |
