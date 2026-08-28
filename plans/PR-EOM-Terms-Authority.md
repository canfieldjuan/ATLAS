# PR-EOM-Terms-Authority

## Why this slice exists

Issue #2156 and the operator's current-session decisions require Terms to be a
universal, versioned customer obligation that is independent from the existing
welcome/profile onboarding token and from the later recurring-residential card
flow. Atlas currently has no canonical Terms document/version authority, so a
consumer cannot name the exact English/Spanish residential/commercial snapshot
that was reviewed, published, or later accepted.

This is the first provider slice of the accepted onboarding-separation arc. It
adds only the private Atlas authority needed to author, review, publish, and
read an exact Terms version. Invitations, acceptance records, readiness,
Tracker/Website consumers, first-clean money state, and Stripe remain later
vertical slices. The source PDFs were inspected directly; their refined
bilingual copy is deliberately not seeded or published by this PR.

Diff-budget override: the schema guard, immutable publication service, private
HTTP boundary, and focused adversarial proof form one provider contract;
splitting them would merge either unreachable storage or an unguarded
legal-version API.

### Problem-derived contract

- Root cause: the code treats onboarding drafts/tokens as customer-handoff
  state, but no immutable, audience-specific, bilingual Terms version exists.
  A version owned by the mutable application role would not be immutable, and a
  route bound to the global Atlas pool would not be canonical when the slim EOM
  funnel uses its dedicated database. Without one guarded authority on that
  exact database, later acceptances could point to mutable prose or an
  ambiguous "current terms" claim.
- Correct fix must touch/change: add a controlled-DBA Atlas migration that binds
  the version tables/functions to the existing no-login EOM guard owner and
  grants the runtime only required DML; attest ownership, trigger identity,
  search path, constraints, and runtime privilege shape; add one service that
  validates the closed four-document bundle, hashes it canonically,
  creates/replays drafts, publishes/replays versions under the execution model
  below using one post-lock timestamp, and reads the current version; bind the
  private routes to the slim EOM funnel pool; expose the controlled migration
  through one allowlisted `./ops` preflight/apply operation with capability and
  rollback documentation; add real-entrypoint and disposable-PostgreSQL tests
  enrolled in the EOM workflow, plus focused validator/route/operations
  coverage.
- Must not change: no welcome draft, existing public-onboarding token, email,
  Customer/Site handoff, first-clean completion/candidate, payment, Stripe,
  calendar, Tracker, Website, employee timekeeping, commercial billing, or
  deployed Terms content behavior changes. No Terms row is seeded and no
  customer-facing content is published by deployment.

## Scope (this PR)

Ownership lane: eom-onboarding-terms
Slice phase: Vertical slice
Max files: 16

1. Persist draft/published Terms versions containing exactly residential and
   commercial documents in English and Spanish, each with Terms body, Services
   We Cannot Provide body, and the separate additional-work acknowledgment.
2. Canonically hash the complete bundle, make published versions immutable,
   and serialize publication through one current-version pointer.
3. Add authenticated private create, publish, and current-version endpoints
   with actor evidence and idempotent unchanged retries.
4. Prove the real FastAPI entrypoint returns the stored draft/current snapshot
   and rejects malformed, conflicting, missing-schema, malformed stored, and
   non-current replay inputs.
5. Make the controlled migration discoverable and safely operable through the
   repository's canonical `./ops` surface, including retention-first rollback.

### Review Contract

- Acceptance criteria:
  - `tests/test_eom_terms_authority.py` proves only the exact
    residential/commercial x en/es bundle reaches storage; missing, extra,
    blank, oversized, and surrogate-bearing fields fail before a write.
  - `EOMTermsAuthority.create_draft` returns the existing row only when the
    normalized payload hash and material-change flag match; the same label with
    different content returns a conflict.
  - `EOMTermsAuthority.publish` locks the target/current state, permits only a
    draft or the already-current published version, and an unchanged retry
    returns the same version without another state transition. A wall-clock
    timestamp captured after the advisory lock is used for both the immutable
    publication row and current-pointer selection.
  - controlled migration 396 leaves both relations and all three guard
    functions owned by `atlas_eom_handoff_owner`; the direct `atlas` login has
    only table SELECT plus the exact column INSERT/UPDATE privileges the
    service needs.
  - migration 396 prevents UPDATE/DELETE of a published version and prevents
    the current pointer from referencing a draft or being removed; runtime SQL
    cannot disable or replace those guard-owned triggers/functions.
  - route tests exercise `POST /api/v1/eom-funnel/terms/versions`,
    `POST /api/v1/eom-funnel/terms/versions/{id}/publish`, and
    `GET /api/v1/eom-funnel/terms/current` through the real router with service
    authentication and actor attribution.
  - the slim `main_eom` app binds the Terms dependency to
    `get_eom_funnel_db_pool`; a real entrypoint test pins that wiring.
  - a disposable-PostgreSQL test runs the actual migration as DBA and the
    service as the direct unprivileged `atlas` login, including concurrency,
    rollback, trigger, ownership, and ACL probes.
  - `./ops db controlled eom-terms-authority preflight|apply` dispatches only
    the pinned migration-396 runner from a normal worktree; the capability map
    and database runbook name the same commands and require retained history on
    application rollback.
  - the diff contains no seed INSERT and touches no existing onboarding,
    completion, money, calendar, or delivery implementation.
- Reachability proof: the mounted EOM FastAPI router is called by an ASGI client;
  the observable effects are a closed draft response, a published/current
  response, and stable API errors for rejected boundaries.
- Affected surfaces: migration chain, new Terms authority service, private EOM
  funnel API models/dependency/routes, guarded operations contract, focused
  tests.
- Risk areas: mutable published content, ambiguous document bundles, hash
  instability, duplicate labels, concurrent publication/timestamp ordering,
  missing schema, operational apply/rollback ambiguity, authorization/actor
  omission, accidental customer-visible publication.
- Reviewer rules triggered: R1, R2, R3, R4, R6, R7, R8, R9, R10, R11, R14.

### Publication execution model

- PostgreSQL is the closed execution surface. Draft creation linearizes on the
  unique version-label constraint. Publication runs in one database transaction
  at the connection's configured isolation level (the service does not
  override it) and first acquires one transaction-scoped advisory lock shared
  by every Terms publication; correctness does not depend on a weaker
  isolation level.
- While holding that lock, the service row-locks the target version, reads the
  wall clock once, row-locks the target version, reads the singleton pointer,
  performs at most one content-preserving draft-to-published transition, and
  then selects that version as current with that same timestamp. Commit is the
  linearization point for this service path. Its invariant is: publication and
  selection timestamps follow lock order, the singleton points to the last
  lock-ordered authority publication, every earlier published row remains
  immutable history, and an authority publish cannot commit its version
  transition without its pointer update.
- Duplicate requests for the current published version replay without a write.
  A late retry for a published version already superseded by another committed
  publication conflicts rather than rewinding the pointer. Distinct concurrent
  publications serialize; both may remain published history and the one whose
  lock-ordered transaction commits last is current.
- Validation failures happen before the transaction. Database errors,
  cancellation, process death, or connection loss before COMMIT release the
  transaction-scoped lock and roll back both version and pointer changes. A
  response lost after COMMIT is safe to retry under the replay/conflict rules
  above. There is no lease, expiry, background worker, or cross-database write.

### Boundary-change enumeration

- Boundary path/seam: private authenticated Terms draft creation, publication,
  and current-version read.
- Replaced-path behaviors: none; this is additive and existing onboarding paths
  remain authoritative for their current profile/handoff behavior.
- Guard-relevant fields: versionLabel, materialChange, residential/commercial,
  en/es, terms, servicesWeCannotProvide,
  additionalWorkAcknowledgement, actor id/name, version UUID.
- Caller x input shape: Tracker/private service bearer plus actor headers x one
  exact four-document JSON bundle; private bearer without actor is admitted
  only for the read route.

### Deployed-config probing

- Deployed/default config values: existing EOM funnel API service bearer only;
  no new environment value or fallback.
- Explicit value probe: authenticated create/publish/current route tests.
- Absent value probe: missing service bearer and missing actor tests retain the
  existing fail-closed dependencies.
- Default-session/default-context probe: every row is fixed to
  `effingham_maids`; no caller-supplied tenant is accepted.
- Side-effect ordering: validation and canonical hashing precede the
  transaction; publication changes the version to published before updating
  the locked current pointer; neither route sends mail or calls a provider.

### Files touched

- `.agent/capabilities.yaml`
- `.agent/runbooks/database.md`
- `.github/workflows/atlas_eom_lead_pipeline_checks.yml`
- `atlas_brain/eom_api/funnel.py`
- `atlas_brain/main_eom.py`
- `atlas_brain/services/eom_terms_authority.py`
- `atlas_brain/storage/migrations/396_eom_terms_authority.sql`
- `atlas_brain/storage/migrations/__init__.py`
- `ops`
- `plans/PR-EOM-Terms-Authority.md`
- `scripts/apply_eom_first_clean_completion_schema.py`
- `scripts/apply_eom_terms_authority_schema.py`
- `tests/test_eom_first_clean_completion_dba_runner.py`
- `tests/test_agent_operations_contract.py`
- `tests/test_eom_terms_authority.py`
- `tests/test_migrations_runner.py`

## Mechanism

Migration 396 stores the complete closed document bundle as JSONB beside a
canonical SHA-256 hash and actor evidence. Draft rows may transition once to
published; a trigger then makes them append-only. A singleton pointer selects
the current published version without rewriting history, using the same
post-lock wall-clock timestamp as the publication. The service owns all
normalization, replay, conflict, and transaction behavior. The funnel routes
only validate/reproject the private HTTP contract and map typed service errors.
The canonical `./ops` command exposes only the dedicated migration-396
preflight/apply wrapper and points operators to the retention-first runbook.

## Intentional

- The source PDFs are evidence for later copy authoring, not seed data. Exact
  refined English and Spanish copy still requires operator approval.
- No IP address, device fingerprint, invitation token, signature, acceptance,
  readiness, email, or card state is added in this authority-only slice.
- Every published version is retained. Publishing a new version moves only the
  current pointer; it does not mutate the prior published row.
- Application rollback retains the migration, guarded objects, runtime grants,
  and every stored version; destructive schema rollback is forbidden.
- The bundle is one version across both audiences and both languages so a
  single version/hash identifies the complete approved release.

## Deferred

- EOM Terms invitations, acceptances, executed-copy delivery, material-version
  readiness, and manual-existing-customer invitations: next Atlas Terms slice.
- Tracker proxy, Website Onboarding tab/public bilingual acceptance page, and
  admin readiness UI: subsequent consumer slices.
- Structured one-time/recurring service plan, exact first-clean balance/payment
  allocation, and recurring-residential Stripe SetupIntent: later accepted
  slices in issue #2156.
- Exact refined bilingual Terms publication: blocked on operator approval of
  the complete customer-visible copy.

Parking predicate: findings that require invitations, delivery, public-page,
payment, Stripe, or another new mechanism are parked into their accepted
follow-up slice unless they prove this authority stores or publishes false
state.

Parked hardening: none.

## Verification

- PASS: current focused Terms plus guarded-operations contract probes (31
  passed, 1 environment-gated test skipped).
- PASS: the same Terms suite against isolated PostgreSQL 16 (29 passed; no
  skip), controlled DBA-runner suite (21 passed), and migration-runner suite
  (117 passed, 1 unrelated environment-gated test skipped).
- PASS: adjacent EOM render-profile and capability-manifest suites (76 passed).
- PASS: targeted Ruff lint, canonical formatting for the new/rewritten Python,
  Python compilation, and `git diff --check`.
- PASS: the real database proof applies migration 396 as DBA and calls the
  service as direct unprivileged `atlas`. It covers owner/ACL/function
  attestation; runtime trigger-replacement denial; immutable updates; draft
  pointer rejection; duplicate and distinct concurrent publications; stale
  retry conflict; rollback after the version transition but before pointer
  selection; DBA trigger execution; and readiness failure while a trigger is
  disabled.
- Pending for this review follow-up: clean-tree `./ops test local-review`,
  mechanical push, current-head hosted checks, and review reconciliation.
- Hosted full Unit Gate remains GitHub-only under `.agent/capabilities.yaml`.

## Estimated diff size

| File | LOC |
|---|---:|
| `.agent/capabilities.yaml` | 15 |
| `.agent/runbooks/database.md` | 67 |
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 9 |
| `atlas_brain/eom_api/funnel.py` | 144 |
| `atlas_brain/main_eom.py` | 1 |
| `atlas_brain/services/eom_terms_authority.py` | 953 |
| `atlas_brain/storage/migrations/396_eom_terms_authority.sql` | 422 |
| `atlas_brain/storage/migrations/__init__.py` | 1 |
| `ops` | 49 |
| `plans/PR-EOM-Terms-Authority.md` | 276 |
| `scripts/apply_eom_first_clean_completion_schema.py` | 55 |
| `scripts/apply_eom_terms_authority_schema.py` | 42 |
| `tests/test_agent_operations_contract.py` | 71 |
| `tests/test_eom_first_clean_completion_dba_runner.py` | 44 |
| `tests/test_eom_terms_authority.py` | 1053 |
| `tests/test_migrations_runner.py` | 3 |
| **Total** | **3206** |
