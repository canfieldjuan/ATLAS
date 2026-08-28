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
  Without that authority, later invitations and acceptances could only point to
  mutable prose or an ambiguous "current terms" claim.
- Correct fix must touch/change: add a guarded Atlas migration for immutable
  published versions plus the singleton current-version pointer; add one
  service that validates the closed four-document bundle, hashes it
  canonically, creates/replays drafts, publishes/replays versions under a
  serialized transaction, and reads the current version; expose those actions
  through authenticated, actor-attributed private funnel routes; add focused
  model/service/route/migration tests.
- Must not change: no welcome draft, existing public-onboarding token, email,
  Customer/Site handoff, first-clean completion/candidate, payment, Stripe,
  calendar, Tracker, Website, employee timekeeping, commercial billing, or
  deployed Terms content behavior changes. No Terms row is seeded and no
  customer-facing content is published by deployment.

## Scope (this PR)

Ownership lane: eom-onboarding-terms
Slice phase: Vertical slice

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
    returns the same version without another state transition.
  - migration 396 prevents UPDATE/DELETE of a published version and prevents
    the current pointer from referencing a draft or being removed.
  - route tests exercise `POST /api/v1/eom-funnel/terms/versions`,
    `POST /api/v1/eom-funnel/terms/versions/{id}/publish`, and
    `GET /api/v1/eom-funnel/terms/current` through the real router with service
    authentication and actor attribution.
  - the diff contains no seed INSERT and touches no existing onboarding,
    completion, money, calendar, or delivery implementation.
- Reachability proof: the mounted EOM FastAPI router is called by an ASGI client;
  the observable effects are a closed draft response, a published/current
  response, and stable API errors for rejected boundaries.
- Affected surfaces: migration chain, new Terms authority service, private EOM
  funnel API models/dependency/routes, focused tests.
- Risk areas: mutable published content, ambiguous document bundles, hash
  instability, duplicate labels, concurrent publication, missing schema,
  authorization/actor omission, accidental customer-visible publication.
- Reviewer rules triggered: R1, R2, R3, R4, R6, R7, R9, R10, R11, R14.

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

- `atlas_brain/eom_api/funnel.py`
- `atlas_brain/services/eom_terms_authority.py`
- `atlas_brain/storage/migrations/396_eom_terms_authority.sql`
- `plans/PR-EOM-Terms-Authority.md`
- `tests/test_eom_terms_authority.py`

## Mechanism

Migration 396 stores the complete closed document bundle as JSONB beside a
canonical SHA-256 hash and actor evidence. Draft rows may transition once to
published; a trigger then makes them append-only. A singleton pointer selects
the current published version without rewriting history. The service owns all
normalization, replay, conflict, and transaction behavior. The funnel routes
only validate/reproject the private HTTP contract and map typed service errors.

## Intentional

- The source PDFs are evidence for later copy authoring, not seed data. Exact
  refined English and Spanish copy still requires operator approval.
- No IP address, device fingerprint, invitation token, signature, acceptance,
  readiness, email, or card state is added in this authority-only slice.
- Every published version is retained. Publishing a new version moves only the
  current pointer; it does not mutate the prior published row.
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

- PASS: `./ops test focused tests/test_eom_terms_authority.py -q` (27 passed).
- PASS: `./ops test focused tests/test_migrations_runner.py -q` (117 passed,
  1 environment-gated test skipped).
- PASS: adjacent `tests/test_eom_public_onboarding.py` (41 passed) and
  `tests/test_eom_first_clean_completion.py` (14 passed, 74 disposable-DB tests
  skipped by focused-mode credential isolation).
- PASS: targeted Ruff lint/format and Python compilation for changed Python.
- PASS: migration 396 and the authority service were exercised against an
  isolated PostgreSQL 16 container. Draft/replay/publish/current succeeded;
  published edits, publish-time content rewrites, draft pointer selection, and
  current-pointer deletion/truncation failed; disabling either a row or
  statement integrity trigger made schema readiness fail. The throwaway
  container was then removed.
- Pending before push: plan sync, diff check, and `scripts/push_pr.sh` mechanical
  local review bundle.
- Hosted full Unit Gate remains GitHub-only under `.agent/capabilities.yaml`.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/eom_api/funnel.py` | 144 |
| `atlas_brain/services/eom_terms_authority.py` | 467 |
| `atlas_brain/storage/migrations/396_eom_terms_authority.sql` | 145 |
| `plans/PR-EOM-Terms-Authority.md` | 190 |
| `tests/test_eom_terms_authority.py` | 541 |
| **Total** | **1487** |
