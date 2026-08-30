# PR-EOM-Card-Service-Commitment

## Why this slice exists

Issue #2156 and the operator's current decision require residential customers
to accept Terms after the first clean while sending only recurring customers
to card setup. Atlas currently treats every active residential post-clean
candidate as card-required. A candidate proves that the first clean happened,
but it records no business decision about whether service is recurring or
one-time. The card-session guard and readiness projection therefore cannot
distinguish those cases.

This fixes the root rather than filtering one-time customers in a downstream
UI: Atlas gains one authoritative operator decision that every consumer and the
database enrollment boundary use. The diff is expected to exceed the 400-line
soft cap because the immutable schema, application and database guards,
authenticated route, controlled DBA operation, exact readiness projection, and
negative PostgreSQL boundary proof are one payment-safety authority. Splitting
them would merge either writable state without enforcement or enforcement with
no reachable authoring path.

### Problem-derived contract

- Root cause: residential customer type is being used as a proxy for an absent
  service-commitment fact. Calendar recurrence, Site frequency, and Tracker
  service kind are operational scheduling data and cannot authoritatively say
  whether the customer agreed to recurring service.
- Correct fix must touch/change: persist exactly one immutable `recurring` or
  `one_time` decision for an eligible pending post-clean candidate; bind it to
  the contact, stable operation key, deciding actor, and server time; make exact
  retries idempotent and conflicting replays fail closed; expose the decision
  through an authenticated capability-gated route and candidate projection;
  require `recurring` before card enrollment at both application and database
  boundaries; return one-time readiness as not required; wire the controlled
  migration; and prove both sides of each guard through the real route and a
  disposable PostgreSQL schema.
- Must not change: first-clean evidence or Customer/Site linkage; Site
  frequency, native schedules, Calendar visits, payroll, attendance, location,
  or routing; Terms text/version/invitation/acceptance; Stripe setup mode,
  provider identifiers, webhook confirmation, retry fencing, or return-URL
  rules; charges, prices, invoices, ACH/check handling, refunds, or other money
  movement; raw payment-method storage/logging/display; commercial card policy;
  and unrelated Atlas, Tracker, Website, CRM, or deployment behavior.

## Scope (this PR)

Ownership lane: eom-onboarding-card-vault
Slice phase: Vertical slice
Max files: 22

1. Add Atlas's immutable residential service-commitment authority and expose it
   through the existing authenticated EOM funnel.
2. Make the existing card-vault admission/readiness authority consume that
   decision and enforce the same rule in PostgreSQL before enrollment.
3. Add the controlled migration path, deployment record, CI enrollment, and
   focused real-entrypoint/real-PostgreSQL proof needed to deploy the authority
   before downstream consumers.

### Review Contract

- Acceptance criteria:
  - `POST /post-clean-onboarding-candidates/{candidate_id}/service-commitment`
    records one decision for an active EOM residential customer with a pending
    candidate, returns 201 for the first write and 200 for an exact retry, and
    rejects missing, malformed, reused-with-drift, inactive, commercial, or
    already-classified inputs; settled by focused route/service tests.
  - Candidate listing returns only the bounded decision, deciding name, and
    timestamp aliases in addition to its existing projection; settled by the
    first-clean candidate projection test.
  - Card session admission reaches no Stripe provider method for one-time,
    unclassified, inactive, or non-residential subjects; recurring admission
    keeps the existing hosted-session contract; settled by the card-vault
    negative/provider-call tests.
  - Readiness returns `cardRequired=false`, `cardReady=true`, and
    `reason=not_required` for commercial and residential one-time contacts;
    unclassified residential contacts remain fail-closed with
    `service_commitment_required`; settled by the readiness matrix tests.
  - PostgreSQL admits an enrollment only after a matching recurring decision,
    rejects one-time and absent decisions, and prevents update/delete/truncate
    of decision evidence; settled by the disposable-PostgreSQL migration test.
  - The runtime attestation rejects drift in decision columns, constraints,
    trigger bodies/types/enabled state, ownership, and runtime privileges;
    settled by positive and negative schema-attestation probes.
  - The controlled DBA runner, `./ops db controlled` dispatcher, capability
    record, runbook, migration registry, and EOM CI path lists all include
    migration 399; settled by the operations, runner, migration, and workflow
    tests.
- Reachability proof: an authenticated request traverses the real EOM FastAPI
  router and service to a persisted decision, then card readiness observes its
  effect. A separate integration test executes migrations 395/398/399 in a
  disposable schema with the real `atlas` runtime role and observes both the
  admitted recurring enrollment and rejected one-time/absent enrollments.
- Affected surfaces: EOM funnel route/capability projection, post-clean
  candidate projection, card-vault eligibility/readiness, one controlled
  migration and its operator path, EOM-focused CI selection, and focused tests.
- Risk areas: authorization, immutable audit evidence, idempotency,
  check-then-act races, database privilege drift, payment-provider effects,
  stale deployment schema, and accidental one-time/commercial card prompts.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R6, R8, R10, R12, R13, R14.

### Boundary-change enumeration

| Boundary | Replaced behavior | Guard fields | Callers and input shapes | Disposition |
|---|---|---|---|---|
| Service-commitment decision | No authority or authoring path existed. | candidate id/status/context, contact id/type/customer type/status/context, closed decision enum, operation key, actor id/name | Authenticated Tracker relay: recurring, one-time, malformed/missing, exact retry, drifted retry, absent/ineligible subject | Add one fail-closed route; preserve no-write behavior for every rejected shape. |
| Card-session admission | Every otherwise eligible residential post-clean candidate could proceed. | existing public token/Terms/candidate/contact fields plus immutable decision | Public onboarding: recurring, one-time, unclassified, inactive/non-residential, replay/concurrent request | Intentionally change recurring to admit and one-time/unclassified to reject before provider I/O; preserve all existing Terms/provider guards. |
| Card readiness | Residential post-clean candidates were always card-required. | contact type/customer type/status, candidate status, decision, enrollment status | Public onboarding refresh: commercial, recurring ready/pending, one-time, unclassified, no candidate | Intentionally add one-time `not_required` and unclassified `service_commitment_required`; preserve commercial and provider-confirmed readiness. |
| Schema/runtime admission | Migration 398 alone was sufficient for card-vault readiness. | migration records, exact relation/column/constraint/trigger/ACL definitions | Atlas runtime on explicit deployed schema, absent migration 399, drifted schema, default disabled card config | Require exact migration 399 attestation before any card provider effect; absent/default remains fail-closed. |

The decision and lifecycle inputs are closed database enums/states rather than
open free text. Operation and actor strings are validated at one choke point and
are audit evidence, not classifiers. Guard-class closure is therefore bounded by
the closed decision vocabulary and exact eligibility predicate; it is not an
enumerative open-input recognizer.

### Deployed-config probing

- No new environment value or fallback is introduced. The existing
  `ATLAS_EOM_FUNNEL_CARD_VAULT_ENABLED` default remains `false`.
- Explicit enabled configuration still must satisfy the existing public
  onboarding and dedicated Stripe credential validators, then also pass exact
  migration-399 schema attestation before provider I/O.
- Absent/default configuration continues to disable new session issuance.
- Current production migration-399 state is could-not-determine until the
  controlled preflight runs against the protected deployment target; rollout
  stops there rather than weakening the guard.


### Files touched

- `.agent/capabilities.yaml`
- `.agent/runbooks/database.md`
- `.github/workflows/atlas_eom_lead_pipeline_checks.yml`
- `atlas_brain/eom_api/card_vault.py`
- `atlas_brain/eom_api/funnel.py`
- `atlas_brain/services/eom_card_service_commitment.py`
- `atlas_brain/services/eom_card_vault.py`
- `atlas_brain/services/eom_first_clean_completion.py`
- `atlas_brain/storage/migrations/399_eom_card_service_commitments.sql`
- `atlas_brain/storage/migrations/__init__.py`
- `ops`
- `plans/PR-EOM-Card-Service-Commitment.md`
- `scripts/apply_eom_card_service_commitment_schema.py`
- `scripts/apply_eom_first_clean_completion_schema.py`
- `tests/test_agent_operations_contract.py`
- `tests/test_eom_card_service_commitment.py`
- `tests/test_eom_card_vault.py`
- `tests/test_eom_first_clean_completion.py`
- `tests/test_eom_first_clean_completion_dba_runner.py`
- `tests/test_eom_terms_acceptance.py`
- `tests/test_migrations_runner.py`

## Mechanism

Migration 399 creates a guard-owned, append-only decision relation with unique
candidate, contact, and operation-key constraints. Its insert trigger rechecks
the canonical candidate/contact relationship and eligibility, while a second
trigger rejects any new card enrollment without the matching immutable
`recurring` row. Runtime code locks the candidate and contact in one PostgreSQL
transaction, compares all operation-bound facts for idempotent replay, and
inserts only through column-scoped runtime grants.

The authenticated funnel route exposes that service and candidate listing
projects only the decision audit fields. Card-vault eligibility joins the same
authority before reserving provider state. Readiness maps `one_time` to
`not_required`, maps an absent residential decision to
`service_commitment_required`, and otherwise preserves provider-confirmed
pending/ready behavior. Runtime schema attestation and the controlled DBA
preflight require the exact new boundary before issuance can start.

Candidate listing remains available during a schema-first rolling deployment:
before migration 399 exists it returns the three new fields as null; after the
relation exists it projects decisions only when the exact schema attestation
passes, and rejects a present-but-drifted relation. The migration-399 operator
path separately attests migration 398 before applying and re-attests both sides
of the composed card boundary afterward.

## Intentional

- There is no inferred/default decision. Office staff must choose one-time or
  recurring explicitly; schedule labels and Calendar recurrence remain outside
  the authority.
- Migration 399 refuses to run when enrollment evidence already exists. That
  forces explicit reconciliation rather than inventing historical commitment
  decisions from card state.
- A pre-399 candidate queue remains readable with null commitment fields. This
  preserves the first-clean queue while migration 399 is staged; it does not
  permit card enrollment or provider I/O, both of which require the exact new
  attestation.
- One-time residential customers still use the existing Terms flow. This slice
  changes only whether the later card step is required.
- The Stripe webhook and already-open session completion contract stay active;
  classification gates new enrollment, not provider-confirmed completion of an
  enrollment that already exists.

## Deferred

Parking predicate: UI polish, inferred recurrence, broader schedule cleanup,
and provider/retry changes that do not invalidate the immutable decision or
permit an unsafe card effect are parked outside this authority PR.

- Tracker proxy/projection support follows after this Atlas provider PR is
  deployed and its capability is observable.
- Website manager choice and customer card-setup UI follow after Tracker can
  relay the closed Atlas contract.
- Production apply/restart is blocked until the protected DBA configuration is
  available and the preflight proves the target; it is an operational rollout,
  not a reason to weaken or fake the schema gate.
- Parked hardening: none.

## Verification

Completed before plan sync:

- Focused service, card-vault, controlled-runner, migration-runner, and operator
  contract tests: `356 passed, 1 skipped`.
- Full first-clean plus Terms/card-boundary integration files against disposable
  PostgreSQL runtime and DBA roles: `131 passed`.
- Ruff lint: `All checks passed!`; Python compile completed without error; and
  `git diff --check` completed without error.
- Guard-class closure, boundary-change enumeration, and deployed-config probing:
  each reported `OK` against `origin/main`.
- The disposable PostgreSQL container was stopped after the integration proof.
- Plan synchronization, document shape, and plan/code consistency all pass
  against `origin/main`.

### Review correction verification

The cold correction audit found no contract gap, untraced change, or forbidden
dependency:

- Card-vault schema attestation is again the stable prerequisite for consuming
  an already-open provider result, while the additive commitment attestation is
  required only for new issuance and readiness (`eom_card_vault.py:818`, `:822`,
  `:1420`, `:1559`, `:1758`). This preserves the declared existing-session
  completion behavior without reopening an unclassified issuance path.
- Migration 399 now acquires a `SHARE ROW EXCLUSIVE` table lock before deciding
  that legacy enrollments are absent, and holds it through trigger creation
  (`399_eom_card_service_commitments.sql:81-88`). The PostgreSQL concurrency
  proof observes the waiting lock, commits the competing legacy insert, and
  proves the migration then refuses adoption without creating the decision
  relation (`test_eom_terms_acceptance.py:2887-2968`).
- The candidate projection proof now creates a real first-clean candidate,
  applies migrations 396 through 399, records the decision through the private
  route, reads it back through the candidate route, and compares serialized
  fields to the persisted row (`test_eom_first_clean_completion.py:287-314`,
  `:3037-3098`). The previous SQL-text fake no longer supplies this proof.
- The card-vault regression proves commitment-schema drift blocks a second
  issuance but does not strand confirmation of the already-open session
  (`test_eom_card_vault.py:1274-1289`).
- The accepted-candidate setup extraction is test-only reuse required by the
  migration race; it preserves the same issue, acceptance, and candidate facts
  used by the existing database-guard test
  (`test_eom_terms_acceptance.py:459-505`, `:3007-3009`).
- No first-clean writer, Terms content/acceptance behavior, Stripe provider
  adapter, webhook result semantics, money movement, commercial policy,
  schedule/calendar/payroll behavior, Tracker, or Website file changed.

Correction evidence: the full card-vault test file reports `74 passed`; the
three focused real-PostgreSQL guard/projection cases report `3 passed`; Ruff
reports `All checks passed!`; and `git diff --check` completes without output.

### Cold diff reconstruction

No contract gap, untraced change, or forbidden dependency remains in the cold
read of the diff:

- Migration 399 creates the single closed decision relation, makes its evidence
  append-only, rejects ambiguous pre-existing enrollment, and places a recurring
  decision guard directly on enrollment (`399_eom_card_service_commitments.sql:8`,
  `:81`, `:88`, `:125`, `:161`, `:181`).
- The service validates the closed request, attests the exact runtime schema,
  serializes replay, rechecks candidate/contact eligibility under row locks, and
  inserts one immutable receipt (`eom_card_service_commitment.py:48`, `:110`,
  `:471`).
- The private funnel exposes only the bounded request/receipt, authenticates the
  route, requires actor plus idempotency evidence, and advertises that exact
  registered method/path (`funnel.py:219`, `:957`, `:1210`, `:2040`).
- Candidate reads remain available before migration 399 with null aliases, then
  require the exact attestation before joining decision evidence
  (`eom_first_clean_completion.py:1090`).
- Card issuance now requires both schema boundaries and an explicit recurring
  decision before provider work; readiness preserves commercial behavior,
  marks one-time as not required, and fails unclassified residential customers
  closed (`eom_card_vault.py:818`, `:830`, `:882`, `:1749`).
- The shared controlled runner proves migration 398's recorded and actual schema
  before 399, re-attests after apply, and checks migration-399 bookkeeping
  against the runtime view (`apply_eom_first_clean_completion_schema.py:48`,
  `:412`, `:490`, `:527`). The wrapper, migration registry, `./ops` dispatcher,
  capability inventory, runbook, CI paths, and focused tests are the supporting
  deployment/test surfaces declared in Scope.
- The changed-file inventory below contains only those authority, API,
  deployment, documentation, and focused-test surfaces. It does not touch the
  first-clean writer, Terms semantics/content, Stripe adapter/provider contract,
  money movement, commercial policy, schedules, payroll, Tracker, or Website.

Still required before the correction push: plan synchronization and the
repository's single local pre-push review path. The full unit gate stays
GitHub-only under repository policy.

## Estimated diff size

| File | LOC |
|---|---:|
| `.agent/capabilities.yaml` | 11 |
| `.agent/runbooks/database.md` | 35 |
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 9 |
| `atlas_brain/eom_api/card_vault.py` | 1 |
| `atlas_brain/eom_api/funnel.py` | 96 |
| `atlas_brain/services/eom_card_service_commitment.py` | 620 |
| `atlas_brain/services/eom_card_vault.py` | 50 |
| `atlas_brain/services/eom_first_clean_completion.py` | 42 |
| `atlas_brain/storage/migrations/399_eom_card_service_commitments.sql` | 244 |
| `atlas_brain/storage/migrations/__init__.py` | 1 |
| `ops` | 5 |
| `plans/PR-EOM-Card-Service-Commitment.md` | 326 |
| `scripts/apply_eom_card_service_commitment_schema.py` | 40 |
| `scripts/apply_eom_first_clean_completion_schema.py` | 47 |
| `tests/test_agent_operations_contract.py` | 30 |
| `tests/test_eom_card_service_commitment.py` | 394 |
| `tests/test_eom_card_vault.py` | 77 |
| `tests/test_eom_first_clean_completion.py` | 122 |
| `tests/test_eom_first_clean_completion_dba_runner.py` | 234 |
| `tests/test_eom_terms_acceptance.py` | 407 |
| `tests/test_migrations_runner.py` | 3 |
| **Total** | **2794** |
