# PR-EOM-Operator-Mutation-Contract

## Why this slice exists

Website issue #109 is the Slice 0B child of the EOM CRM write-boundary arc.
0A has landed the contact-write guard, and ATLAS #2303 cleared the call-action
mutation gate that blocked this slice. The next missing contract is an Atlas
owned mutation boundary that the tracker and later CRM UI can call before any
caller is migrated.

Diff-budget override: this slice is over the 400 LOC soft cap because the
Atlas contract is only safe as one vertical seam: the authenticated route,
command normalizer, provider-owned atomic create/update path, exact identity
resolver, lifecycle receipt/idempotency evidence, capability advertisement, and
real DB + ASGI proofs have to land together before any tracker/website caller is
migrated. Splitting the route from the provider would expose an unwired contract;
splitting the provider from the tests would leave the new contact writer
unproven at the tenant/idempotency boundary; splitting the exact EOM resolver
would keep the old substring identity bug in the same slice that claims to
define the canonical mutation boundary.

### Problem-derived contract

- Root cause: operator-authored EOM contact creates/edits do not have a single
  authenticated, idempotent, audited Atlas entry point. The generic provider
  create/update surface is too broad for external callers, and the EOM operator
  resolver needs exact last-10 identity matching rather than inheriting the
  generic CRM API's partial phone-search behavior. Without a bounded domain entry point, the next tracker
  slice would either keep minting tracker-local customers or widen access to
  generic CRM mutation rules.
- Correct fix must touch/change:
  - add one EOM domain service for operator-authored contact create/update
    commands, sibling to inbound lead ingress;
  - reuse the existing `DatabaseCRMProvider` persistence boundary so the repo
    still has only the existing provider-owned `INSERT INTO contacts` sites;
  - add one authenticated/flag-gated funnel route that accepts a service
    bearer, actor headers, and an `Idempotency-Key`;
  - require caller provenance as channel + external ref, but let the domain
    tier choose how `contacts.source/source_ref`, contact metadata, and
    lifecycle metadata store that provenance;
  - normalize email/phone/blank values in one EOM helper path, use exact
    last-10 phone matching inside the EOM operator resolver, and preserve the
    generic CRM API's existing phone substring compatibility for other callers,
    including stored extension numbers;
  - record `contact_created` or `contact_updated` lifecycle evidence with the
    actor and operation key for every fresh create/update through the new path;
  - replay the same idempotency key as HTTP 200 with the same contact, and
    reject the same key with a different normalized payload;
  - advertise the new route through the existing capability manifest so later
    tracker/website callers can gate on deployed support;
  - add HTTP-boundary tests and provider/domain tests for happy path, auth,
    malformed payloads, idempotent replay, payload conflict, exact-match
    resolution, ambiguous identity, and lifecycle actor evidence.
- Must not change:
  - no tracker or website caller migration in this PR;
  - no customer-visible portal UI, onboarding copy, pricing, or schedule
    semantics;
  - no new `contacts.created_by` column and no destructive contacts migration;
  - no lead-to-customer promotion replacement; existing funnel handoff/booking
    transitions remain the lifecycle transition services;
  - no rewrite of existing `contacts.source` values in place;
  - no new contact persistence path outside `DatabaseCRMProvider`;
  - no work on #110, #111, #112 tracker/website halves, #113, or open Atlas
    PRs from other lanes.

## Scope (this PR)

Ownership lane: eom-crm/lead-funnel
Slice phase: vertical slice

1. Add the Atlas-only operator contact mutation contract behind the existing
   EOM funnel service auth boundary.
2. Add the domain/provider behavior and tests proving create, update, replay,
   conflict, exact phone matching, provenance, and lifecycle audit evidence.
3. Add a capability-manifest name for the deployed route.

### Review Contract

- Acceptance criteria:
  - [ ] `POST /eom-funnel/operator-contacts` is unavailable when the funnel API
        is disabled, rejects missing/invalid bearer auth, requires actor
        headers, and validates `Idempotency-Key`.
  - [ ] A fresh create stamps `business_context_id='effingham_maids'`, uses the
        existing provider insert helper, normalizes email/phone/blanks, and
        writes `contact_created` lifecycle evidence with actor + operation key.
  - [ ] A fresh update can edit ordinary identity/contact fields on an EOM
        contact and writes `contact_updated` lifecycle evidence with actor +
        operation key.
  - [ ] Replaying the same idempotency key, same authenticated actor, and same
        normalized payload returns the same contact with `idempotent: true` and
        HTTP 200; replaying the key with a different actor or normalized payload
        returns a conflict.
  - [ ] Phone resolution uses exact normalized last-10 equality, not substring
        matching; ambiguous phone/email resolution fails closed.
  - [ ] Legacy lead rows must already have a supported EOM funnel stage before
        this route can claim or edit them.
  - [ ] Existing lead/customer lifecycle transition ownership is not widened:
        updates through this route do not replace booking, handoff, lost, or
        reopen transitions.
  - [ ] The contact write-boundary inventory still reports only provider-owned
        create sites.
  - [ ] The lead review capability manifest advertises the new route only when
        the route is registered.
- Reachability proof: real ASGI request against the funnel router plus
  provider/domain tests that assert persisted contact rows and lifecycle rows.
- Affected surfaces: `atlas_brain/eom_api/funnel.py`,
  `atlas_brain/services/eom_crm_mutations.py`,
  `atlas_brain/services/eom_lead_ingress.py`,
  `atlas_brain/services/crm_provider.py`, EOM funnel tests, EOM provider
  integration tests, and the contact write-boundary inventory if the provider
  insert helper move changes it.
- Risk areas: auth, tenant isolation, idempotency/replay, ambiguous identity
  matching, lifecycle audit atomicity, source/provenance drift, route response
  compatibility, and contact-write guard drift.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R6, R8, R10, R12, R13, R14.
- Guard/set closure declaration:
  - Supported source channels, contact types, editable contact fields, operator
    lifecycle event families, and advertised capability routes are CLOSED and
    ENUMERATED in this PR's domain boundary (`eom_crm_mutations.py`,
    `funnel.py`, and the lifecycle writer). Out-of-set values are rejected with
    domain errors before persistence or omitted from served capabilities.
  - Operator email grammar and phone identity normalization are OPEN input
    guards with DERIVED choke points: the domain normalizer admits only the
    stated email grammar and SQL-compatible ASCII phone digits, rejects
    malformed/out-of-grammar identities with 422, and treats untrusted inbound
    non-ASCII phone glyphs as no phone identity unless another admitted
    identity exists.
  - Operator provenance is CLOSED to the required channel/ref pair for this
    route. Unknown channels and database-invalid source refs reject with 422;
    admitted refs persist into the contact metadata provenance map and lifecycle
    evidence.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: EOM operator contact mutation admission at the funnel
  route, EOM domain normalization/resolution, and generic provider phone-search
  compatibility.
- Replaced-path behaviors: no caller is replaced yet; this adds the canonical
  path future callers will use, applies exact normalized last-10 matching inside
  the EOM operator resolver, and keeps the generic CRM API's substring lookup
  compatibility for phone fragments and stored extension-number shapes.
- Guard-relevant fields: `contactId`, `contactType`, `fullName`, `email`,
  `phone`, `address`, `city`, `state`, `zip`, `notes`, `sourceChannel`,
  `sourceRef`, actor headers, and `Idempotency-Key`.
- Caller x input shape:
  - New tracker/service operator route caller sends camelCase JSON with one
    operation key plus actor headers; browser users do not call Atlas directly.
  - `atlas_brain/mcp/crm_server.py` generic CRM callers keep existing
    `search_contacts` semantics: email and query lookup are unchanged, short
    phone fragments keep substring lookup, and full/country-code phones gain an
    additional normalized last-10 equality fallback without dropping submitted
    digit substring compatibility.
  - `atlas_brain/mcp/invoicing_server.py`, `atlas_brain/services/customer_context.py`,
    `atlas_brain/tools/scheduling.py`, `scripts/import_eom_customers_live.py`, and
    `atlas_brain/api/email_drafts.py` call the same generic provider search seam;
    their input shapes are preserved by the generic substring-compatible phone
    branch and unchanged email/query branches.
  - Existing EOM inbound lead callers in scheduling, call intelligence, SMS
    intelligence, and lead-pipeline tests now share the ASCII digit normalizer
    and full-number preference rule: a full extracted phone wins, a full
    transport phone backfills a fragment, and fragments stay fragments only
    when no full phone is available.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: the existing `ATLAS_EOM_FUNNEL_API_ENABLED`
  default remains disabled; no new env var is added.
- Explicit value probe: tests override `EOMFunnelConfig(api_enabled=True,
  service_token_sha256=...)` and exercise the new route with a valid bearer.
- Absent value probe: tests exercise disabled config and missing/invalid bearer
  responses.
- Default-session/default-context probe: actor headers are required even after
  service auth; no ambient user/session is inferred.
- Side-effect ordering: provider/domain tests assert contact mutation and
  lifecycle evidence are written in one command, and idempotent replay reads the
  lifecycle receipt before attempting another mutation.

### Files touched

- `atlas_brain/eom_api/funnel.py`
- `atlas_brain/services/crm_provider.py`
- `atlas_brain/services/eom_crm_mutations.py`
- `atlas_brain/services/eom_lead_ingress.py`
- `atlas_brain/storage/migrations/364_eom_operator_contact_operation_key_index.sql`
- `plans/PR-EOM-Operator-Mutation-Contract.md`
- `tests/contact_write_boundary/baseline.json`
- `tests/test_contact_write_boundary.py`
- `tests/test_eom_lead_conversion.py`
- `tests/test_eom_lead_conversion_integration.py`
- `tests/test_migrations_runner.py`
- `tests/test_tenant_stamping.py`

## Mechanism

The route accepts a closed Pydantic request and delegates to the new
`eom_crm_mutations` command service. That service owns request normalization
and calls a provider-owned atomic method. The provider method serializes on the
operation key, checks lifecycle evidence for idempotent replay/conflict, locks
the target or resolved contact row, performs either an ordinary-field update or
a create through the existing provider insert helper, then inserts a
`contact_created` or `contact_updated` lifecycle row with actor/provenance
metadata before the transaction commits.

The admitted execution model is one database transaction per operator command.
Before reading existing receipts or contact identities, the provider takes
transaction-scoped advisory locks in sorted order across the operation key,
explicit contact id when present, source-ref provenance key, and each normalized
phone/email identity key. The phone/email lock namespace is shared with the
atomic EOM inbound writer, so interleavings on the same admitted identity
serialize before either writer can resolve or insert. Inside the transaction,
idempotency is receipt-first: an existing lifecycle row for the same operation
key and matching actor-bound request fingerprint returns the recorded contact,
while the same key with a different actor or payload fingerprint fails before
mutation. With no receipt,
the provider resolves and locks source/provenance/phone/email contact rows with
`FOR UPDATE`, rejects ambiguous or cross-contact identities, writes exactly one
contact create/update, then writes exactly one lifecycle receipt before commit.
The crash boundary is the transaction commit: a rollback leaves no partial
contact/receipt pair, and a committed receipt makes later replays read-only.

Phone normalization is shared with EOM inbound identity handling and strips only
ASCII digits so Python and SQL normalize the same input class. The EOM operator
resolver uses exact normalized last-10 equality. Generic provider phone search
keeps substring lookup compatibility, including full phone inputs that appear
inside a stored phone with extension digits, while still admitting normalized
last-10 matches for ordinary stored numbers.

## Intentional

- No tracker or website caller is migrated here; this is the Atlas-first
  prerequisite that later slices consume.
- No new contact insert literal is added. The generic provider insert block is
  refactored into a helper if atomic reuse is needed, so the 0A guard remains
  meaningful.
- Existing `contacts.source` values are not rewritten. New operator-created
  contacts get a domain-owned source/source_ref mapping, while updates record
  caller provenance in contact metadata and lifecycle metadata instead of
  changing historical origin.
- Existing lead lifecycle transition routes remain authoritative for promotion,
  booking, lost, and reopen states; operator contact updates do not become a
  backdoor transition API.

## Deferred

- Tracker customer creation migration: website #110 / Slice 0C after 0B is
  deployed live.
- Legacy writer convergence/deprecation: website #111 and its D1-D5 children.
- Legacy EOM writer identity-fence convergence, including the existing
  `atlas_brain/autonomous/tasks/email_backfill.py:198` backfill caller that
  still uses generic `find_or_create_contact`; this PR adds the Atlas operator
  mutation contract and does not migrate existing autonomous/backfill callers.
- Normalized contact identity functional indexes for EOM phone/email lookups;
  this is post-contract performance hardening, while this PR proves correctness
  and idempotency for the new operator mutation route.
- Remaining #112 tracker/website capability-gating halves.
- Provenance alerting and missing-provenance observability: website #113.
- Additive source/provenance column split if a later slice needs queryable
  first-class fields; this PR uses existing `contacts.source/source_ref` plus
  contact metadata provenance and lifecycle metadata.

Parking predicate: this slice parks only post-contract operational hardening
that requires a migrated tracker/website caller, live deployment telemetry, or
a later first-class source/provenance schema decision. It does not park known
correctness, CI, security, authorization, data-integrity, idempotency, or
merge-blocking defects in the Atlas operator mutation contract.

Parked hardening: none.

## Verification

- `python -m py_compile atlas_brain/eom_api/funnel.py atlas_brain/services/crm_provider.py atlas_brain/services/eom_lead_ingress.py atlas_brain/services/eom_crm_mutations.py tests/test_eom_lead_conversion.py tests/test_eom_lead_conversion_integration.py tests/test_tenant_stamping.py tests/test_migrations_runner.py tests/test_contact_write_boundary.py`
  -- passed.
- `python -m py_compile atlas_brain/services/crm_provider.py tests/test_tenant_stamping.py tests/test_eom_lead_conversion_integration.py`
  -- passed.
- `python -m pytest tests/test_tenant_stamping.py::test_generic_contact_phone_search_preserves_full_phone_extension_lookup tests/test_tenant_stamping.py::test_generic_contact_phone_search_keeps_partial_phone_substring_lookup -q`
  -- 2 passed.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas@127.0.0.1:5433/atlas python -m pytest tests/test_eom_lead_conversion_integration.py::test_operator_contact_mutation_claims_legacy_contact_by_padded_email 'tests/test_eom_lead_conversion_integration.py::test_operator_contact_mutation_rejects_explicit_id_identity_collision[email-owned@example.com-target@example.com]' -q`
  -- 2 passed.
- `python -m py_compile atlas_brain/services/crm_provider.py tests/test_eom_lead_conversion_integration.py`
  -- passed.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas@127.0.0.1:5433/atlas python -m pytest tests/test_eom_lead_conversion_integration.py::test_operator_contact_mutation_rejects_ambiguous_direct_provenance -q`
  -- 1 passed.
- `python -m pytest tests/test_eom_lead_conversion.py tests/test_tenant_stamping.py tests/test_migrations_runner.py tests/test_contact_write_boundary.py -q`
  -- 293 passed, 1 skipped, 1 warning.
- `python -m py_compile atlas_brain/eom_api/funnel.py atlas_brain/services/eom_crm_mutations.py atlas_brain/services/crm_provider.py tests/test_eom_lead_conversion.py tests/test_eom_lead_conversion_integration.py`
  -- passed.
- `python -m pytest tests/test_eom_lead_conversion.py::test_private_operator_contact_rejects_malformed_identity_before_crm_call tests/test_eom_lead_conversion.py::test_private_operator_contact_rejects_database_invalid_text_before_crm_call -q`
  -- 19 passed.
- `python -m py_compile atlas_brain/services/eom_crm_mutations.py atlas_brain/services/crm_provider.py tests/test_eom_lead_conversion.py tests/test_eom_lead_conversion_integration.py`
  -- passed.
- `python -m pytest tests/test_eom_lead_conversion.py::test_private_operator_contact_rejects_malformed_identity_before_crm_call -q`
  -- 6 passed.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas@127.0.0.1:5433/atlas python -m pytest tests/test_eom_lead_conversion_integration.py::test_operator_contact_mutation_rejects_legacy_lead_without_stage -q`
  -- 1 passed.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas@127.0.0.1:5433/atlas python -m pytest tests/test_eom_lead_conversion_integration.py::test_operator_contact_mutation_rejects_non_object_contact_metadata -q`
  -- 1 passed.
- `python -m py_compile atlas_brain/services/crm_provider.py tests/test_eom_lead_conversion_integration.py`
  -- passed.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas@127.0.0.1:5433/atlas python -m pytest tests/test_eom_lead_conversion_integration.py::test_operator_contact_mutation_crossed_identities_fail_without_deadlock -q`
  -- 1 passed.
- `python -m py_compile atlas_brain/services/eom_crm_mutations.py tests/test_eom_lead_conversion_integration.py`
  -- passed.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas@127.0.0.1:5433/atlas python -m pytest tests/test_eom_lead_conversion_integration.py::test_operator_contact_mutation_creates_replays_and_records_actor -q`
  -- 1 passed.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas@127.0.0.1:5433/atlas python -m pytest tests/test_eom_lead_conversion_integration.py::test_operator_contact_mutation_rejects_explicit_id_identity_collision tests/test_eom_lead_conversion_integration.py::test_operator_contact_mutation_rejects_ambiguous_exact_identity tests/test_eom_lead_conversion_integration.py::test_operator_contact_mutation_rejects_ambiguous_direct_provenance -q`
  -- 4 passed.
- `python -m py_compile atlas_brain/services/crm_provider.py tests/test_tenant_stamping.py`
  -- passed.
- `python -m pytest tests/test_tenant_stamping.py::test_generic_contact_phone_search_preserves_full_phone_extension_lookup tests/test_tenant_stamping.py::test_generic_contact_phone_search_preserves_country_code_to_extension_lookup tests/test_tenant_stamping.py::test_generic_contact_phone_search_keeps_partial_phone_substring_lookup -q`
  -- 3 passed.
- `python -m py_compile atlas_brain/services/crm_provider.py tests/test_eom_lead_conversion_integration.py`
  -- passed.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas@127.0.0.1:5433/atlas python -m pytest tests/test_eom_lead_conversion_integration.py::test_operator_contact_mutation_matches_stored_phone_with_extension tests/test_eom_lead_conversion_integration.py::test_operator_contact_mutation_does_not_match_extension_suffix -q`
  -- 2 passed.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas@127.0.0.1:5433/atlas python -m pytest tests/test_eom_lead_conversion_integration.py::test_operator_contact_mutation_rejects_explicit_id_identity_collision tests/test_eom_lead_conversion_integration.py::test_operator_contact_mutation_rejects_ambiguous_exact_identity tests/test_eom_lead_conversion_integration.py::test_operator_contact_mutation_rejects_ambiguous_direct_provenance tests/test_eom_lead_conversion_integration.py::test_operator_contact_mutation_crossed_identities_fail_without_deadlock tests/test_eom_lead_conversion_integration.py::test_operator_contact_mutation_matches_stored_phone_with_extension tests/test_eom_lead_conversion_integration.py::test_operator_contact_mutation_does_not_match_extension_suffix -q`
  -- 7 passed.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas@127.0.0.1:5433/atlas python -m pytest tests/test_eom_lead_conversion_integration.py::test_operator_contact_mutation_claims_legacy_contact_by_padded_email 'tests/test_eom_lead_conversion_integration.py::test_operator_contact_mutation_rejects_explicit_id_identity_collision[email-owned@example.com-target@example.com]' -q`
  -- 2 passed.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas@127.0.0.1:5433/atlas python -m pytest tests/test_eom_lead_conversion_integration.py::test_operator_contact_mutation_claims_legacy_contact_by_padded_email tests/test_eom_lead_conversion_integration.py::test_operator_contact_mutation_rejects_explicit_id_identity_collision tests/test_eom_lead_conversion_integration.py::test_operator_contact_mutation_rejects_ambiguous_exact_identity tests/test_eom_lead_conversion_integration.py::test_operator_contact_mutation_rejects_ambiguous_direct_provenance tests/test_eom_lead_conversion_integration.py::test_operator_contact_mutation_crossed_identities_fail_without_deadlock tests/test_eom_lead_conversion_integration.py::test_operator_contact_mutation_matches_stored_phone_with_extension tests/test_eom_lead_conversion_integration.py::test_operator_contact_mutation_does_not_match_extension_suffix -q`
  -- 8 passed.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas@127.0.0.1:5433/atlas python -m pytest tests/test_eom_lead_conversion_integration.py -q -k "operator_contact_mutation or inbound_atomic_uses_ascii_phone_normalizer or share_phone_identity_lock"`
  -- 6 passed, 46 deselected.
- `python -m pytest tests/test_crm_read_scoping.py tests/test_leads_intake.py tests/test_eom_lead_conversion.py tests/test_tenant_stamping.py tests/test_migrations_runner.py tests/test_contact_write_boundary.py -q`
  -- 412 passed, 1 skipped, 1 warning.
- `python -m pytest tests/test_contact_write_boundary.py -q`
  -- 65 passed.
- `python scripts/check_contact_write_boundary.py --baseline tests/contact_write_boundary/baseline.json`
  -- passed.
- `python scripts/maturity_sweep.py atlas_brain/storage --tests-root tests --baseline tests/maturity_sweep/baseline_atlas_brain_storage.json --min-score 8 --sensitive-glob '**/billing/**' --sensitive-glob '**/billing*' --sensitive-glob '**/paid*' --sensitive-glob '**/auth/**' --sensitive-glob '**/auth*' --sensitive-glob '**/webhook*' --sensitive-glob '**/webhooks/**' --sensitive-glob '**/*webhook*/**' --sensitive-glob '**/payment*' --sensitive-glob '**/invoicing/**' --sensitive-glob '**/*invoice*' --sensitive-glob '**/*deletion*' --sensitive-glob '**/delete*/**' --sensitive-glob 'atlas_brain/security/**' --sensitive-glob 'atlas_brain/storage/**'`
  -- passed.
- `python scripts/check_guard_class_closure.py` -- passed.
- `python scripts/sync_pr_plan.py plans/PR-EOM-Operator-Mutation-Contract.md --check`
  -- passed.
- `git diff --check` -- passed.
- Pending before push:
  - `bash scripts/local_pr_review.sh` through `scripts/push_pr.sh`.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/eom_api/funnel.py` | 151 |
| `atlas_brain/services/crm_provider.py` | 541 |
| `atlas_brain/services/eom_crm_mutations.py` | 287 |
| `atlas_brain/services/eom_lead_ingress.py` | 18 |
| `atlas_brain/storage/migrations/364_eom_operator_contact_operation_key_index.sql` | 29 |
| `plans/PR-EOM-Operator-Mutation-Contract.md` | 361 |
| `tests/contact_write_boundary/baseline.json` | 1 |
| `tests/test_contact_write_boundary.py` | 6 |
| `tests/test_eom_lead_conversion.py` | 259 |
| `tests/test_eom_lead_conversion_integration.py` | 929 |
| `tests/test_migrations_runner.py` | 45 |
| `tests/test_tenant_stamping.py` | 76 |
| **Total** | **2703** |
