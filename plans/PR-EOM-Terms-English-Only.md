# PR-EOM-Terms-English-Only

## Why this slice exists

The operator set a product boundary on 2026-08-29: customer-facing documents
and data are English-only, while employee-facing documents and data remain
bilingual English/Spanish. The current EOM Terms authority instead requires
both `en` and `es` document bundles and permits Spanish invitations and
receipts. Atlas is the provider and must own this rule before Tracker and
Website consumers remove their Spanish Terms affordances.

### Problem-derived contract

- Root cause: the customer-bound Terms domain models Spanish as both required
  publication data and an admitted delivery locale, so a supposedly canonical
  release cannot be English-only and a reachable customer path can emit and
  store Spanish Terms evidence.
- Correct fix must touch/change: the Terms authority's closed document shape,
  the invitation locale admission boundary, customer-facing renderers, the
  private/public API locale schemas, and focused tests proving English passes
  while Spanish is rejected before schema, database, or email side effects.
- Must not change: residential/commercial audience selection; legal prose;
  acknowledgement, signing, revocation, expiry, readiness, or delivery
  semantics; content hashing and immutable version history; token/security
  behavior; database migrations or constraints; candidate approval; Tracker
  or Website consumers in this provider PR; and every employee-facing
  bilingual document/data surface.

## Scope (this PR)

Ownership lane: eom-onboarding/terms-english-only
Slice phase: Vertical slice

1. Make `en` the only accepted and stored locale for new customer-bound EOM
   Terms versions, invitations, sessions, acceptances, emails, and receipts.
2. Add focused guard and renderer proof, including Spanish rejection before
   any service dependency or outbound-delivery effect.

### Review Contract

- Acceptance criteria:
  - `normalize_eom_terms_documents` accepts exactly residential/commercial,
    each containing exactly one `en` bundle with the three established section
    fields; missing `en` and extra `es` both fail.
  - the HTTP invitation request schema and service admission boundary accept
    `en` and reject `es`.
  - direct customer renderer calls reject `es`, and all accepted output labels,
    subjects, prompts, and receipt fields are English.
  - Spanish invitation rejection occurs before schema readiness, database
    access, or sender invocation.
  - response schemas advertise only `en`; no Spanish Terms branch remains in
    the customer-bound modules.
  - focused Terms authority/acceptance tests and repository format/lint checks
    pass.
- Reachability proof: `POST /api/v1/eom-funnel/terms/invitations` validates the
  locale before `EOMTermsAcceptanceService.issue_and_send`; the service then
  pins `en` documents into invitation and acceptance evidence and its email
  renderers. HTTP and direct-service tests settle both entrypoints.
- Affected surfaces: `atlas_brain/services/eom_terms_authority.py`,
  `atlas_brain/services/eom_terms_acceptance.py`,
  `atlas_brain/eom_api/funnel.py`, and their two focused test modules.
- Risk areas: over-broad removal of employee localization, accepting a mixed or
  extra locale bundle, rejecting valid English, validating only at HTTP while
  direct callers bypass the rule, or performing effects before rejection.
- Reviewer rules triggered: R1 requirements match, R2 test evidence, R5
  backward compatibility, R10 maintainability, and R14 independent
  code-grounded verification. The standing boundary-probe and effect-trace
  rules also apply; the severity sweep is scaled to this closed locale change.

### Boundary-change enumeration

- Boundary path/seam: Terms version `documents` normalization changes from
  audience x (`en`, `es`) x section to audience x `en` x section.
- Replaced-path behaviors: Spanish document publication, invitation issuance,
  renderer output, session projection, and acceptance projection become
  inadmissible; English behavior remains.
- Guard-relevant fields: `documents.<audience>.<locale>` and invitation
  `locale`.
- Caller x input shape: authenticated private draft creation with exact English
  bundles; authenticated invitation POST/direct service call with `locale=en`;
  negative probes with missing `en`, extra `es`, and `locale=es`.

### Deployed-config probing

- Deployed/default config values: no locale environment/config fallback exists;
  locale is explicit request/document data.
- Explicit value probe: `en` succeeds through normalization and renderers;
  `es` fails at HTTP, direct service, and renderer boundaries.
- Absent value probe: missing `en` in either audience fails the closed mapping;
  missing request locale fails the request schema.
- Default-session/default-context probe: N/A; no default locale is introduced.
- Side-effect ordering: direct service rejection is asserted with an unusable
  pool and an uncalled recording sender, proving rejection precedes schema,
  database, and transport work.

### Files touched

- `atlas_brain/eom_api/funnel.py`
- `atlas_brain/services/eom_terms_acceptance.py`
- `atlas_brain/services/eom_terms_authority.py`
- `plans/PR-EOM-Terms-English-Only.md`
- `tests/test_eom_terms_acceptance.py`
- `tests/test_eom_terms_authority.py`

## Mechanism

Use one domain locale constant in the immutable Terms authority. Reuse that
closed set in acceptance admission, retain explicit validation at every
exported customer renderer, remove Spanish rendering branches, and narrow
Pydantic request/response literals. Keep the database's existing superset
checks unchanged for schema compatibility; the application write authority is
the enforced product boundary.

## Intentional

- This is customer Terms localization only. Employee-facing bilingual behavior
  is neither imported nor modified.
- Both residential and commercial English documents remain required; locale
  policy does not collapse audience-specific business terms.
- No translation or legal prose is generated by code.
- Existing database constraint names and migration history stay intact. Atlas
  admits no new Spanish records through supported code paths, while avoiding a
  schema migration for an empty, unpublished customer Terms corpus.

## Deferred

- Tracker proxy schemas and Website customer Terms controls must narrow to
  English in their own sequential consumer PRs after this provider contract.
- Publication of the approved English residential/commercial v1 bundle and a
  live acceptance exercise remain rollout work after all consumers converge.

Parked hardening: none.

## Verification

- PASS: `./ops test focused tests/test_eom_terms_authority.py
  tests/test_eom_terms_acceptance.py` completed with 65 passed and 3 skipped;
  the skipped cases require the explicitly configured disposable PostgreSQL
  integration database.
- PASS: boundary probes cover exact English bundles, missing English, extra
  Spanish, HTTP `en`/`es`/missing locale, direct-service Spanish rejection
  before dependencies, and direct-renderer Spanish rejection.
- PASS: `git diff --check`, `scripts/audit_plan_doc.py`, and
  `scripts/audit_plan_code_consistency.py`.
- PASS cold diff audit: every product/test change traces to the contract; every
  required provider boundary is present; no database migration, consumer,
  employee-facing, legal-document, token, or delivery-state module changed.
- Pending before push: committed-diff local PR review and GitHub-owned unit
  gate.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/eom_api/funnel.py` | 10 |
| `atlas_brain/services/eom_terms_acceptance.py` | 94 |
| `atlas_brain/services/eom_terms_authority.py` | 2 |
| `plans/PR-EOM-Terms-English-Only.md` | 161 |
| `tests/test_eom_terms_acceptance.py` | 90 |
| `tests/test_eom_terms_authority.py` | 7 |
| **Total** | **364** |
