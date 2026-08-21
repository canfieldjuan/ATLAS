# PR-EOM-Contact-Directory

## Why this slice exists

Website #240 (child of website #105, Slice 3): an Atlas `customer` contact
created or matched through the portal's Add Contact form is unreachable from
every portal surface after refresh. The tracker proxy deliberately creates no
local mirror (eom-timetracker PR #196), the portal Customers tab reads tracker
operational customers only, and Atlas's sole funnel read
(`/eom-funnel/leads`) admits only `contact_type='lead'` in stages
new/estimate_booked/won. A created customer (contact_type='customer',
lead_stage NULL) can never satisfy it, so the record is write-only from the
portal's point of view.

### Problem-derived contract

- Root cause: Slice 2 shipped a canonical write for both contact kinds
  (`POST /eom-funnel/operator-contacts` accepts contactType lead|customer)
  without a canonical read for either. The only portal-facing read,
  `DatabaseCRMProvider.list_eom_new_lead_review_items`, filters
  `contact_type='lead' AND lead_stage IN ('new','estimate_booked','won')`,
  which is deliberately narrower than the write surface. There is no
  discovery read over the canonical EOM contact set.
- Correct fix must touch/change: `atlas_brain/eom_api/funnel.py` (a new
  authenticated, closed-projection `GET /eom-funnel/contact-directory` route
  plus its `contact.directory` entry in `_CAPABILITY_ROUTES` so callers derive
  deployment proof from the registered method/path);
  `atlas_brain/services/crm_provider.py` (a new read-only, tenant-scoped
  `list_eom_contact_directory` provider method with keyset pagination on
  immutable columns and bounded search); `tests/test_eom_contact_directory.py`
  (route closure, capability, and real-Postgres scope proofs); the
  `atlas_eom_lead_pipeline_checks` workflow enrollment for that test file.
- Must not change: the `/eom-funnel/leads` contract (byte-identical envelope
  and query); the operator mutation boundary and its idempotency/lifecycle
  semantics; `known-contacts` link verification; onboarding draft/public-link
  routes; any write path (this PR performs no mutation anywhere); tracker and
  website repos (separate PRs in the slice); database schema (no migration --
  the read uses existing columns).

### Diff-budget justification

The 400-LOC cap is exceeded by the mandatory plan doc plus the test matrix:
the runtime change is ~280 LOC across two files, and the tests
(route-closure, capability, negative-control, and six real-Postgres proofs)
are the majority of the diff. Splitting tests from the route would ship an
unproven admission boundary; splitting route from provider would ship a dead
route. The slice is indivisible at PR granularity.

## Scope (this PR)

Ownership lane: eom-crm/contact-directory
Slice phase: Vertical slice

1. Add `GET /eom-funnel/contact-directory`: authenticated (existing funnel
   bearer + actor), tenant-scoped to `effingham_maids`, admitting
   `status='active' AND contact_type IN ('lead','customer')`, with a closed
   camelCase projection, closed `kind` filter (all|lead|customer), bounded
   `search` (name/email/phone ILIKE with escaped metacharacters plus a
   digits-run phone fallback), keyset pagination on `(created_at, id)`
   descending, and rejection of unknown query-parameter names.
2. Advertise it as `contact.directory` -> `("GET",
   "/eom-funnel/contact-directory")` through the existing route-derived
   capability manifest, so the tracker can gate its proxy on the registered
   method/path instead of a copied string.

Max files: 6

### Review Contract

- Acceptance criteria:
  - Authentication is required and the actor boundary holds -- settled by
    tests/test_eom_contact_directory.py::test_the_route_refuses_an_unauthenticated_caller,
    ::test_a_wrong_service_token_is_refused, and
    ::test_a_missing_actor_header_is_refused.
  - Tenant scoping cannot leak another business context, archived rows are
    excluded, and kinds outside lead/customer never appear -- settled against
    real Postgres by
    ::test_tenant_scope_and_lifecycle_hold_against_real_postgres (negative
    control: removing the business_context_id condition, or the
    status='active' condition, makes this test fail; both were exercised
    locally and restored).
  - Both active leads and customers are returned and a lost lead stays
    findable with its stage -- settled by
    ::test_both_contact_kinds_come_back_and_the_projection_is_closed and
    ::test_a_lost_lead_is_rendered_with_its_stage.
  - The kind filter is closed and forwarded verbatim -- settled by
    ::test_the_kind_filter_is_closed (negative control: loosening the route
    Literal to str makes it fail) and
    ::test_each_admitted_kind_is_forwarded_verbatim.
  - Unknown query-parameter names are rejected 422 -- settled by
    ::test_an_unknown_query_parameter_is_rejected_not_ignored (negative
    control: removing the rejection call makes it fail).
  - Search matches name, email, and phone, matches a formatted stored phone
    from a digits-only query, and treats LIKE metacharacters literally --
    settled against real Postgres by
    ::test_search_matches_name_email_and_phone_against_real_postgres and
    ::test_a_like_metacharacter_searches_literally_against_real_postgres
    (negative control: removing `_escape_eom_directory_like_pattern` makes the
    latter fail against the seeded 'Percent 5% Off' / 'Percent 55 Co' pair).
  - Pagination is deterministic keyset on immutable `(created_at, id)` and a
    full traversal neither drops nor duplicates rows -- settled against real
    Postgres by ::test_keyset_traversal_neither_drops_nor_duplicates, with
    cursor round-trip proven by
    ::test_pagination_reports_has_more_and_a_cursor_that_round_trips.
  - Malformed or short cursors and blank or overlong search values fail
    closed 422 -- settled by ::test_a_malformed_cursor_is_rejected,
    ::test_a_short_cursor_is_rejected, ::test_a_blank_search_is_rejected, and
    ::test_an_overlong_search_is_rejected.
  - Directory reads perform no writes and touch no other provider method --
    settled by ::test_the_directory_read_touches_no_other_provider_method
    (the spy provider raises on any attribute access outside the directory
    read) plus a before/after live-DB snapshot recorded in the PR body.
  - The projection is closed on both envelope and item, and a row outside the
    admitted kinds/status/customer-type sets can never be emitted -- settled
    by ::test_both_contact_kinds_come_back_and_the_projection_is_closed,
    ::test_a_row_outside_the_directory_kinds_can_never_be_emitted,
    ::test_an_archived_row_can_never_be_emitted, and
    ::test_a_junk_customer_type_can_never_be_emitted.
  - Capability advertisement matches the actually registered method/path and
    appears on the pipeline read -- settled by
    ::test_the_directory_is_advertised_in_the_capability_manifest and
    ::test_the_lead_review_response_advertises_the_directory, with the
    existing manifest suite (tests/test_eom_funnel_capability_manifest.py)
    still green.
  - The deployed aggregate serves the route under /api/v1 -- settled by
    ::test_the_real_aggregate_serves_the_route_at_its_deployed_path.
- Reachability proof: the tracker's `_atlas_funnel_read` calls
  `{ATLAS_FUNNEL_BASE_URL}/eom-funnel/contact-directory` once the follow-up
  tracker PR allow-lists the path; until then the route is reachable at
  `/api/v1/eom-funnel/contact-directory` on the deployed aggregate
  (atlas_brain/api/__init__.py:111 mounts the funnel router), proven by
  tests/test_eom_contact_directory.py::test_the_real_aggregate_serves_the_route_at_its_deployed_path.
- Affected surfaces: atlas_brain/eom_api/funnel.py (new route + capability
  entry; existing routes untouched), atlas_brain/services/crm_provider.py
  (new read-only method + three module-level search helpers; existing methods
  untouched), the funnel capability manifest consumed by the tracker, and the
  atlas_eom_lead_pipeline_checks workflow (test enrollment only).
- Risk areas: tenant scoping and archived-row leakage on the new SQL; LIKE
  metacharacter injection into the search pattern; keyset drop/duplicate under
  traversal; capability over-advertising; accidental writes from a read path;
  regression of the untouched `/eom-funnel/leads` contract.
- Reviewer rules triggered: R1, R2, R5, R14

### Boundary-change enumeration

This diff adds an admission boundary (the directory's filter set); no
existing boundary is modified.

- Boundary path/seam: `GET /eom-funnel/contact-directory` admission -- query
  parameters {limit, cursor, search, kind} plus the SQL admission predicate
  (`business_context_id='effingham_maids' AND status='active' AND
  contact_type IN ('lead','customer')`).
- Replaced-path behaviors: none replaced -- `/eom-funnel/leads` keeps its
  exact filter (`lead` in stages new/estimate_booked/won) and envelope;
  `known-contacts` keeps id-only verification.
- Guard-relevant fields: `kind` (closed Literal all|lead|customer), `search`
  (1..120 chars, non-blank, LIKE-escaped, digit fallback only for
  phone-shaped input with >=4 digits), `cursor` (16..512 chars, urlsafe-b64
  `created_at|id`, tz-aware, 422 on any malformation), `limit` (1..200),
  unknown parameter names (422), and on the response side `status`
  (Literal['active']), `contact_type` / `customer_type` (validated against
  the operator boundary's own EOM_OPERATOR_CONTACT_TYPES /
  EOM_CUSTOMER_TYPES sets).
- Caller x input shape: single caller is the tracker's server-side
  `_atlas_funnel_read` (browser never calls Atlas directly). Probed shapes:
  no-param default page; each kind value; unknown kind; unknown parameter
  name; blank/overlong/metacharacter search; digits-only phone search;
  malformed/short cursor; cursor round-trip across pages; a provider row
  outside the admitted kind/status/customer-type sets (can never be emitted).

### Closure declarations (GUARD_CLASS_CLOSURE)

- Directory kind filter `{all, lead, customer}`: CLOSED -- 'all' plus the
  operator boundary's own EOM_OPERATOR_CONTACT_TYPES; sourced DERIVED for the
  SQL admission (`_EOM_DIRECTORY_CONTACT_KINDS` mirrors the canonical pair and
  the response validator reads EOM_OPERATOR_CONTACT_TYPES directly).
  Out-of-set input -> 422 at the route Literal (safe side: refuse, never
  widen a filter silently).
- Phone punctuation family `" ()+.-"` + 4-digit minimum: CLOSED, authored
  here as the search contract (no upstream source exists). Out-of-set
  character -> the digit fallback does not run and the query is matched as
  text only (cheap side: a missed phone match is a smaller failure than a
  false phone match on incidental digits). The class-closure proof is the
  generated tokens x containers x families test with a spec-derived oracle
  whose contract literals are pinned independently of the implementation
  (tests/test_eom_contact_directory.py::test_search_admission_grammar_holds_across_tokens_containers_and_families).
- LIKE metacharacter set `{%, _, \\}`: CLOSED, DERIVED from the SQL LIKE
  specification (the complete metacharacter vocabulary for LIKE/ESCAPE).
  Out-of-set characters pass through literally; the invariant proof over
  generated inputs is
  ::test_escaped_patterns_never_leave_an_active_like_metacharacter.
- Projection field set: CLOSED, authored here as the browser contract.
  An out-of-set provider key -> Pydantic extra='forbid' rejects the row
  (safe side: 500 loudly rather than leak an unreviewed field).
- Admitted lifecycle status `{'active'}`: CLOSED, authored (this slice's
  admission decision). Out-of-set rows are excluded by the SQL predicate and,
  independently, can never serialize (status Literal['active']) -- incomplete
  enforcement fails to the safe side twice.

### Deployed-config probing

N/A - no guard/config/env-fallback change: the route uses the existing
`EOMFunnelConfig` bearer gate unchanged and introduces no new configuration.

### Files touched

- `.github/workflows/atlas_eom_lead_pipeline_checks.yml`
- `atlas_brain/eom_api/funnel.py`
- `atlas_brain/services/crm_provider.py`
- `plans/PR-EOM-Contact-Directory.md`
- `tests/test_eom_contact_directory.py`

## Mechanism

`list_eom_contact_directory` (DatabaseCRMProvider) builds one SELECT over
`contacts` with a fixed tenant/status predicate, a kind predicate (`= ANY`
over the closed pair, or a single value), an optional search predicate
(ILIKE over full_name/email/phone with `%`/`_`/`\` escaped via
`_escape_eom_directory_like_pattern`, plus a digits-only phone comparison
when `_eom_directory_phone_search_digits` classifies the query as
phone-shaped), and an optional keyset predicate
`(created_at, id) < (cursor_created_at, cursor_id)`; ordering is
`created_at DESC, id DESC` -- immutable columns, so a rename mid-traversal
cannot drop or duplicate a row (the same reasoning documented on
`list_billing_recipients`).

The route (funnel.py) authenticates with the existing
`require_eom_funnel_api` + `require_eom_funnel_actor` dependencies, rejects
unknown query-parameter names, validates search/cursor shape (reusing the
lead-review cursor codec), overfetches limit+1 to derive hasMore/nextCursor,
and validates every row through `EOMContactDirectoryItem`
(extra='forbid'; `status` pinned to Literal['active']; contact_type and
customer_type validated against the operator boundary's own constant sets so
the projection is a second, independent enforcement of admission).
`contact.directory` joins `_CAPABILITY_ROUTES`, so `served_capabilities()` /
`served_capability_routes()` advertise it only while the route is actually
registered.

## Intentional

- Lost leads (lead_stage='lost', status still 'active') are included, with
  their stage: the pipeline hides them by design, and the directory is
  exactly where a pipeline-hidden record must remain findable. Hiding them
  here would recreate the disappearing-record defect one stage later.
- The pipeline read's latest-intake email/phone overlay is NOT reused: the
  directory's job is discoverability of the canonical record, and search over
  interaction-metadata overlays would triple the query for marginal recall.
- The identity-matching digit SQL (extension-stripping + RIGHT-10) is not
  reused for search: identity needs exact-suffix semantics, search needs
  substring containment; a false positive in search is a visible extra row,
  not a wrong write target.
- Unknown query-parameter names are rejected (422) on this route only. The
  pipeline read keeps FastAPI's tolerant default because its callers predate
  this slice; the directory has exactly one caller, so it can be exact.
- No new envelope fields on `/eom-funnel/leads`: the manifest already carries
  the capability name and route signature, which is the proof callers use.

## Deferred

- Archived-contact visibility (an explicit include-archived view) belongs to
  the later archive/restore slice; the admission predicate and its Literal
  mirror will both need widening together.
- Directory reads over interaction-derived contact info (the intake overlay)
  if operators report missing search recall on web-form leads.

Parking predicate: park anything that widens the projection, admits more
lifecycle states, or adds mutation semantics -- this slice is read-only
 discovery, and its consumers gate on the capability manifest, so widening
later is additive.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_eom_contact_directory.py` with
  ATLAS_MIGRATION_TEST_DATABASE_URL set: 32 passed (26 route/unit + 6
  real-Postgres), public.contacts count and max(updated_at) identical before
  and after, zero leftover test schemas.
- `python -m pytest tests/test_eom_funnel_capability_manifest.py
  tests/test_eom_link_verification.py tests/test_eom_lead_conversion.py`:
  284 passed total -- the untouched funnel surfaces stay green.
- Negative controls (each exercised locally, test failed, enforcement
  restored): tenant-scope condition removed; status='active' condition
  removed; LIKE escaping removed; kind Literal loosened to str;
  unknown-parameter rejection removed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 3 |
| `atlas_brain/eom_api/funnel.py` | 151 |
| `atlas_brain/services/crm_provider.py` | 131 |
| `plans/PR-EOM-Contact-Directory.md` | 296 |
| `tests/test_eom_contact_directory.py` | 763 |
| **Total** | **1344** |
