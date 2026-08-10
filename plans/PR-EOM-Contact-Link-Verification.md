# PR-EOM-Contact-Link-Verification

## Why this slice exists

Website #167 is the tracker half of Slice 0F (write-boundary observability, #113).
It was split out of ATLAS #2344 after three review rounds showed two of its signals
could not be closed from inside ATLAS. The stale-reservation signal shipped in the
tracker on its own. The remaining one is blocked here.

The tracker stores its own copy of an Atlas contact id in
`customers.atlas_contact_id`. A non-null value there proves nothing: it proves a
write happened once, not that the contact still exists or that it was ever an EOM
contact. Nothing on either side notices when that link stops resolving, so a
customer whose Atlas contact is gone looks perfectly healthy in the tracker until
someone opens the record by hand. That is the same silent-failure class Slice 0
exists to close, and the reason the original defect survived long enough to need a
backfill.

The tracker cannot ask. `_atlas_funnel_read` is hard-locked to `/eom-funnel/leads`
(`backend/time_tracker_api.py:3376`), and no Atlas route answers contact existence
anyway.

### Problem-derived contract

- Root cause: no read boundary exists for "does this contact id still resolve to a
  live EOM contact", so a system holding an Atlas contact id cannot validate it.
- Correct fix must touch/change: a read-only EOM funnel route that takes contact
  ids and returns which are known; a tenant-scoped provider query behind it; the
  capability manifest, because callers gate on it; and the CI job's test list, or
  the proof never runs.
- Must not change: any write path, the funnel's auth model, the contacts schema,
  or what the funnel discloses about a contact beyond its id.

## Scope (this PR)

Ownership lane: eom-crm/lead-funnel
Slice phase: Vertical slice

1. Add `GET /eom-funnel/known-contacts`: given up to 100 `contact_id` query
   values, return the subset that names a live `effingham_maids` contact, and
   nothing else about them.
2. Add `DatabaseCRMProvider.list_known_eom_contact_ids`, tenant-scoped in the SQL
   rather than filtered afterwards.
3. Prove both directions, including a real-PostgreSQL proof that the tenant scope
   is in the query and not just in the docstring.

Not in this PR: the tracker-side consumer. It is a separate PR in eom-timetracker
and depends on this deploying first.

### Review Contract

- Acceptance criteria:
  - A dangling id is reported unknown and a live id is reported known — settled by
    `tests/test_eom_known_contacts.py::test_a_dangling_link_is_reported_as_unknown`
    and `::test_a_live_contact_is_not_reported_as_dangling`.
  - A contact in another business context is never reported known, proven against
    real PostgreSQL rather than a stub — settled by
    `::test_tenant_scope_holds_against_real_postgres`, which seeds an
    `effingham_maids` row beside a `churnsignals` row and asks for both.
    Negative control run: removing `WHERE c.business_context_id =
    'effingham_maids'` from the provider query fails this test.
  - The response never contains an id the caller did not submit — settled by
    `::test_an_id_the_caller_never_submitted_is_never_returned`, which stubs a
    provider that returns an extra id. Negative control run: replacing the
    request-ordered filter with `list(known_set)` fails this test.
  - An empty request is rejected rather than answered "clean" — settled by
    `::test_an_empty_check_is_rejected_rather_than_answered_clean` (422, provider
    never called).
  - The cap holds on both sides: 100 accepted, 101 rejected — settled by
    `::test_exactly_the_cap_is_accepted` and `::test_the_cap_is_enforced_at_the_boundary`.
  - A malformed id is rejected, not silently dropped — settled by
    `::test_a_malformed_id_is_rejected_not_silently_dropped`.
  - The route refuses unauthenticated and wrong-token callers without reaching the
    provider — settled by `::test_the_route_refuses_an_unauthenticated_caller` and
    `::test_a_wrong_service_token_is_refused`.
  - The response body carries only `knownContactIds`, `checked`, `limit` — settled
    by `::test_the_route_discloses_no_contact_data_beyond_the_id`.
  - The capability is advertised, since callers gate on the manifest — settled by
    `::test_link_verification_is_advertised_in_the_capability_manifest`.
  - The new test file actually runs in CI — settled by
    `.github/workflows/atlas_eom_lead_pipeline_checks.yml:216` (pytest argument)
    plus lines 72 and 155 (both path-filter blocks, so a test-only edit still
    triggers the job).
- Reachability proof: real entrypoint is `GET /api/v1/eom-funnel/known-contacts`
  on the aggregate app, behind `require_eom_funnel_api` +
  `require_eom_funnel_actor` like every other funnel route. Observable effect is
  the response body; the route performs no writes.
- Affected surfaces: `atlas_brain/eom_api/funnel.py` (route, response model,
  capability entry, cap constant), `atlas_brain/services/crm_provider.py` (one new
  read method), the EOM lead-pipeline CI job.
- Risk areas: tenant leakage through an unscoped id lookup; disclosure beyond the
  id; unbounded id lists as a query-cost or URL-length vector; a capability that
  callers cannot discover; a new test file that silently never runs in CI.
- Reviewer rules triggered: R1 (tenant scope), R2 (read-only boundary), R7
  (input bounds), R9 (capability manifest), R14 (CI registration).

### Boundary-change enumeration

- Boundary path/seam: `GET /eom-funnel/known-contacts` — a new admission boundary
  over a list of contact ids.
- Replaced-path behaviors: none. No existing route answered this question, so
  nothing is replaced and no caller changes behavior.
- Guard-relevant fields: `contact_id` (repeatable query param) — list length
  (1..100), per-value UUID parseability, duplicates; and
  `contacts.business_context_id`, which is part of the query rather than a filter
  applied to its result.
- Caller x input shape: authenticated tracker x {live id, dangling id,
  foreign-tenant id, duplicate ids, malformed id, empty list, 100 ids, 101 ids};
  unauthenticated caller and wrong-token caller x any id (both refused before the
  provider is reached).

### Deployed-config probing

N/A - no guard/config boundary change. The route reads no environment or config
value; the cap is a module constant and the tenant is the same literal the other
EOM funnel reads already use.

### Files touched

- `.github/workflows/atlas_eom_lead_pipeline_checks.yml`
- `atlas_brain/eom_api/funnel.py`
- `atlas_brain/services/crm_provider.py`
- `plans/PR-EOM-Contact-Link-Verification.md`
- `tests/test_eom_known_contacts.py`

## Mechanism

The route takes `contact_id` as a repeatable query parameter, deduplicates while
preserving submission order, and hands the list to
`list_known_eom_contact_ids`. The provider runs a single
`WHERE c.business_context_id = 'effingham_maids' AND c.id = ANY($1::uuid[])`, so
tenant scope is part of what the query can return rather than a filter applied to
its result — an unscoped variant would have to leak first and be cleaned up after.

The route then answers in terms of what was asked: it intersects the provider's
rows against the submitted ids and emits them in submission order. That is not
belt-and-braces. Without it, a provider row the caller never submitted would
appear in the response as a "known" verdict the caller cannot attribute to any
link it holds.

Ids ride in the query string, so the 100-id cap is a URL-length budget: 100 ids
costs roughly 4.8 KB of `contact_id=<uuid>&`, inside the 8 KB request line proxies
accept. Callers with more links page through them.

Archived and lost contacts count as known. The question is whether the link
resolves; a link to a closed contact is intact. Only a dangling or cross-tenant id
means the write boundary was bypassed.

## Intentional

- **A cross-tenant id is reported exactly like a missing one.** Distinguishing
  them would be more useful to the caller and is deliberately refused: this
  credential is scoped to EOM, and confirming that some id exists under another
  business context would make the route a cross-tenant existence oracle. Both
  answers mean the same thing to the caller anyway — the link does not point at an
  EOM contact.
- **Id-only response.** A caller holding a stored contact id is asking whether its
  link resolves; that needs no name, email, or phone, so none is disclosed. The
  existing `/eom-funnel/leads` projection stays the only route that returns
  identity fields.
- **GET with repeated query params, not POST with a body.** The operation is a
  read and the method should say so. The cost is the URL-length cap above, which
  is the reason the cap exists at all.
- **An empty id list is a 422, not an empty-and-clean 200.** A caller that
  accidentally sends nothing must not read the answer as "every link is fine" —
  that is the false-assurance failure this slice exists to remove.
- **Rejected: a per-id `GET /eom-funnel/contacts/{id}`.** Simpler, but the tracker
  would issue one request per customer to audit its links, which makes the audit
  expensive enough to not run — and a monitor that does not run is the thing being
  fixed.

## Deferred

- The tracker-side consumer (website #167's remaining half): a linkage-validity
  signal in the tracker audit that calls this route. Separate repo, separate PR,
  and it needs this deployed first. #167, #113 and #107 stay open until it lands.

Parked hardening: none.

## Verification

- `tests/test_eom_known_contacts.py` — 14 tests, run against a throwaway
  `postgres:16` with `ATLAS_MIGRATION_TEST_DATABASE_URL` set: **14 passed**.
- Negative controls, both run and both failing as required before restore:
  removing the tenant predicate from the provider query fails
  `test_tenant_scope_holds_against_real_postgres`; replacing the request-ordered
  intersection with `list(known_set)` fails
  `test_an_id_the_caller_never_submitted_is_never_returned`. Result: pass (each
  control failed only its target test; restored source passes all 14).
- Neighbouring EOM suites for regression —
  `test_eom_funnel_capability_manifest.py`, `test_eom_lead_conversion.py`,
  `test_eom_contacts_api_tenant_scope.py`, `test_crm_read_scoping.py`,
  `test_eom_known_contacts.py`: **495 passed**.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 3 |
| `atlas_brain/eom_api/funnel.py` | 67 |
| `atlas_brain/services/crm_provider.py` | 33 |
| `plans/PR-EOM-Contact-Link-Verification.md` | 204 |
| `tests/test_eom_known_contacts.py` | 347 |
| **Total** | **654** |
