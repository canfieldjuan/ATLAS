# PR-EOM-Terms-Capability-Manifest

## Why this slice exists

Issue #2156 and merged PR #2505 establish Atlas as the Terms invitation and
acceptance authority, with Tracker as the authenticated proxy. Atlas now serves
the six routes Tracker needs, but the existing capability manifest does not
advertise any of them. Tracker's established compatibility boundary requires an
exact capability name plus the exact registered method/path before it exposes a
provider-backed control. A Tracker-only bridge would therefore either remain
permanently unavailable or bypass the fail-closed deployment-drift guard.

This provider-first prerequisite extends the existing manifest with facts about
routes the same build already serves. It creates no new Terms behavior and no
customer-visible surface by itself.

### Problem-derived contract

- Root cause: Atlas's capability inventory is incomplete. The six existing
  Terms routes are registered on the canonical EOM funnel router, but none has a
  semantic entry in `_CAPABILITY_ROUTES`; consequently Atlas cannot prove to a
  separately deployed Tracker that those exact routes are present in the
  running build.
- Correct fix must touch/change: add one semantic capability name for each of
  the existing invitation issue, invitation revoke, readiness read, delivery
  confirmation, public session, and public acceptance routes; keep advertised
  membership derived from the router's registered method/path pairs; and extend
  the canonical manifest tests so the six exact pairs, their real response
  reachability, and fail-closed omission remain enforced.
- Must not change: do not add, remove, re-authenticate, or change any Terms
  route, request/response model, acceptance service, persistence, migration,
  email, token, IP, publication, readiness, or delivery behavior. Do not touch
  Tracker or Website in this PR. Do not change existing capability names,
  generic manifest derivation, CRM, onboarding, payments, billing, calendar, or
  timekeeping.

## Scope (this PR)

Ownership lane: eom/terms-bridge
Slice phase: Vertical slice
Max files: 3

1. Extend Atlas's existing enumerated EOM funnel capability map with six
   semantic Terms name-to-method/path entries for routes already registered by
   this build.
2. Prove the existing manifest read advertises all six exact pairs through the
   mounted router and keeps the established omit-on-absence behavior.

### Review Contract

- Acceptance criteria:
  - `tests/test_eom_funnel_capability_manifest.py::test_terms_capabilities_pin_the_existing_route_contract`
    proves each of the six semantic names maps to the exact method/path served by
    the current Terms decorators.
  - `tests/test_eom_funnel_capability_manifest.py::test_lead_review_response_advertises_terms_routes`
    calls `GET /eom-funnel/leads` through the mounted ASGI router with an empty
    queue and observes all six names and exact route pairs in the response.
  - Existing `test_every_advertised_capability_has_a_registered_route`,
    `test_every_mapped_and_registered_route_is_advertised`, and
    `test_capability_map_has_no_entry_for_an_unregistered_route` settle both
    directions for the expanded map.
  - Existing `test_manifest_omits_a_capability_whose_route_is_not_registered`
    settles the fail-closed outside-set behavior: a mapped signature absent from
    the router is omitted while unrelated names remain available.
  - Cold diff reconstruction confirms no Terms decorator, model, service,
    persistence, auth, email, token, or consumer repository changed.
- Reachability proof: an authenticated ASGI request to the existing
  `GET /eom-funnel/leads` entrypoint returns the six Terms names and method/path
  pairs even when no lead rows exist.
- Affected surfaces: `atlas_brain/eom_api/funnel.py` capability metadata and
  the focused EOM funnel capability-manifest tests.
- Risk areas: over-advertising a route the build does not serve, method/path
  typo under-advertising a working Terms control, semantic-name drift between
  Atlas and Tracker, response regressions for an empty lead queue.
- Reviewer rules triggered: R1, R2, R3, R5, R10, R12, R13, R14.

### Fix-loop disposition preflight

- Root decision: the reachability proof must exercise the canonical deployed
  `atlas_brain.main_eom:app`, not a test-authored router assembly.
- Source trace: Codex R2/R14 thread at
  `tests/test_eom_funnel_capability_manifest.py:229` -> local `_app()` mounts
  `funnel_mod.router` directly -> `atlas_brain/main_eom.py` mounts the real
  funnel router beneath `/api/v1`.
- Upstream files: `tests/test_eom_funnel_capability_manifest.py`.
- Fix strategy: upstream-root. Replace only the new Terms reachability test's
  synthetic app with the canonical `main_eom.app`, call the deployed path, and
  restore dependency overrides after the assertion.
- Blocking predicate: the test can remain green while the Render entrypoint
  omits or remounts the funnel router.
- Disposition: fix the confirmed in-scope reachability gap in this PR.
- Allowed files: `tests/test_eom_funnel_capability_manifest.py` and this plan.
- Max files: 3.
- Parked hardening target: none; this is required reachability proof, not
  adjacent hardening.

### Guard-class closure declaration

- Member set: the six Terms semantic names and exact method/path signatures
  added to `_CAPABILITY_ROUTES`, together with the focused test oracle that pins
  the same contract.
- Membership is **CLOSED**: this slice covers exactly the six existing routes
  deferred by `PR-EOM-Terms-Acceptance` for the Tracker bridge. Terms authority
  draft/publish/current routes are operator authority surfaces, not Tracker
  invitation/acceptance bridge members, and no other Terms proxy route exists in
  the current router.
- Membership source is **ENUMERATED**: the semantic names are authored in the
  capability map; their signatures are transcribed from the six canonical
  FastAPI route decorators in `atlas_brain/eom_api/funnel.py`. The focused test
  is an independent contract oracle and intentionally repeats those six pairs
  so a name or path change cannot silently alter both implementation and proof.
- Advertised membership remains **DERIVED** at `served_capabilities()`: Atlas
  emits a name only when its mapped signature is present in `router.routes`.
- Out-of-set default is **OMIT**: an unknown semantic name, an unlisted route,
  or a mapped signature absent from the running router is not advertised. This
  is safer because a consumer disables an unavailable control rather than
  sending a customer or operator request to an unproved route.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: `_CAPABILITY_ROUTES` -> `served_capabilities()` ->
  `GET /eom-funnel/leads` capability metadata.
- Replaced-path behaviors: the six registered Terms routes previously produced
  no manifest proof; after this change their exact semantic name/signature pairs
  are included. All existing capability decisions are preserved.
- Guard-relevant fields: semantic capability name, HTTP method, registered
  route path, and membership of the exact `(method, path)` pair in
  `router.routes`.
- Caller x input shape: manifest-aware Tracker x all six exact names and route
  pairs; pre-Terms Tracker x additive unknown response entries; empty and
  populated lead queues x the same route-derived manifest.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: no environment or deployment setting
  participates. The default is the current build's registered EOM funnel router
  plus the explicit capability map.
- Explicit value probe: all six exact Terms method/path pairs are registered and
  appear in the real manifest response.
- Absent value probe: a mapped signature removed from the router is omitted by
  the existing forced-degradation test.
- Default-session/default-context probe: the manifest response advertises the
  route contract when the lead queue is empty; no customer, actor, locale, or
  token state participates in capability derivation.
- Side-effect ordering: N/A. Capability derivation and the exercised lead-list
  response are read-only; this PR adds no write or external call.

### Files touched

- `atlas_brain/eom_api/funnel.py`
- `plans/PR-EOM-Terms-Capability-Manifest.md`
- `tests/test_eom_funnel_capability_manifest.py`

## Mechanism

Six entries join the existing explicit `_CAPABILITY_ROUTES` map. The unchanged
`served_capabilities()` function intersects that map with the actual router
registry before serializing the capability names and corresponding route pairs
on the existing lead-review response. Focused tests pin the new semantic
contract and exercise its real response entrypoint; the generic bidirectional
and forced-omission tests continue to cover the whole map.

## Intentional

- This PR does not add a dedicated capabilities endpoint. Tracker already reads
  the lead-review envelope, and the existing exact name-plus-route mechanism is
  the compatibility contract used by adjacent funnel features.
- Terms authority draft, publish, and current-version routes are not advertised
  to Tracker. They remain Atlas operator-authority surfaces and are not needed
  for the approved customer invitation/acceptance bridge.
- No deployment flag is added. The manifest reports immutable facts about the
  running build and omission already supplies the rollback/degradation path.

## Deferred

- Tracker's closed admin/public Terms proxy, strict request/response projection,
  exact capability gating, and availability fields are the next consumer PR in
  this assigned arc.
- Website Onboarding workspace and public bilingual acceptance UI remain the
  subsequent Website slice from issue #2156.
- Terms copy refinement, Stripe/card vaulting, service-plan choices, and
  payment behavior remain separately operator-owned work under issue #2156.

Parking predicate: UI, copy, analytics, bulk invitation tooling, transport
retry, and consumer ergonomics stay parked unless they prove this manifest can
advertise a route the running Atlas build does not serve.

Parked hardening: none.

## Verification

- Passed locally: `./ops test focused
  tests/test_eom_funnel_capability_manifest.py -q` (`11 passed`).
- Passed locally: targeted `ruff check` on both changed Python files, Python
  compilation, strict guard class-closure audit, and `git diff --check`.
- Scoped formatter inspection: both new hunks are formatter-stable. Whole-file
  `ruff format --check` remains non-clean only on pre-existing lines outside
  this diff; those unrelated lines are intentionally untouched.
- Pending before push: final plan/diff synchronization, guarded local PR review,
  and push wrapper.
- GitHub-only: required Unit Gate and repository PR gates.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/eom_api/funnel.py` | 15 |
| `plans/PR-EOM-Terms-Capability-Manifest.md` | 217 |
| `tests/test_eom_funnel_capability_manifest.py` | 55 |
| **Total** | **287** |
