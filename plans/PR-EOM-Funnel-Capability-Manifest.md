# PR-EOM-Funnel-Capability-Manifest

## Why this slice exists

Website (Vercel) and tracker (Render) auto-deploy from `main`. Atlas is deployed
by hand -- host pull plus `systemctl --user restart atlas-api`, as ATLAS #2275
documents. So callers routinely run ahead of the backend, and nothing in the
protocol lets them find out.

The observed instance: tracker PR #124 (lost/reopen proxy routes) and website PR
#102 (Mark-lost UI) merged and auto-deployed while their Atlas counterpart
`f82e90820` sat on `main` undeployed. A shipped, visible button called routes
its backend did not serve.

This is the Atlas half of website #112 (Slice 0E), and first in that issue's
enforced Atlas -> tracker -> website ordering.

### Why this slice is over the 400-LOC target

497 added lines, of which **75 are production code**:

| File | LOC | What it is |
|---|---:|---|
| `plans/PR-EOM-Funnel-Capability-Manifest.md` | 214 | this document, required by the plan contract |
| `tests/test_eom_funnel_capability_manifest.py` | 205 | both-directions boundary probe + degradation case |
| `atlas_brain/eom_api/funnel.py` | 75 | the actual change |
| `tests/test_eom_lead_conversion.py` | 3 | the new key on three existing envelope assertions |

The production change is one field, one map, and one derivation. Splitting it
would ship a response field that nothing populates, or a derivation with nothing
to derive from -- neither half is independently meaningful.

The test weight is deliberate rather than padding: this is a guard whose only
value is refusing to advertise a capability the build does not serve, so it needs
both directions proved, a forced-degradation case standing in for the live skew
that ATLAS #2300 already closed, and the empty-queue and pre-manifest-caller
cases. Tests plus the required plan doc are 84% of the diff and 0% of the
behaviour.

### Correction to the issue's premise, verified live

Website #112 states the skew is currently live. **It is not, as of this PR.**
Deploying `497d3155f` (ATLAS #2300) closed it: the running service's own
served OpenAPI document now lists `/api/v1/eom-funnel/leads/{contact_id}/lost` and
`.../reopen`.

That resolves the *instance* and changes nothing about the *cause*. The deploy
asymmetry is untouched, so the next slice reopens the same gap. It does mean the
issue's acceptance criterion "with Atlas not serving `/lost`, the portal does not
present a failing Mark-lost button" can no longer be demonstrated against the
live service, so the degradation path is forced in test instead -- see
`test_manifest_omits_a_capability_whose_route_is_not_registered`.

### Problem-derived contract

- Root cause: the funnel API has no way to state what it serves, so a caller has
  no alternative to assuming. Assumption is correct exactly until Atlas lags,
  which the deploy asymmetry guarantees it eventually will.
- Correct fix must touch/change: an existing funnel read gains a capability
  field, sourced from what this build actually serves. Additive, so a caller
  that predates it keeps working.
- Must not change: any funnel mutation route, the closed lead projection's
  identity fields, auth, or the CRM. No new route, no new dependency.

## Scope (this PR)

Ownership lane: eom-crm/lead-funnel
Slice phase: vertical slice

Capability negotiation for website #112 (Slice 0E), Atlas half -- first of the
three layers in that issue's enforced Atlas -> tracker -> website ordering.

Overlap note: ATLAS #2301 (`claude/pr-eom-lost-replay-generation`) also touches
`tests/test_eom_lead_conversion.py`. Checked, and it does not touch the three
envelope assertions edited here, so the two are textually independent. Whichever
lands second should still re-run that file rather than trusting the merge.

1. `capabilities: list[str]` on `EOMLeadReviewResponse`, defaulting to empty.
2. `_CAPABILITY_ROUTES`, an explicit name -> (method, path) enumeration.
3. `served_capabilities()`, which intersects that map with the router's
   **actually registered** routes and memoizes the result.

### Files touched

- `atlas_brain/eom_api/funnel.py`
- `plans/PR-EOM-Funnel-Capability-Manifest.md`
- `tests/test_eom_funnel_capability_manifest.py`
- `tests/test_eom_lead_conversion.py`

### Review Contract

1. The manifest never advertises a route this build does not serve, which is the
   only direction that can recreate the original failure.
2. A caller written before this slice still reads every field it knew; the added
   key is additive and the tracker's `_parse_atlas_lead_review_response` ignores
   unknown keys.
3. No mutation route, auth path, or CRM call changes.
4. The closed lead projection stays closed -- the envelope assertions remain
   exact-equality, so an unexpected key still fails them.

Affected surfaces: the `GET /eom-funnel/leads` response envelope, and the three
existing envelope assertions that pin it by exact equality. No route is added,
removed, or re-authenticated.

Risk areas: the map is hand-written strings with no compiler tying them to
routes, so a typo would silently under-advertise -- which reads exactly like
"this build does not serve it" and would disable a working control. Probed by
`test_capability_map_has_no_entry_for_an_unregistered_route`, which fails on any
map entry that matches no registered route. Second risk is the memoization
leaking across a route change; probed by an autouse fixture that resets the
cache and by computing lazily rather than at import, since the routes are
registered by decorators below the definition.

- Reviewer rules triggered: R1, R2, R5, R9, R10, R12, R14.

R5 (backward compatibility) applies because this changes a public API response
shape. The change is additive: `capabilities` is a new key with a default, every
prior key keeps its name, type, and meaning, and the tracker's
`_parse_atlas_lead_review_response` reads by `.get()` and ignores unknown keys.
Not BREAKING, and pinned by
`test_pre_manifest_caller_still_reads_every_field_it_knew` (asserts each key the
old caller reads is still present) and
`test_response_model_defaults_capabilities_for_a_caller_that_omits_it`
(constructing the envelope without the field does not raise). The three existing
envelope assertions stay exact-equality, so a silent rename or drop still fails.

R9 (guard-shaped) applies: this is a declaration of what is allowed to be
called, and it fails on its second side if it ever over-advertises.

R12 (deployment safety and CI enrollment) applies because the ordering is
load-bearing -- Atlas must be **deployed**, not merely merged, before the tracker
half ships. Documented in Deployment compatibility above, with the rollback path
stated: reverting past this commit removes the key, and a manifest-aware caller
must read absence as "advertise nothing" and disable the gated controls, which is
the same degradation path as an old Atlas. Disable path: there is no flag to turn
off, because the field is inert until a caller reads it -- this PR ships a
statement of fact and no behaviour change. CI enrollment verified rather than
assumed: `tests/unit_gate_baseline.txt` is a known-**failures** list, not a test
registry (the gate fails only on a node absent from it), so a new passing test
file is collected by the repo-wide run and must not be added there --
`tests/test_eom_lead_conversion.py` is likewise absent from it.

R14 (verify against the codebase, not the PR story) applies universally. The
concrete instance here: website #112 asserts the lost/reopen skew is live, and
the running service's own served OpenAPI document contradicts it. Recorded above rather
than inherited from the issue text.

**boundary-probe:** both directions plus the forced-degradation case.
Advertised-implies-registered
(`test_every_advertised_capability_has_a_registered_route`);
registered-and-mapped-implies-advertised, with a non-empty assertion so the loop
cannot pass vacuously
(`test_every_mapped_and_registered_route_is_advertised`); and a mapped-but-
unregistered capability must be omitted while its neighbours are unaffected
(`test_manifest_omits_a_capability_whose_route_is_not_registered`). Empty-queue
and pre-manifest-caller cases are covered separately, because deriving
capabilities from rows rather than routes would disable every control on an idle
portal.

**Mutation-probe (run, not asserted):** replacing the `if signature in
registered` filter with an unconditional include makes
`test_manifest_omits_a_capability_whose_route_is_not_registered` fail, so that
test is not vacuous. Restored before commit.

**Guard-class closure declaration**

- **Member set:** `_CAPABILITY_ROUTES` keys, the capability names advertisable
  by this service.
- **Names are CLOSED / ENUMERATED.** An explicit literal dict. A route cannot
  join by existing, by path shape, or by naming convention.
- **Advertised membership is DERIVED.** A name is emitted only if its
  `(method, path)` is registered on the router. The manifest therefore cannot
  claim a capability this build does not serve -- the failure the slice exists
  to prevent.
- **Out-of-set default: OMIT.** An unregistered or unmapped route is absent from
  the manifest. A caller reading absence as "disable the control" fails closed;
  the worst case is a disabled control that would have worked, never a visible
  control that 404s.
- **Both sides covered:** advertised-implies-registered, registered-implies-
  advertised, and forced-omission.

### Boundary-change enumeration

- Boundary path/seam: `GET /eom-funnel/leads` response envelope.
- Replaced-path behaviours: none. One key is added; every prior key keeps its
  name, type, and meaning.
- Guard-relevant fields: `capabilities`.
- Caller x input shape: pre-manifest caller x populated queue; manifest-aware
  caller x empty queue; envelope constructed without the field at all.

**Reachability proof:** the endpoint test asserts `lead.lost` and `lead.reopen`
appear in a real HTTP response through the ASGI transport, not on the model in
isolation.

## Mechanism

An explicit capability-name map, intersected at first call with the router's
registered `(method, path)` pairs, memoized, and serialized on the existing
lead-review read. Computed lazily because the routes are registered by
decorators further down the module; an import-time constant would read a
partially-populated router and under-report.

## Intentional

- **Derived, not declared.** A hand-maintained list is the same class of artifact
  as the assumption it replaces: it can be wrong in the dangerous direction. This
  cannot over-advertise by construction.
- **On an existing read, not a new endpoint.** The portal already calls this
  route to populate the queue, so gating needs no extra round trip and no new
  auth surface.
- **Semantic names, not raw paths.** Callers should gate on "can this deployment
  mark a lead lost", not on a URL template that will be rewritten someday.
- **Omission rather than a served/not-served boolean map.** Absence is the
  natural encoding of "this build predates the capability", and it is what a
  caller running against an even older Atlas already sees.

## Deferred

- Tracker graceful degradation and website control gating: the remaining two
  thirds of website #112, in its enforced Atlas -> tracker -> website order.
  This PR is the prerequisite both consume.
- Provenance and unlinked-customer alerting: website #113.
- Automating Atlas deployment: real gap, separate infrastructure decision. This
  slice makes the skew safe, not impossible.

Parking predicate: this slice parks everything a caller does with the manifest,
and ships only the Atlas-side statement of fact.

Parked hardening: none.

## Verification

```
$ python -m pytest tests/test_eom_funnel_capability_manifest.py -q
8 passed

$ python -m pytest tests/test_eom_lead_conversion.py tests/test_eom_render_profile.py \
    tests/test_eom_funnel_capability_manifest.py -q
242 passed

$ python scripts/check_guard_class_closure.py
OK: no guard-shaped change without a property test.
```

Live premise check, read-only, against the running service:

```
$ curl -s http://127.0.0.1:8012/openapi.json | ...
/api/v1/eom-funnel/leads/{contact_id}/lost
/api/v1/eom-funnel/leads/{contact_id}/reopen
```

## Deployment compatibility

Per website #112, every PR in this arc declares its minimum peer versions.

| Peer | Minimum | Note |
|---|---|---|
| Atlas | this commit | Must be **deployed**, not merely merged, before the tracker half ships. |
| Tracker | none | Pre-manifest callers ignore the new key. |
| Website | none | Pre-manifest callers ignore the new key. |

Rollback: reverting Atlas past this commit removes the `capabilities` key. A
manifest-aware caller must read an absent key as "advertise nothing" and disable
the gated controls, which is the same degradation path as an old Atlas. That
consumer-side behaviour is specified and tested in the tracker half.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/eom_api/funnel.py` | 72 |
| `plans/PR-EOM-Funnel-Capability-Manifest.md` | 276 |
| `tests/test_eom_funnel_capability_manifest.py` | 215 |
| `tests/test_eom_lead_conversion.py` | 3 |
| **Total** | **566** |
