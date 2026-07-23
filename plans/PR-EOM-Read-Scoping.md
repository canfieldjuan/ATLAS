# PR-EOM-Read-Scoping

## Why this slice exists

Issue #2151 read-scoping (the deferred half of Phase 2, named in
plans/PR-EOM-Tenant-Separation.md Deferred). After #2155, writes are
tenant-stamped and the backfill classified 679 rows — but every read on the
CRM MCP surface is still all-tenant by default. Verified at HEAD:

- `atlas_brain/mcp/crm_server.py` tools take `business_context_id` as an
  optional arg and pass it through only when supplied (search `:111`,
  list `:352`, create `:241`); `get_contact` (`:179`) and every other
  id-addressed tool (`update/delete/log_interaction/get_interactions/
  get_contact_appointments`) perform no tenant check at all —
  `crm_provider.get_contact` is `SELECT * ... WHERE id = $1` (`:239-241`).
- Consequence: an assistant session operating "the EOM CRM" reads and can
  mutate churnsignals rows (and vice versa), and the MCP `create_contact`
  tool keeps minting NULL-context rows — the exact provenance hole called
  out in PR #2155 review ("source is settable via the MCP tool while
  business_context_id is optional").

### Problem-derived contract

- Root cause: the MCP CRM surface has no notion of a deployment tenant —
  scoping exists only as an optional per-call argument nobody defaults, so
  cross-tenant reads/mutations are the silent default.
- Correct fix must touch/change: a deployment-level default
  (`MCPConfig.crm_default_business_context`, env
  `ATLAS_MCP_CRM_DEFAULT_BUSINESS_CONTEXT`); the CRM MCP tools to honor it —
  scoped search/list (default page + NULL-context legacy page, the
  claimable population from #2153), fail-closed tenant guards on all five
  id-addressed tools, and default-stamping on the MCP create tool.
- Must not change: `crm_provider` (already has the scoped + IS-NULL
  filters), any B2B module, `atlas_brain/api/contacts.py` (shared intel-UI surface
  serving both tenants — see Intentional), behavior when no default is
  configured (legacy unscoped, backward compatible), the lead-intake
  endpoint, schema.

## Scope (this PR)

Ownership lane: eom-crm/lead-funnel
Slice phase: vertical slice

1. `MCPConfig.crm_default_business_context` (default None — nothing changes
   until a deployment opts in via env).
2. `atlas_brain/mcp/crm_server.py`: `_default_context()` /
   `_visible_under_default()` / `_scoped_search()` / `_guard_contact_id()`
   helpers; `search_contacts` + `get_contact`-by-name use scoped search
   (default page, then NULL-context page; explicit arg always wins);
   `list_contacts` and `create_contact` default their context arg; the five
   id-addressed tools (`get/update/delete/log_interaction/
   get_interactions/get_contact_appointments`) treat foreign-tenant rows as
   "not found" (fail-closed, no cross-tenant existence leak).
3. Proof: `tests/test_crm_read_scoping.py` — 14 tests covering default+
   fallback search order, explicit-arg precedence, no-default legacy
   behavior, foreign-tenant invisibility on reads and refusal on
   mutations, NULL-legacy visibility, create default-stamp, list default.

### Review Contract

- Acceptance criteria:
  1. With a default configured and no explicit arg: search queries the
     default tenant page, then the NULL-context page; a scoped hit skips
     the fallback (test-asserted call order).
  2. An explicit `business_context_id` always wins and suppresses the
     fallback.
  3. No default configured → byte-identical legacy behavior (unscoped, no
     guards) — test-asserted.
  4. `get_contact` on a foreign-tenant UUID returns `found:false`;
     NULL-context and same-tenant rows remain visible.
  5. `update/delete/log_interaction` against a foreign-tenant contact
     refuse without invoking the provider mutation (test-asserted
     not-awaited).
  6. MCP `create_contact` stamps the default when the caller passes no
     context (closes the NULL-minting hole from #2155 review); explicit
     context wins.
  7. A bare `search_contacts()` call is valid when a default exists
     (list-my-tenant), preserving the requires-one-of error otherwise.
- Reachability proof: the CRM MCP server tools themselves (stdio/SSE per
  `settings.mcp`); deployment opt-in is
  `ATLAS_MCP_CRM_DEFAULT_BUSINESS_CONTEXT=effingham_maids` in the Atlas
  `.env` (operator step, this box, post-merge). Observable effect: EOM
  assistant sessions see/mutate only EOM + legacy-NULL rows; churnsignals
  rows read as nonexistent.
- Affected surfaces: CRM MCP tools only. `atlas_brain/api/contacts.py` untouched.
- Risk areas: hiding legacy NULL rows (mitigated: NULL page is part of the
  default scope everywhere); breaking B2B MCP usage (mitigated: explicit
  arg wins; and the default is opt-in per deployment); double-query cost
  on scoped search miss (two indexed lookups, `idx_contacts_business_context`).
- Reviewer rules triggered: R1 (#2151 read-scoping), R2 (14 tests), R3
  (tenant isolation, fail-closed guards, no existence leak), R4 (mutations
  refused pre-provider on foreign rows), R5 (no-default = legacy behavior;
  default None), R10 (helpers shared across tools, no per-tool drift), R11
  (new config field, env-driven, default off), R12 (deployment: inert
  until the env var is set; no restart-order concerns), R14.

### Files touched

- `atlas_brain/config.py`
- `atlas_brain/mcp/crm_server.py`
- `plans/PR-EOM-Read-Scoping.md`
- `tests/test_crm_read_scoping.py`

## Mechanism

One config field + four small helpers. `_scoped_search` mirrors the
tenant-page-then-NULL-page resolution the reviewers converged on in #2153
for the provider dedupe, so both surfaces define "my population" the same
way: `{default tenant} ∪ {NULL legacy}`. Id-addressed tools resolve the
contact once via `get_contact` and fail closed; mutation providers are
never invoked for foreign rows. Everything is inert until the env var is
set, which keeps every other Atlas deployment/test unaffected.

## Intentional

- **`atlas_brain/api/contacts.py` untouched** — the intel UI consumes it for BOTH
  tenants; defaulting it to EOM would break churnsignals timelines. Its
  (pre-existing, already-tracked) authlessness is a separate concern; the
  #2153 slice already stopped leaking contact ids from the public intake.
- **Guard costs one extra `get_contact` per id-addressed call** — accepted:
  correctness over a single indexed PK read; only paid when a default is
  configured.
- **`delete_contact` guarded too** even though EOM rarely deletes —
  cross-tenant delete is the worst-case accident.
- **No sentinel for "all tenants"** — cross-tenant admin reads can pass the
  other tenant explicitly per call; a wildcard would reopen the hole by
  convention.

## Deferred

- `atlas_brain/api/contacts.py` auth + optional scoping (own slice; UI-coupled).
- Operator env opt-in on this box post-merge
  (`ATLAS_MCP_CRM_DEFAULT_BUSINESS_CONTEXT=effingham_maids`).
- Phase 3 residue: 4 unclassifiable NULL rows (3 `manual`,
  1 `manual_invoice_setup`) for operator eyeball.

Parked hardening: none new.

## Verification

- Suite `tests/test_crm_read_scoping.py` — 14 passed.
- Suites `tests/test_tenant_stamping.py` + `tests/test_leads_intake.py` —
  44 passed combined (adjacent lane suites).
- `python -m py_compile` on both touched Python files — clean.
- NOT run: live MCP session against the running server (env not yet set;
  post-merge operator step, then a scoped search spot-check).

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/config.py` | 8 |
| `atlas_brain/mcp/crm_server.py` | 73 |
| `plans/PR-EOM-Read-Scoping.md` | 152 |
| `tests/test_crm_read_scoping.py` | 193 |
| **Total** | **426** |
