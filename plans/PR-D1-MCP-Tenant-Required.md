# PR-D1-MCP-Tenant-Required

## Why this slice exists

Website #124 (D1), the highest-risk 0D child: the agent-callable MCP
`create_contact` could mint a contact with no tenant.

### Correction to the issue premise (verified on `main` d855836a6)

Most of #124's stated hole is already closed by #2298. What is actually true:

- `create_contact` already **refuses EOM-tenant creation** and routes it to
  ingress (`crm_server.py`, the `== EOM_BUSINESS_CONTEXT_ID` guard).
- `update_contact` is already tenant-safe: scoped lookup (`_guarded_contact`),
  claim-on-write (`_claim_if_legacy`), and an EOM stage-change guard. It never
  mints a tenant.
- Criterion 1 ("route through the canonical domain tier") does not map: the
  boundary is EOM-specific and EOM is already blocked; there is no generic tier
  for a non-EOM (e.g. churnsignals) contact.
- Criterion 3 ("normalized matcher, not substring") is a provider-wide
  `search_contacts` change shared by all callers -- deferred, as in D4.

The one real, reachable, unmet gap is **criterion 2**: a NULL-tenant create.

### Problem-derived contract

- Root cause: `effective_business_context_id = business_context_id or
  _default_context()` can be `None`, the EOM guard does not catch `None`, and the
  provider then writes a tenantless row. Prod is exposed -- the live runtime sets
  no `ATLAS_MCP_CRM_DEFAULT_BUSINESS_CONTEXT`, so `_default_context()` returns
  `None`.
- Correct fix touches: `create_contact` gains a tenant-required refusal before
  the provider write; tests pin it.
- Must not change: the EOM guards, `update_contact`, `_default_context()`
  semantics, the provider matcher, or non-EOM creates that supply a tenant.

## Scope (this PR)

Ownership lane: eom-crm/lead-funnel
Slice phase: production hardening

1. In `create_contact`, after computing `effective_business_context_id`, refuse
   with a typed error when it is `None`/blank -- before the EOM guard (a NULL
   tenant is never EOM and would otherwise slip past it) and before the provider
   write.

### Files touched

- `atlas_brain/mcp/crm_server.py`
- `plans/PR-D1-MCP-Tenant-Required.md`
- `tests/test_crm_read_scoping.py`

### Review Contract

1. An agent create with no resolvable tenant is refused and the provider is
   never called.
2. A create with an explicit tenant, or with a configured default, still creates
   -- only "no tenant at all" is newly rejected.
3. A whitespace-only tenant is treated as missing (the `.strip()` is
   load-bearing).
4. The EOM-creation guard and `update_contact` are unchanged.

Affected surfaces: `create_contact` only. No provider change (the two `INSERT
INTO contacts` sites are untouched; 0A's guard passes). No schema change.

Risk areas: over-refusal breaking a legitimate agent create. Probed by the
explicit-tenant and configured-default tests, which must still create. The only
newly-rejected case is a create that would have produced a NULL-context row --
which is the intended fix (criterion 4: a write that previously succeeded now
fails).

- Reviewer rules triggered: R1, R2, R3, R5, R14. (R3: this governs which
  agent-authored writes reach the CRM under which tenant. R5: it changes the
  `create_contact` response contract -- a create with no resolvable tenant now
  returns `{success: false}` where it previously returned a created contact.
  That is deliberate and is criterion 4's required break; it is not a silent
  schema change -- the response shape (`{success, error}` vs `{success,
  contact}`) is the tool's existing error convention, and every call that
  supplies a tenant, explicitly or by configured default, is unchanged. No
  persisted-data or wire-format compatibility is affected.)

**boundary-probe:** both sides -- missing/blank tenant refuses; explicit and
default-configured tenants create.

**Mutation-probe (run, not asserted):** removing the guard fails 2 tests;
weakening it to a truthiness check (dropping `.strip()`) fails the blank-tenant
test.

## Mechanism

One typed-refusal branch inserted before the existing EOM guard.

## Intentional

- **Refuse, do not default.** #124 criterion 2 is explicit: a missing tenant is
  a typed refusal, not a silent default. Inventing a fallback tenant would
  reproduce the mis-tenanting this closes.
- **Before the EOM guard.** A NULL tenant is never the EOM tenant, so ordering
  matters -- the tenant check must run first or `None` slips past.

## Deferred

- The provider substring matcher in `search_contacts` -- provider-wide, separate
  change (as in D4).
- D3, D5 (website #126, #128).

Parking predicate: this slice parks everything except the create-time tenant
requirement.

Parked hardening: none.

## Verification

```
$ python -m pytest tests/test_crm_read_scoping.py -q
76 passed

$ python scripts/check_contact_write_boundary.py --baseline ... --inventory-baseline ...
(exit 0 -- no new contact write site)
```

The 6 pre-existing `test_mcp_servers.py` failures (Email/IMAP/Twilio/Calendar
tools) are unrelated to CRM and present on `main` before this change.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/mcp/crm_server.py` | 18 |
| `plans/PR-D1-MCP-Tenant-Required.md` | 132 |
| `tests/test_crm_read_scoping.py` | 84 |
| **Total** | **234** |
