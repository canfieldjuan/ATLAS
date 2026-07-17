# PR-Followup-Tool-Surface-Qualification

## Why this slice exists

The #2114 qualification made the draft-only follow-up role safe on two axes, and the
next bounded step it named was "read-only / dry-run MCP qualification of the worker's
tool surface -- no send capability belongs in that slice." #2126 fixed the worker's
RESULT shape (server-owned approval, no send action). This slice fixes the complementary
INPUT surface: a deterministic qualifier that admits only read tools, so the qualified
worker can perform no send/mutation at all -- a dry run is the only thing it can do. It
is the enforcement primitive a later slice will point at a live MCP `list_tools` result
to prove the worker's real surface sends nothing.

### Problem-derived contract

From the "no send capability" requirement, a correct fix must:
- Define the read-only tool surface a draft-only follow-up worker may be given, grounded
  in real Atlas MCP read tools.
- Qualify a proposed tool surface fail-closed: any tool that is not a known read tool --
  a send/mutate tool, an unknown tool, or a malformed entry -- disqualifies the surface.
- Close the class rather than blacklist verbs: the mutating tools share no single lexical
  verb (`send_email` vs `create_contact` vs `record_payment` vs `log_interaction`), so a
  denylist leaks; an allowlist choke point does not.
- Guarantee the allowlist itself is genuinely read-only (disjoint from known mutating
  tools), enforced, not just documented.

## Scope (this PR)

Ownership lane: followup-workflow
Slice phase: vertical slice

One new schema module (`followup_tool_surface.py`) with the read-only allowlist, a
fail-closed qualifier, and an import-time self-audit, plus its tests. No runtime caller,
no live MCP connection, no send path. Wiring the qualifier to a live MCP `list_tools`
surface and to a worker is a later slice.

### Review Contract

- Acceptance criteria:
  - [ ] Every allowlisted read tool qualifies; the full read surface qualifies.
  - [ ] Every known mutating tool disqualifies and is flagged as mutating.
  - [ ] A mixed read+send surface disqualifies and names the send tool; an unknown tool
        disqualifies (not flagged mutating); an empty surface is trivially qualified.
  - [ ] Case/whitespace variants and blank/non-string entries fail closed.
  - [ ] The allowlist is disjoint from known mutating tools (enforced at import).
- Reachability proof: N/A -- deterministic qualifier only, no runtime caller and no MCP
  connection. Proof is the test suite.
- Affected surfaces: one new schema module and its test file; nothing imports it yet.
- Risk areas: the allowlist membership (grounded in verified MCP tool names) and the
  fail-closed closure (any non-allowlisted tool rejected); the read-only self-audit.
- Reviewer rules triggered: R14, R2. R2 because the qualifier governs what capability the
  worker may be granted (an authorization surface); R14 for the guard/allowlist.

### Files touched

- `atlas_brain/schemas/followup_tool_surface.py`
- `plans/PR-Followup-Tool-Surface-Qualification.md`
- `tests/test_followup_tool_surface.py`

## Mechanism

`FOLLOWUP_READONLY_TOOLS` is a frozenset allowlist of verified read tools (CRM / Email /
Calendar reads). `qualify_followup_tool_surface(offered)` compares each offered name
EXACTLY (raw, not stripped/normalized -- a tool name is a capability identity) and
disqualifies the surface if any is not an exact allowlist member, returning a frozen
`ToolSurfaceQualification` (qualified, offered, disallowed, mutating, reason). The
allowlist is the closure -- `KNOWN_MUTATING_TOOLS` is not the guard; it drives the
import-time disjointness self-audit (a `raise`, not `assert`, so `-O` cannot strip it)
and sharper negative tests. `is_read_only_tool` is a single-name convenience.

## Intentional

- The guard is a fail-closed allowlist, not a mutating-verb denylist: the mutating tools
  have no common verb (`log_interaction` in particular), so only an allowlist closes the
  class -- any unknown/future tool is rejected by default.
- An empty surface qualifies (it can send nothing). Usefulness -- a worker needing at
  least a lookup tool -- is a separate concern from the no-send guarantee this qualifies.
- The allowlist is the ceiling of what a draft-only worker may be given, grounded in
  verified read tool names; the KNOWN_MUTATING set is representative, not exhaustive,
  because the allowlist closure (not that set) is what rejects a mutating tool.

## Deferred

- Wiring the qualifier to a live MCP `list_tools` result (dry-run connection check) and
  to an actual worker -- a later slice; needs a runtime worker to exist.
- A dedicated read-only follow-up MCP server (the invoicing-readonly precedent) if the
  worker is exposed as a connector -- later.

## Verification

```
python -m pytest tests/test_followup_tool_surface.py -q
```
46 tests pass: every read tool and the full read surface qualify; every known mutating
tool disqualifies and is flagged mutating; mixed read+send disqualifies and names the
send tool; unknown tools, case variants, blanks, and non-string entries fail closed; an
empty surface is trivially qualified; the allowlist is disjoint from known mutating tools.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/schemas/followup_tool_surface.py` | 175 |
| `plans/PR-Followup-Tool-Surface-Qualification.md` | 108 |
| `tests/test_followup_tool_surface.py` | 97 |
| **Total** | **380** |
