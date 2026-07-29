# PR-Followup-Draft-Contract

## Why this slice exists

The #2114 local-model qualification proved Qwen3 30B-A3B Instruct 2507 fit for a
DRAFT-ONLY customer-follow-up role (100/105 strict, zero hard-gate failures, zero
send attempts), but conditionally: its only failures copied an untrusted
approval_state of "approved" from the tool result instead of normalizing it. The
finding is that a model can never be the authority for approval state, and that a
deterministic schema validator plus server-owned approval is required before any
draft is used. This slice specifies that contract: the draft-only result shape
plus the fail-closed validator that makes the qualified model safe to use. It is a
contract only -- it does not implement or expose a live send path.

### Problem-derived contract

From the #2114 evidence, a correct fix must:
- Define the draft-only result shape with canonical machine fields the worker
  copies (not infers): success, status, error_code, stages_completed,
  failed_stage, stable customer_id/draft_id, next_permitted_actions.
- Fail closed on the seven qualified failure groups: no results, ambiguous
  customer, permission denied, partial completion, contradictory data, injected
  identifier/instruction, corrupted approval -- each requiring a canonical
  error_code and failed_stage.
- Make the worker unable to be the approval authority: reject any worker-emitted
  approval_state of "approved", and derive real approval from server-side state
  only.
- Forbid any send / side-effect action in the result's next_permitted_actions
  (draft-only; no send tool in the worker's set).
- Be deterministic validation with no live send path and no external side effect.

## Scope (this PR)

Ownership lane: followup-workflow
Slice phase: vertical slice

The draft-only workflow contract + validator. Read-only / dry-run MCP
qualification of the worker's tool surface, and any downstream consumer, are
later slices. No send path is specified or authorized here.

### Review Contract

- Acceptance criteria:
  - [ ] A drafted result validates; success is true iff status is "drafted".
  - [ ] Each of the seven failure groups validates only with a canonical
        error_code and failed_stage, and is rejected without them.
  - [ ] A worker-emitted approval_state of "approved" is rejected (on any status),
        and resolve_approval derives approval from server-side state only and
        never approves a non-drafted result.
  - [ ] A next_permitted_actions list containing any send action is rejected.
  - [ ] An unknown status value is rejected.
- Reachability proof: N/A for a production surface -- contract + validator only,
  no runtime caller and no send path. Proof is the test suite.
- Affected surfaces: one new schema module and its test file; no existing file
  modified; nothing imports the module yet.
- Risk areas: the approval authority boundary (rejected unconditionally +
  server-owned resolve, boundary-probed); the send denylist; canonical-field
  fail-closed on every failure group.
- Reviewer rules triggered: R2, R14.

### Files touched
- `atlas_brain/schemas/followup_workflow.py`
- `tests/test_followup_workflow.py`
- `plans/PR-Followup-Draft-Contract.md`

Max files: 3

## Mechanism

FollowUpDraftResult is a Pydantic v2 model (followup_draft.v1, canonical "schema"
key via serialize_by_alias) whose after-validator enforces the fail-closed rules:
success matches a "drafted" status; a non-drafted status requires error_code and
failed_stage; a worker-emitted approval_state of "approved" is rejected as the
corrupted-approval failure; and next_permitted_actions may not intersect a send
denylist. resolve_approval returns approval derived only from a server_approved
argument and only for a drafted result, so the worker's approval claim is never
trusted. validate_followup_draft is the deterministic entry point.

## Intentional

- The worker may not emit "approved" at all (not merely when contradictory): the
  qualified failure was copying an untrusted approval, so approval is server-owned
  and the worker's claim is always rejected.
- Contract + validator only, no send path and no side-effect tool -- the #2114
  bounded next step explicitly does not authorize a live send path.
- status is a closed enum so an invented status is rejected (the harness saw
  models invent failure/failed/error differently across surfaces).

## Deferred

- Read-only / dry-run MCP qualification of the worker's tool surface -- a later
  slice.
- The downstream side-effect service that enforces server-owned approval and a
  token-gated send -- a separate, later slice, out of scope here.
- Wiring the qualified model (Qwen3 30B-A3B Instruct 2507) into a worker -- config
  when a runtime caller exists.

## Verification

```
python -m pytest tests/test_followup_workflow.py -q
```
23 tests pass: a drafted result validates and preserves canonical schema key +
stable ids; each of the seven failure groups validates only with error_code +
failed_stage and is rejected without them; success must match "drafted"; a
worker-emitted "approved" is rejected on both failure and drafted results;
resolve_approval is server-owned and never approves a non-drafted result; a send
action in next_permitted_actions is rejected; an unknown status is rejected.

## Estimated diff size

| File | Lines |
|---|---|
| atlas_brain/schemas/followup_workflow.py | 193 |
| tests/test_followup_workflow.py | 260 |
| plans/PR-Followup-Draft-Contract.md | 122 |
| **Total** | **575** |

Over the 400 soft cap after the review-driven guard closures (canonical codes,
closed stage enum, action allowlist, normalized approval, required ids) and their
tests, on a safety-critical contract. Carried by the Diff-budget override line in
the PR body.
