# PR-Vertical-First-Product-Consent

## Why this slice exists

The operator identified a recurring session failure: hardening and autonomy
harness work kept expanding ahead of vertical product proof, and user-facing
product shape could be changed without explicit consent. The prior response
persisted this only in Codex memory, which does not bind every future repo
agent. Root cause: Atlas had no living product spine and still pointed fresh
sessions at stale `BUILD_SPEC.md` as if it were current truth. This fixes the
root by moving the rules into the repo contract and replacing `BUILD_SPEC.md`
as current operating discipline.

This exceeds the 400 LOC target because review found stale roadmap paths still
lived in `CLAUDE.md`, `AUDITOR_PROMPT.md`, `CONTEXT.md`, and the operating-model
doc; leaving those references would make the deprecation self-contradictory.

## Scope (this PR)

Ownership lane: workflow/product-discipline
Slice phase: Workflow/process

1. Add a living `docs/CURRENT_PRODUCT_DISCIPLINE.md` that codifies vertical-first
   delivery, hardening parking, product-shape consent, and source-of-truth
   order.
2. Mark `BUILD_SPEC.md` deprecated so agents stop treating the stale
   voice-to-voice draft as the current product roadmap.
3. Update `AGENTS.md`, `CLAUDE.md`, `AUDITOR_PROMPT.md`, and
   `docs/SESSION_BOOTSTRAP.md` so new/restarted sessions must read the
   discipline doc and apply the consent gate.
4. Neutralize stale `CONTEXT.md` priority language so fresh sessions cannot
   treat old voice-era notes as current roadmap.
5. Update `docs/ai_dev_operating_model.md` so the auditor read path no longer
   names `BUILD_SPEC.md` as priorities/DoD.

### Review Contract

Acceptance criteria:

- Fresh-session docs no longer instruct builders to treat `BUILD_SPEC.md` as
  current truth.
- `AGENTS.md` contains a mandatory product-shape consent gate.
- `AGENTS.md` and the discipline doc make vertical product proof the default
  and require workflow/hardening slices to name the real blocker, risk, or
  failed product run.
- The change does not define new report, landing, pricing, or other customer
  product shape.

Affected surfaces: builder/reviewer process docs only.

Risk areas: overcorrecting into another process arc, or accidentally defining
  product behavior without operator consent.

Triggered reviewer rules: R14 codebase verification.

Reachability proof: local review must see the new docs and plan in sync; text
  checks verify the stale `BUILD_SPEC.md` read path was removed.

### Files touched

- `AGENTS.md`
- `AUDITOR_PROMPT.md`
- `BUILD_SPEC.md`
- `CLAUDE.md`
- `CONTEXT.md`
- `docs/CURRENT_PRODUCT_DISCIPLINE.md`
- `docs/SESSION_BOOTSTRAP.md`
- `docs/ai_dev_operating_model.md`
- `plans/PR-Vertical-First-Product-Consent.md`

## Mechanism

- `docs/CURRENT_PRODUCT_DISCIPLINE.md` becomes the short active orientation doc
  for slice discipline.
- `BUILD_SPEC.md` is intentionally reduced to a deprecation pointer.
- `AGENTS.md` section 0 makes the discipline and product-shape consent gate
  part of the repo contract.
- `docs/SESSION_BOOTSTRAP.md` updates the fresh-session read list and recurring
  mistake checklist so new sessions start from the same rules.
- `CLAUDE.md` and `AUDITOR_PROMPT.md` stop presenting `BUILD_SPEC.md` as live
  roadmap/DoD.
- `CONTEXT.md` is marked historical debt/session-note context, and read lists
  qualify it as non-authoritative for current roadmap/priority/state.
- `docs/ai_dev_operating_model.md` points the enforcement-funnel auditor read
  list at `AGENTS.md` and the discipline doc instead of `BUILD_SPEC.md`.

## Intentional

- This does not create a CI checker. The point is to codify the rule in the
  mandatory docs first, without turning the correction into another harness arc.
- This does not define customer-facing report, landing, pricing, or output
  shape. It only says those decisions require explicit consent.
- This renames the new doc from product spine to product discipline because the
  ordered product spine/current-state/DoD is not safe to invent in a process PR.

## Deferred

- Optional future checker: fail plan docs for workflow/process slices that do
  not name the vertical proof they unblock. Defer until this failure repeats on
  a real product slice.
- Real ordered product spine: define current state, ordered slices, and
  product-level DoD in an operator-approved product-planning slice. This PR
  intentionally does not choose that product shape.

Parked hardening: none.

## Verification

- `python scripts/sync_pr_plan.py plans/PR-Vertical-First-Product-Consent.md origin/main`
- `python scripts/sync_pr_plan.py --check plans/PR-Vertical-First-Product-Consent.md origin/main`
- Targeted text check for discipline/deprecation/vertical-proof/consent wording.
- `ATLAS_SESSION_STATE_FILE=SESSION_STATE.vertical-first-product-consent.local.md bash scripts/local_pr_review.sh --current-pr-body-file /tmp/pr-body-vertical-first-product-consent.md`

## Estimated diff size

| File | LOC |
|---|---:|
| `AGENTS.md` | 43 |
| `AUDITOR_PROMPT.md` | 50 |
| `BUILD_SPEC.md` | 190 |
| `CLAUDE.md` | 34 |
| `CONTEXT.md` | 9 |
| `docs/CURRENT_PRODUCT_DISCIPLINE.md` | 82 |
| `docs/SESSION_BOOTSTRAP.md` | 4 |
| `docs/ai_dev_operating_model.md` | 11 |
| `plans/PR-Vertical-First-Product-Consent.md` | 127 |
| **Total** | **550** |
