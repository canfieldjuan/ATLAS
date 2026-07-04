# Current Product Discipline

**Status:** Active operating discipline for Atlas builder/reviewer sessions.
**Last updated:** 2026-07-04

This file prevents workflow drift. It is not the ordered product roadmap,
current-state tracker, or product-level definition of done. Active issues,
accepted plan docs, and explicit operator decisions own specific product
behavior.

## Source Of Truth Order

Use this order when deciding what to build:

1. Explicit operator instruction in the current session.
2. The active GitHub issue and accepted plan doc for the current slice.
3. This product discipline and `AGENTS.md`.
4. `CANONICAL.md` and `INTEGRATION_MAP.md` for current wiring, plus
   `CONTEXT.md` for historical debt/session notes only.

`BUILD_SPEC.md` is deprecated historical context and must not be used as the
current product roadmap or definition of done.

## Current Operating Rule

Build vertical, end-to-end product slices first. A product slice is not done
until it proves a real buyer-visible or operator-visible path through the real
entrypoint and an observable output, state change, artifact, job, or gate.

Prefer:

- the thinnest useful path that can be run;
- real adapters and real producers where practical;
- focused tests/smokes that prove the visible behavior is wired;
- parking non-blocking hardening instead of widening the slice.

Do not turn setup, harnesses, watchers, audits, maturity gates, or process
polish into the main work unless it unblocks the current vertical proof, fixes
a real safety/security/privacy/money risk, or is justified by a recent vertical
slice that failed because that infrastructure gap existed. The plan must name
the blocker, risk, or failed product run that justifies the workflow/process
slice.

## Not The Product Roadmap

The ordered product spine, current state, and product-level definition of done
are intentionally not created here. Defining them is a product-planning slice
that requires operator approval, because it chooses what "done" means across
customer-facing surfaces. Until that slice lands, use the active GitHub issue
and accepted plan doc to decide the current product path.

## Hardening Parking Rule

Inline hardening is allowed only when it:

- blocks the current vertical proof;
- fixes a real safety/security/privacy/money risk, including standalone
  production-hardening slices where that risk is the reason for the slice;
- prevents the slice's output from being false or misleading;
- resolves a reviewer BLOCKER.

Everything else goes to `HARDENING.md` or a GitHub issue with owner/context,
risk, effort, and the trigger that would promote it.

## Product Shape Consent Gate

Do not change user-facing product shape without explicit operator consent in the
current session or accepted issue/plan.

This includes:

- report, snapshot, email, or PDF structure;
- landing-page positioning, claims, or copy;
- pricing, checkout, subscription, and entitlement surfaces;
- buyer-visible tables, cards, sections, labels, or promises;
- customer-facing output semantics: what a user sees, buys, receives, or
  believes the product does.

If implementation reveals a product-shape decision, stop changing that surface.
Document the decision needed in the plan's `Deferred` section or in
`HARDENING.md`, and continue only on the technical path that does not decide the
product shape.
