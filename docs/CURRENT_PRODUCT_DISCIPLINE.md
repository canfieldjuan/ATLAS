# Current Product Discipline

**Status:** Active operating discipline for Atlas builder sessions and Codex connector review.
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
- resolves a reviewer BLOCKER **that independently meets one of the three
  above**.

Everything else goes to `HARDENING.md` or a GitHub issue with owner/context,
risk, effort, and the trigger that would promote it.

**A severity badge is not a qualification.** Assess blast radius against the
severity rubric yourself and state the concrete failure path, or downgrade. An
automated reviewer that files every finding as P1/BLOCKER otherwise turns the
fourth clause into "all hardening is inline," and the rule is swallowed by the
one clause that delegates its own decision. Observed on #2216: nine review
rounds of bot-badged P1s, each individually real, each qualifying under an
unqualified fourth clause; the PR grew from +1213/-99 to +3409/-172 during
review on a core that was proven by round four.

**State the parking predicate at plan time.** Every plan says, before any
finding arrives, which class of finding the slice parks by default -- for
example "races narrower than a single request, and findings whose blast radius
is one recoverable record, are parked." `Parked hardening: none` is then a claim
earned against a stated predicate rather than a silent default.

Without one, the set of findings the slice owns is an empty set with no default
for members discovered later: every new finding is in scope by construction, and
there is no state in which the slice is finishable. That is the scope-level form
of the closure declaration in `docs/GUARD_CLASS_CLOSURE.md` -- same defect, one
level up from code.

**A hardening fix that introduces new mechanism is never inline.** If closing a
parked-class finding needs a new table, migration, subsystem, or dependency, it
is a separate slice by definition: the new mechanism carries its own surface,
and that surface generates the next finding inside the same review loop. Park it
and link the follow-up. (#2216 added a migration and a delivery-receipts
subsystem at round 8 in response to a hardening finding; the next bot poll found
a P1 inside the new mechanism.)

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
