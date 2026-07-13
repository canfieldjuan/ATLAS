# PR-Open-Input-Evidence-Gate-Slice

## Why this slice exists

The Resolution Audit S6 sanitizer arc generated the repo's longest review
threads: #2053 (55), #2076 (55), #2061 (52), #2046 (51), #2054 (31), all one
subsystem. #2037 was the origin -- it did the whole sanitizer in one +1262
change and drew ~24 findings across four open-input classes at once, then
paused. The team already sliced it into S6A/A.1/B/C/E, yet each slice still ran
~50 threads, because each slice was still built by enumeration. #2077 codified
the *closure* bar (3k.1) and the *non-convergence* breaker (3k.2) -- the
after-the-fact tools. This slice adds the missing BEFORE-code gate: reject the
enumerative method on the plan, and bound one open-input class per slice, so the
50-comment storm never starts. The operator asked for this after tracing the
comment explosion to enumeration x surface across #2037 and the S6 series.

### Problem-derived contract

- Root cause: nothing requires an open-input guard PR to commit to an
  evidence-gated method before code, and nothing bounds how many open-input
  classes one PR may touch. So a builder enumerates cases (each unhandled case
  is both a missing line and a review comment) across a wide surface (#2037's
  four classes at once), and the finding count scales as classes x
  cases-per-class. 3k.1/3k.2 only fire after the enumeration exists.
- Correct fix must touch/change: add one `AGENTS.md` 3k section (3k.3) with the
  plan-stage method gate (name the choke point, the safe default, the evidence
  keys; reject an enumerative plan) plus the one-class-per-slice surface bound,
  pointing to `docs/GUARD_CLASS_CLOSURE.md` for the evidence-gate mechanics
  rather than restating them.
- Must not change: any runtime, product, or test code; the 3k.1 bar, 3k.2
  breaker, or `GUARD_CLASS_CLOSURE.md` mechanics (this only adds the plan-stage
  gate + slice bound and points to them); rule numbering; the general
  minimal-code / no-over-build discipline (a separate deferred slice); or any
  adjacent lane. Docs-only.

## Scope (this PR)

Ownership lane: workflow/review-discipline
Slice phase: Workflow/process

Max files: 2

1. Add `AGENTS.md` 3k.3: the plan-stage method gate (primary) + one-class-per-
   slice surface bound for open-input guard/sanitizer/classifier work, citing the
   S6 arc as the receipt and pointing to `docs/GUARD_CLASS_CLOSURE.md` for
   mechanics.

### Review Contract

- Acceptance criteria:
  - [ ] 3k.3 names the plan-stage method gate as PRIMARY: the plan must state the
        choke-point decision, the safe default for ambiguous input, and the
        bounded evidence keys, and an enumerative plan (denylist / case table) is
        rejected at the plan stage.
  - [ ] 3k.3 adds the one-class-per-slice surface bound and states, from the S6
        data, that slicing alone distributes findings while evidence-gating
        collapses a class (primary lever = method, secondary = slice).
  - [ ] 3k.3 points to `docs/GUARD_CLASS_CLOSURE.md` for the evidence-gate
        mechanics and does not restate them (single-source, per the #2077
        lesson).
  - [ ] No runtime/product/test code; 3k.1/3k.2 and the GUARD mechanics are
        unchanged; rule numbering is unchanged (3k.3 is additive after 3k.2).
- Reachability proof: N/A -- a process rule; verified by the cross-reference
  resolving to a real section and the consumer/ASCII gates passing.
- Affected surfaces: docs / reviewer-process only.
- Risk areas: restatement drift (mitigated -- points to GUARD, does not copy);
  scope creep into cross-cutting reconciliation (explicitly avoided -- one file
  plus the plan).
- Reviewer rules triggered: R1, R14.

### Files touched

- `AGENTS.md`
- `plans/PR-Open-Input-Evidence-Gate-Slice.md`

## Mechanism

One additive section, `AGENTS.md` 3k.3, placed after 3k.2 and before 3l. It
defines the trigger (open-input guard/sanitizer/classifier over free text or
producer-supplied structure), the plan-stage method gate (name the choke point +
safe default + evidence keys; reject an enumerative plan before code), and the
one-class-per-slice surface bound. The evidence-gate mechanics stay in
`docs/GUARD_CLASS_CLOSURE.md`; 3k.3 points to them. The "Why" cites the S6 arc
with its thread counts. Deliberately built minimal (one file plus this plan, no
edits to REVIEWER_RULES / OVERNIGHT / GUARD) to avoid the cross-doc restatement
drift that ran #2077 to eight rounds.

## Intentional

- Method gate is PRIMARY, slice is SECONDARY. The S6 data shows slicing alone
  reproduced the series (~24 findings became ~244 across five slices); the lever
  that actually converged a class was evidence-gating (#2076, #2061). So the
  plan-stage method gate leads and the slice bound follows.
- Single home, no restatement: the rule lives only in 3k.3 and points to
  `GUARD_CLASS_CLOSURE.md` for the mechanics. Restating the bar in a second doc
  is exactly what drifted across #2077's rounds.
- No CI/template enforcement in this slice. A plan-doc-template field or a
  reviewer-rule pointer is a separate, heavier follow-up; the reviewer enforces
  3k.3 immediately from the AGENTS contract.

## Deferred

- General minimal-code / no-over-build / single-source-must-never-diverge
  discipline (the auth-module case): its own lighter slice, out of scope here.
- Optional plan-doc-template field or reviewer-rule pointer to 3k.3 (tooling);
  separate follow-up.

Parked hardening: none.

## Verification

- run sync_pr_plan.py, the plan/body consumer gates, and the ASCII scan before
  push.

## Estimated diff size

| File | LOC |
|---|---:|
| `AGENTS.md` | 37 |
| `plans/PR-Open-Input-Evidence-Gate-Slice.md` | 121 |
| **Total** | **158** |
