# PR-Builder-Convergence-Discipline

## Why this slice exists

Resolution Audit S6C (#2076) ran ~9 review rounds and ~35 findings on one
transcript sanitizer without converging: each push fixed the cited senders and
the next push reported new same-class senders, and every miss dropped a customer
question. The operator asked whether this is a codifiable pattern. It is: the
builder closed each cited example with the narrowest local patch and never
abstracted to the generating decision, and it tried to enumerate an open
semantic category ("is this token a sender?") that no denylist or allowlist can
close. The existing guard docs cover closed/enumerable classes; they do not yet
name the open-category case, the asymmetric-cost default, or a builder-side
non-convergence circuit-breaker. This slice codifies those three so the next
occurrence is caught by rule instead of by nine rounds of review.

### Problem-derived contract

- Root cause: the guard/class-closure discipline assumes the recognizer's class
  is enumerable (a grammar over a bounded vocabulary). When the decision instead
  rests on membership in an open semantic category, both the denylist and the
  member-allowlist are unbounded, and a property test over members cannot
  converge. There is no rule telling the builder to (a) evidence-gate instead of
  enumerate, (b) bias the default by error-cost asymmetry, or (c) stop
  instance-patching once threads stop trending to zero.
- Correct fix must touch/change: extend `docs/GUARD_CLASS_CLOSURE.md` with the
  open-category evidence-gated closure and the asymmetric-cost default; add the
  convergence circuit-breaker plus a Decision-Seam Analysis requirement to
  `AGENTS.md`; add the open-category exception to `docs/REVIEWER_RULES.md` R13.
  Each cites the #2076 arc as the worked example, matching the existing
  episode-cited rule style.
- Must not change: any runtime, product, or test code; the existing three
  requirements in `GUARD_CLASS_CLOSURE.md`; the bot-round noise cap semantics;
  rule numbering R1-R14; or any adjacent lane. Docs-only.

## Scope (this PR)

Ownership lane: workflow/review-discipline
Slice phase: Workflow/process

Max files: 6

1. Add "When the recognizer itself is open (evidence-gated closure)" plus the
   asymmetric-cost default and the evidence-keyed oracle refinement to
   `docs/GUARD_CLASS_CLOSURE.md`.
2. Add `AGENTS.md` 3k.2 convergence circuit-breaker + Decision-Seam Analysis.
3. Add the open-category exception to `docs/REVIEWER_RULES.md` R13.
4. Review-driven root-cause widening (+1 file): the exception and the 3k.2
   cap-reconciliation must land in EVERY canonical statement, not only the ones
   first edited, or a doc that restates the un-reconciled bar (or the OVERNIGHT
   runbook, which declares itself canonical / wins-on-conflict) reopens the
   conflict. The complete enumerable set -- `GUARD_CLASS_CLOSURE.md`,
   `REVIEWER_RULES.md` (R13 + the guard-LGTM gate), `AGENTS.md` (3k.1 + 3k.2),
   and `docs/OVERNIGHT_ARC_WORKFLOW.md` -- is reconciled in one pass so no
   statement is left un-reconciled.
5. Single-source structure (the convergent fix for the three-round drift):
   designate `docs/GUARD_CLASS_CLOSURE.md` the sole canonical source for the
   guard-closure bar, the open-category exception, the asymmetric-safe default,
   and the fail-open block-exception set (money/auth/PII/safety); replace the
   parallel re-listings in `docs/REVIEWER_RULES.md`, `AGENTS.md` 3k.1,
   `docs/OVERNIGHT_ARC_WORKFLOW.md`, and (review-surfaced, +1 file)
   `docs/SESSION_BOOTSTRAP.md` (which new builders read first) with short
   pointers, so no secondary doc
   restates the normative content and none can drift from the source. This
   removes ~37 lines of restatement; the requirements themselves are unchanged in
   the canonical doc.

### Review Contract

- Acceptance criteria:
  - [ ] The open-category section distinguishes an enumerable class (existing
        requirement 3) from an open semantic category, and prescribes
        evidence-gating with an asymmetric-safe default, citing #2076.
  - [ ] The asymmetric-cost rule states the default falls to the cheap-error
        side and only the expensive-error class must be closed by construction
        (bar: <= status-quo in the expensive direction).
  - [ ] The oracle refinement bans a `product()` over category members as a
        fixture matrix and requires evidence-keyed generation.
  - [ ] `AGENTS.md` 3k.2 defines the non-convergence trip (flat/rising same-class
        threads over 3 pushes on one decision), forbids the next instance patch,
        and requires a Decision-Seam Analysis, distinguished from the bot-noise
        cap.
  - [ ] `REVIEWER_RULES.md` R13 gains the open-category exception with a block
        condition pointing at the 3k.2 circuit-breaker.
  - [ ] Single source: `GUARD_CLASS_CLOSURE.md` is the sole canonical statement
        of the bar, the open-category exception, the asymmetric-safe default, and
        the block-exception set; its "single source" note states the pointer set
        as "every such doc," not an enumerated list.
  - [ ] No un-pointered restatement remains: the block-set / bar literal appears
        only in the canonical doc (exhaustive repo grep), and no numbered
        requirement re-listing survives in `REVIEWER_RULES.md`, `AGENTS.md` 3k.1,
        `docs/OVERNIGHT_ARC_WORKFLOW.md`, or `docs/SESSION_BOOTSTRAP.md`; the
        cap-relationship paragraph references the grammar- or evidence-derived bar.
  - [ ] No runtime/product/test code changes; existing requirements, rule
        numbering, and the noise-cap semantics are untouched.
- Reachability proof: N/A -- documentation/process rules; verified by rendered
  cross-references resolving to real sections and ASCII/consumer gates passing.
- Affected surfaces: docs / reviewer-process only.
- Risk areas: duplication with existing rules (mitigated by cross-reference, not
  restatement); rule-numbering drift (avoided -- no renumbering).
- Reviewer rules triggered: R1, R14.

### Files touched

- `AGENTS.md`
- `docs/GUARD_CLASS_CLOSURE.md`
- `docs/OVERNIGHT_ARC_WORKFLOW.md`
- `docs/REVIEWER_RULES.md`
- `docs/SESSION_BOOTSTRAP.md`
- `plans/PR-Builder-Convergence-Discipline.md`

## Mechanism

Three targeted additions, each cross-referencing rather than restating the
existing rules. `GUARD_CLASS_CLOSURE.md` gains a section for the case its three
requirements do not cover -- when the recognizer's category is open on both
sides, so neither list nor a member-property-test closes it -- prescribing
evidence-gated recognition (act only on bounded structural evidence; default
ambiguous to the asymmetric-safe side), the asymmetric-cost default, and an
evidence-keyed (not member-keyed) oracle. `AGENTS.md` gains 3k.2: a builder-side
circuit-breaker that trips when same-class findings on one decision fail to
trend toward zero over three pushes, forbidding the next example patch and
requiring a Decision-Seam Analysis (name the shared decision, say why it is
wrong, then fix the seam / waive the bounded residual / re-scope). `REVIEWER_RULES.md`
R13 gains the open-category exception so reviewers block a member-patch response
to a non-converging open class. The #2076 arc is the cited episode throughout.

Single source: after review showed the same normative content restated in four
docs kept drifting (the open-category exception missing in one gate; the block
set written two different ways), `GUARD_CLASS_CLOSURE.md` is made the sole
canonical source for the bar, the open-category exception, the asymmetric-safe
default, and the block-exception set, and it opens with a "single source" note.
`REVIEWER_RULES.md`'s guard-LGTM gate, `AGENTS.md` 3k.1, and the
`OVERNIGHT_ARC_WORKFLOW.md` cap now carry short pointers to it instead of copies,
so the bar or the set changes in exactly one file. 3k.2 (the convergence
circuit-breaker) is the second canonical statement; the OVERNIGHT cap points to
it for how-you-stop.

## Intentional

- Extend, do not duplicate: the "oracle is not a fixture matrix" and "fix the
  class not the string" lessons already live in `GUARD_CLASS_CLOSURE.md` §3 and
  R13; this slice adds only the open-*category* refinement and cross-references
  the rest, to avoid a second competing statement of the same rule.
- The circuit-breaker is deliberately builder-side and distinct from the
  reviewer-side bot-noise cap: the noise cap is for re-litigation of a green
  contract; 3k.2 is for real findings that keep reopening a shared decision.
- No CI enforcement in this slice. A thread-per-push non-convergence detector is
  a heavier, separable slice (it needs the review-thread history producer);
  codifying the rule first lets the reviewer enforce it immediately.

## Deferred

- Advisory CI detector for the 3k.2 trip (count new same-region threads per
  pushed SHA; warn when not trending to zero). Separable follow-up; needs the
  thread-history producer.

Parked hardening: none.

## Verification

- `python scripts/sync_pr_plan.py --check plans/PR-Builder-Convergence-Discipline.md origin/main` -- passed ("plan already in sync").
- `python scripts/audit_pr_body.py <body-file>` -- passed ("pr body audit: PASS").
- Non-ASCII scan (`git diff origin/main -- <f> | grep '^+' | grep -P '[^\x00-\x7F]'`) of the added lines in all five edited docs -- `AGENTS.md`, `docs/GUARD_CLASS_CLOSURE.md`, `docs/OVERNIGHT_ARC_WORKFLOW.md`, `docs/REVIEWER_RULES.md`, and `docs/SESSION_BOOTSTRAP.md` -- reports 0 non-ASCII each (additions use `--`, `->`, ASCII quotes, and spell "section 3k.1" rather than the section-sign); markdown is not gated by the Python ASCII check regardless.
- Managed pre-push hook (git diff --check, plan/body sync, reviewer-rule mapping, local review) -- passed on push.
- Docs-only slice: no test suite applies; cross-references resolve to real sections (`GUARD_CLASS_CLOSURE.md`, `AGENTS.md` 3k.2, R13, `OVERNIGHT_ARC_WORKFLOW.md`).
- Single-source proof: the block-set literal now appears in only one file, the canonical `docs/GUARD_CLASS_CLOSURE.md` (its definition plus the drift example); every secondary doc (`docs/REVIEWER_RULES.md`, `AGENTS.md`, `docs/OVERNIGHT_ARC_WORKFLOW.md`) uses the block-exception-set pointer, and no numbered requirement re-listing remains in the guard-LGTM gate or AGENTS 3k.1.

## Estimated diff size

| File | LOC |
|---|---:|
| `AGENTS.md` | 77 |
| `docs/GUARD_CLASS_CLOSURE.md` | 89 |
| `docs/OVERNIGHT_ARC_WORKFLOW.md` | 11 |
| `docs/REVIEWER_RULES.md` | 39 |
| `docs/SESSION_BOOTSTRAP.md` | 2 |
| `plans/PR-Builder-Convergence-Discipline.md` | 179 |
| **Total** | **397** |
