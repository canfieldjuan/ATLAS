# PR-Guard-Class-Closure

## Why this slice exists

The operator identified a recurring failure mode across the Resolution Audit
S6A structured-privacy arc: on a guard over an open input space, each review
round fixed exactly the string(s) the reviewer/bot reported and shipped, while
the *class* the input belonged to stayed open, so the next input in that class
was reported the next round. Forensic comparison across round heads confirmed
the later-round bugs were pre-existing and behavior-identical across rounds --
not regressions churned in and out, but one latent class being closed one thin
slice per round (the guard ran 9+ rounds this way). The compounding symptom is
additive-branch growth: each round adds a branch whose fall-through is the next
round's hole, so complexity grows while the guarantee does not.

The repo already has the adjacent rules -- `AGENTS.md` §3k (root cause, not
symptom), §3j (class fixes need unseen probes), and `docs/REVIEWER_RULES.md`
(boundary-probe before LGTM). None of them codify the *structural* requirement
that closes the class: a fail-closed choke point plus a grammar-derived property
test. This slice codifies that so the discipline reaches every builder/reviewer
session through the repo contract, not one PR's review thread.

### Problem-derived contract

- Root cause: the existing contract requires "root cause not symptom" and
  "test 5-10 unseen cases," but neither forces the two things that actually
  close an open-input guard class -- (1) a single fail-closed decision point
  (admit only on affirmative recognition) instead of per-input branches with
  unsafe fall-throughs, and (2) a generative property test over the grammar
  instead of a fixture list. Without those codified, a string-scoped fix passes
  every existing gate and the round loop continues.
- Correct fix must touch/change: add one canonical doc defining the three
  requirements (choke point, class-closure, grammar-derived property test) and
  its trigger and reviewer bar; add a short `AGENTS.md` subsection under the
  root-cause gate pointing to it and naming it the open-input form of root
  cause; add a reference in the `docs/REVIEWER_RULES.md` guard section so the
  reviewer bar cites it.
- Must not change: no code, no product/runtime behavior, no existing rule text
  beyond additive cross-references; the new rule composes with §3j/§3k and the
  boundary-probe rule rather than replacing them.

## Scope (this PR)

Ownership lane: process/review-discipline
Slice phase: Workflow/process

Max files: 6

1. Add `docs/GUARD_CLASS_CLOSURE.md` -- the canonical rule: trigger, the failure
   mode, the three mandatory requirements, the reviewer bar, and the
   relationship to the review-round cap.
2. Add `AGENTS.md` §3k.1 -- the open-input-guard form of the root-cause gate,
   summarizing the three requirements and pointing to the doc; note it
   strengthens §3j from "5-10 unseen cases" to "the generated class."
3. Add an open-input-guard paragraph to the boundary-probe section of
   `docs/REVIEWER_RULES.md` so the enforced reviewer bar cites the doc.

### Review Contract

- Acceptance criteria:
  - [ ] `docs/GUARD_CLASS_CLOSURE.md` states the trigger (open input space),
        the three mandatory requirements (fail-closed choke point;
        class-closure not string-closure; grammar-derived property test), the
        reviewer bar, and the round-cap relationship.
  - [ ] Requirement 1 carries the **scope caveat**: the choke point governs the
        safety verdict, not every field's text; families with a documented
        neutral/data-column admit policy (e.g. #2060 access/audience, type/kind,
        publication dates) keep it. The ban is on an open unsafe default, not on
        any deliberate admit.
  - [ ] Requirement 3 has **two layers**: representation parity AND a
        spec-derived semantic oracle. It states explicitly that parity over a
        semantically-wrong base stays green, so parity alone does not satisfy
        the gate.
  - [ ] `AGENTS.md` §3k.1 exists, sits under §3k and before §3l, summarizes the
        three requirements (incl. the semantic-oracle point), and links both
        `docs/GUARD_CLASS_CLOSURE.md` and `docs/REVIEWER_RULES.md`.
  - [ ] `docs/REVIEWER_RULES.md` boundary-probe section references the doc and
        states that a string-scoped fix with string-scoped fixtures is an
        automatic "needs the class fix."
  - [ ] `docs/SESSION_BOOTSTRAP.md` "fix the class" bullet points restarted
        builders at the stronger open-input-guard gate.
  - [ ] The cap-exception wording matches `docs/OVERNIGHT_ARC_WORKFLOW.md`
        (money/auth/PII), so the two docs do not drift.
  - [ ] All cross-references resolve three-way (doc <-> AGENTS <-> REVIEWER_RULES),
        and the new text is additive -- no existing rule is weakened or removed.
  - [ ] The ASCII gate passes: check_ascii_python.sh (which enforces ASCII on
        Python files only) is green; the new standalone doc and the
        REVIEWER_RULES paragraph are ASCII, and the AGENTS.md addition uses that
        file's existing section-sign convention. No code or product/runtime
        files touched.
- Reachability proof: docs-only slice; verification is cross-reference
  consistency (the three files point at each other) and the plan gate, not a
  runtime test.
- Affected surfaces: builder/reviewer contract docs only.
- Risk areas: rule duplication or contradiction with §3j/§3k/boundary-probe;
  mitigated by making the new rule the open-input *specialization* of the
  existing root-cause gate and cross-referencing rather than restating.
- Reviewer rules triggered: R1, R14.

### Files touched

- `AGENTS.md`
- `docs/GUARD_CLASS_CLOSURE.md`
- `docs/REVIEWER_RULES.md`
- `docs/SESSION_BOOTSTRAP.md`
- `plans/PR-Guard-Class-Closure.md`

## Mechanism

`docs/GUARD_CLASS_CLOSURE.md` is the canonical statement. `AGENTS.md` §3k.1 is
the contract hook (builders read AGENTS §3 first), placed under the root-cause
gate because class-closure is the open-input form of root cause: the root cause
of a reported leak/over-scrub on these surfaces is an open default, and the fix
is the choke point. `docs/REVIEWER_RULES.md` gains the enforcement hook in the
guard boundary-probe section so a reviewer walking the triggered rules for a
guard-shaped PR reaches the class-closure bar. The three documents cross-link so
any entry point (builder reading AGENTS, reviewer walking REVIEWER_RULES, or a
direct doc read) reaches the same requirements.

## Intentional

- Docs-only. The rule changes future builder/reviewer behavior; it does not
  retro-touch the S6A guard that motivated it (that guard is another lane's
  open PR and gets the rule applied as a reviewer bar, not a code change here).
- The new rule specializes §3j/§3k rather than replacing them: §3j still
  requires unseen probes; §3k.1 raises the bar to a generated class + choke
  point specifically for open-input guards.
- The round-cap relationship is stated so the rule cannot be read as licensing
  cap-and-waive of confirmed fail-opens on safety/PII surfaces.

## Deferred

- A CI lint that flags guard-shaped diffs lacking a property test is a natural
  follow-up but is out of scope for a docs-codification slice (it needs a
  detector for "guard-shaped" and "property test present"); named here, not
  built.
- Retrofitting existing guards to the choke-point shape is per-guard work owned
  by each guard's lane, not this slice.

Parked hardening: none.

## Verification

- ASCII gate (the check_ascii_python.sh script) -- not applicable, no Python
  files are touched; run for safety.
- Cross-reference check: `docs/GUARD_CLASS_CLOSURE.md`, `AGENTS.md` §3k.1, and
  the `docs/REVIEWER_RULES.md` guard section each name the other two.
- Plan sync (the sync_pr_plan.py script in --check mode) reports the plan in
  sync -- files-touched and diff-size sections match the diff.

## Estimated diff size

| File | LOC |
|---|---:|
| `AGENTS.md` | 33 |
| `docs/GUARD_CLASS_CLOSURE.md` | 129 |
| `docs/REVIEWER_RULES.md` | 13 |
| `docs/SESSION_BOOTSTRAP.md` | 2 |
| `plans/PR-Guard-Class-Closure.md` | 160 |
| **Total** | **337** |
