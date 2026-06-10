# PR-Review-Rule-Fix-Class-Not-Example

## Why this slice exists

The same review failure pattern has now repeated across two quality lanes:
blog-prose/content-quality fixes and raw-ticket clustering recall. In both
cases, the implementation first fixed the reviewer's cited example instead of
the defect class, and only generalized after another reviewer probe. That makes
the process gap explicit: the review contract needs to make hardcoding the
example unable to pass.

This slice turns that recurrence into review mechanism rather than another
one-off reminder. It strengthens the test-evidence gate against trivial
happy-path-only proof, adds R13 to the reviewer rules, front-loads the builder
self-probe requirement, and logs the pattern in `REVIEW_MISSES.md` as the first
real flywheel entry.

## Scope (this PR)

Ownership lane: dev-workflow/review-contract
Slice phase: Workflow/process

1. Strengthen R2/AGENTS/bootstrap so meaningful logic changes need non-trivial,
   realistic coverage beyond the happy path.
2. Add R13, "fix the class, not the example," to `docs/REVIEWER_RULES.md`.
3. Add builder self-probe discipline for class fixes to `AGENTS.md` and the
   fresh-session bootstrap.
4. Add the repeated fix-the-cited-example pattern to `REVIEW_MISSES.md`.

### Review Contract

- Acceptance criteria:
  - [ ] Reviewer rules include R13 and define automatic failure for
        hardcoding reviewer-cited strings, reusing only reviewer examples as
        tests, or failing a held-out/unseen probe.
  - [ ] R2 rejects trivial happy-path-only tests for meaningful logic changes
        when realistic negative/edge/malformed/sparse/varied-input cases are
        left unexercised.
  - [ ] Builder workflow requires 5-10 unseen same-class probes before claiming
        a class fix complete, with explicit disclosure if only the cited example
        was tested.
  - [ ] Generated/unseen class-fix probes must be diverse enough to exercise
        the class, not trivial near-duplicates.
  - [ ] Reviewer guidance tells reviewers to frame class defects with an
        example plus held-out probe discipline, not a single target example.
  - [ ] Review miss ledger records the recurring blog-prose and clustering
        "fix the cited example" pattern and names R13/bootstrap as the durable
        gate.
- Affected surfaces: developer workflow docs, reviewer rule pack, review miss
  ledger.
- Risk areas: process drift, review discipline, PR contract clarity.
- Reviewer rules triggered: R1, R10.

### Files touched

- `AGENTS.md`
- `REVIEW_MISSES.md`
- `docs/REVIEWER_RULES.md`
- `docs/SESSION_BOOTSTRAP.md`
- `plans/PR-Review-Rule-Fix-Class-Not-Example.md`

## Mechanism

R2 is tightened so test evidence must exercise realistic non-happy-path
behavior for meaningful logic changes. R13 is added as a reviewer rule after
R12 so future review findings can cite a named fail condition instead of
relying on ad hoc judgment. The rule requires class-level fixes to include
held-out or generated/parametrized probes and declares hardcoded reviewer
examples insufficient.

The builder workflow and bootstrap add the reciprocal self-check: when a review
finding is about a defect class, the builder must generate 5-10 diverse unseen
same-class cases the reviewer did not mention, verify them, and include that
proof before claiming the fix complete.

`REVIEW_MISSES.md` gets the first real row so the pattern is visible in the
reviewer flywheel.

## Intentional

- This is documentation/process only. It does not add a mechanical audit yet
  because detecting "hardcoded the reviewer's example" generally needs
  reviewer-held probes or property-style tests, not simple static analysis.
- The rule requires property/parametrized or held-out probes where possible,
  but leaves exact mechanics to the slice because some domains cannot generate
  valid randomized cases safely. It still requires those probes to be diverse
  enough to exercise the class.

## Deferred

- Mechanical support for R13, if useful later: templates/checkers that nudge
  reviewers to record held-out probes or property-test evidence in the verdict.

Parked hardening: none.

## Verification

- Command: python scripts/sync_pr_plan.py plans/PR-Review-Rule-Fix-Class-Not-Example.md
  - passed; plan files and diff-size table updated from the real diff.
- Command: rg -n "R13|Fix the class|Class fixes need unseen probes|fixed the cited example|5-10 same-class|happy path|diverse enough" AGENTS.md docs/REVIEWER_RULES.md docs/SESSION_BOOTSTRAP.md REVIEW_MISSES.md plans/PR-Review-Rule-Fix-Class-Not-Example.md
  - passed; R13/self-probe/ledger language is present in the intended files.
- Command: git diff --check
  - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `AGENTS.md` | 35 |
| `REVIEW_MISSES.md` | 1 |
| `docs/REVIEWER_RULES.md` | 38 |
| `docs/SESSION_BOOTSTRAP.md` | 2 |
| `plans/PR-Review-Rule-Fix-Class-Not-Example.md` | 114 |
| **Total** | **190** |
