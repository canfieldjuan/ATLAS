# PR-Contract-Dispositionable-Criteria

## Why this slice exists

#2202 bounded the reviewer's completion matrix. But item 1 of that matrix is
"each acceptance criterion in the Review Contract", and current `main` still
teaches builders to author criteria without a code claim or settling evidence:
the canonical example says only "Behavior A works" / "Edge case B handled", and
the mandatory scaffold says only "List the outcomes the reviewer checks one by
one." A plan can satisfy the shape checks while giving the reviewer no concrete
claim to disposition. The stopping rule is only ever as bounded as the contract
it checks against.

### Why this phase, and why it is admissible now

`Workflow/process` is the phase by definition -- this changes review contracts
and the plan-authoring tool, not product behavior (`AGENTS.md` phase table).

Section 0 admits a workflow slice only when it unblocks the current vertical
proof, fixes a real safety/security/privacy/money risk, or is justified by a
recent product slice that failed because the infrastructure gap existed. This is
merge-gate integrity hardening for the safety/security/concurrency review path:
`docs/REVIEWER_RULES.md` and `scripts/new_pr_plan.sh` are the authoring surfaces
for the gate contract, and on current `main` they permit newly authored
acceptance criteria with no code claim or evidence target. That lets a future
security/concurrency claim reach review in a state the reviewer can only block
or over-broaden. This PR closes that authoring gap at the source.

### Problem-derived contract

- Root cause: nothing requires a Review Contract's acceptance criteria to name
  something the reviewer can look at, so a criterion can name a bare hazard --
  an unbounded review mandate the builder authored for themselves, which the
  reviewer is right to honor.
- Correct fix must: require each acceptance criterion to name a code claim or
  the evidence that settles it; keep behavioral and risk-shaped-but-evidenced
  criteria first-class (1a's reachability proof depends on the former); leave
  risk areas alone, since they name hazards by design and the matrix does not
  disposition them; route newly authored bare hazards back to contract authoring
  instead of asking reviewers to hunt open-ended categories; require
  open-input criteria to name the 3k.3 evidence-gated mechanism instead of
  treating one sampled fixture list as settling evidence; require
  concurrency/open-execution criteria to name the 3k.4 execution model and
  property-level invariant instead of treating one sampled concurrent test as
  settling evidence; and reach every surface a builder authors a contract from
  -- the rule text, the canonical example, and the scaffold that generates plans.
- Must not change: #2202's completion vocabulary (criteria are met / not met /
  could-not-determine; rules are pass / fail / not-verified / n-a), the
  complete-but-blocked state, `success` semantics, or legacy contracts'
  existing disposition behavior.

## Scope (this PR)

Ownership lane: reviewer-contract/plan-authoring
Slice phase: Workflow/process
Max files: 5

1. `AGENTS.md` 1a: an acceptance criterion names a code claim or the evidence
   that settles it, never a bare risk category; behavioral criteria explicitly
   required where a reachability proof is asked for; risk areas exempt; open
   categories route to 3k.3 / 3k.4; newly authored bare hazards fail contract
   authoring instead of sending reviewers to hunt the category.
2. `docs/REVIEWER_RULES.md`: canonical Review Contract example shows settleable
   criteria with its risk-area line unchanged; the reviewer-side note uses the
   criteria vocabulary and states the complete-but-blocked outcome.
3. `scripts/new_pr_plan.sh`: the generated scaffold prompts for the same.

### Review Contract

- Acceptance criteria:
  1. Running `scripts/new_pr_plan.sh` emits an acceptance-criteria prompt that
     requires a code claim or settling evidence and forbids a bare risk category --
     shown by running it and reading the emitted block.
  2. The canonical example in `docs/REVIEWER_RULES.md` contains no criterion
     phrased as a bare risk category, and its risk-area line is unchanged from
     main (risk areas are exempt).
  3. `AGENTS.md` 1a states that a behavioral criterion settled by a command,
     CI job, or generated artifact is admissible.
  4. No text in this diff retroactively changes legacy matrix outcomes:
     reviewers keep investigating legacy contracts against the contract as
     authored, and only the authoring finding is grandfathered to an advisory
     NIT for contracts authored before landing.
  5. Concurrency/open-execution criteria are not declared settled by one sampled
     concurrent test; the rule text, reviewer pack, and emitted scaffold require
     the 3k.4 execution model and property-level invariant.
  6. Open-input criteria are not declared settled by one sampled fixture list; the
     rule text, reviewer pack, and emitted scaffold require the 3k.3
     evidence-gated mechanism.
  7. The scaffold regression tests assert persistent canonical surfaces
     (`AGENTS.md`, `docs/REVIEWER_RULES.md`, and a freshly emitted plan), not
     this in-flight plan path that teardown will archive after merge.
  8. `scripts/audit_plan_code_consistency.py` resolves every path and function
     claim in this plan.
- Reachability proof: `scripts/new_pr_plan.sh` is the real entrypoint builders
  invoke to author a plan; the emitted plan file is the observable
  artifact, and its Review Contract block carries the new prompts.
- Affected surfaces: the plan-authoring scaffold and the two documents that
  define contract shape. No runtime, API, DB, or product surface.
- Risk areas:
  - Over-rejection of legitimate criteria (universally-quantified but reviewable
    claims, behavioral criteria) -> closed by criterion 3 and by 1a stating
    explicitly that breadth is not the defect, a missing referent is.
  - Blocking other sessions' in-flight plans -> closed by criterion 4: risk
    areas are exempt entirely, and legacy contracts keep their existing
    disposition behavior, so a legacy plan is not newly invalid.
  - A surface left un-propagated -> closed by criteria 1 and 2.
  - A concurrency example overclaims from one sampled test -> closed by
    criterion 5, which rejects the sampled-test-only form and requires the
    3k.4 model/invariant surface instead.
  - An open-input example overclaims from a sampled fixture list -> closed by
    criterion 6, which rejects the fixture-list-only form and requires the 3k.3
    evidence-gated mechanism instead.
  - A regression test depends on the root in-flight plan file -> closed by
    criterion 7, which generates a throwaway plan and asserts only persistent
    canonical surfaces.
- Reviewer rules triggered: R1, R2, R10, R13, R14. (R13: this slice answers
  class-level findings -- 'update every contract-authoring surface' -- so the
  fix must be class-wide, not the cited example. R2: it changes a generator,
  so the generated text needs assertions.)

### Files touched

- `AGENTS.md`
- `docs/REVIEWER_RULES.md`
- `plans/PR-Contract-Dispositionable-Criteria.md`
- `scripts/new_pr_plan.sh`
- `tests/test_new_pr_plan.py`

## Mechanism

One authoring test replaces the previous framing for new and materially revised
contracts: does the criterion name a code claim or the evidence that settles it?
A `file:line`, a command and its output, or a CI job all qualify, which keeps
behavioral criteria admissible. A bare hazard qualifies for none of them.

The earlier draft used "structural property" as the test, which excluded exactly
the runtime criteria 1a requires for a reachability proof. Boundedness is the
property that was actually wanted; structure was a proxy that over-rejected.

## Intentional

- **Behavioral criteria are explicitly admissible.** The first draft of this PR
  banned them by accident, which would have made 1a's reachability proof
  unauthorable.
- **Vocabulary follows #2202 exactly**: criteria are met / not met /
  could-not-determine; `not-verified` is the rule vocabulary and is not used for
  criteria here.
- **Complete-but-blocked is preserved.** An unsettleable criterion does not make
  a review incomplete; it makes it complete and not approved.
- **Legacy disposition behavior is preserved.** A pre-existing contract is still
  reviewed against the contract as authored; this PR grandfathers only the
  authoring finding, not the reviewer's ability to investigate legacy evidence.
- **The scaffold is in scope on purpose.** A rule the mandatory helper generates
  violations of is a rule builders break by construction.

## Deferred

- Mechanizing a settleability check. Judgment at plan review is the right start;
  if it earns automation it belongs in a `scripts/` checker with fixture tests
  rather than as prose (the #2197 lesson).

Parked hardening: none.

## Verification

    bash -n scripts/new_pr_plan.sh
    # -> syntax OK; emitted Review Contract block read directly and confirmed
    #    to carry both new prompts

    python scripts/audit_plan_code_consistency.py \
        plans/PR-Contract-Dispositionable-Criteria.md

    python -m pytest tests/test_new_pr_plan.py -q
    # -> 16 passed

Failure detection proven per 3i, not assumed: with both new scaffold prompts
reverted to their prior one-line forms, the suite FAILS at
`tests/test_new_pr_plan.py:80`; restored, 16 pass. So an edit that deletes the
settleability prompts cannot leave the suite green.

- `git diff --check` clean; 0 non-ASCII characters added.
- Path-trigger table parses unchanged (12 glob rows, 9 prose rows).
- `bash scripts/pre_push_audit.sh --repo-root "$PWD" --script-root "$PWD" --pr-author canfieldjuan`
  passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `AGENTS.md` | 57 |
| `docs/REVIEWER_RULES.md` | 48 |
| `plans/PR-Contract-Dispositionable-Criteria.md` | 195 |
| `scripts/new_pr_plan.sh` | 15 |
| `tests/test_new_pr_plan.py` | 67 |
| **Total** | **382** |
