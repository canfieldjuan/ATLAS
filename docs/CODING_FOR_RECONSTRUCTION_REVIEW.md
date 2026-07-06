# Coding For Reconstruction Review

These are builder rules. They describe how to write code when the PR will be
reviewed by reconstructing the diff independently instead of trusting the PR
description.

The goal is not to game review. The goal is to make the code, tests, plan, and
PR body tell the same truth.

## Builder Rules

1. **Derive the contract from the problem before coding.** Before any code,
   write a **Problem-derived contract** in the plan. It is the standard the
   code will be held to, not a description of code you are about to write. It
   must state:
   - the root cause, not the symptom -- what is actually wrong, and why;
   - what a correct fix must touch and change to reach that cause; and
   - what must not change -- the modules, behaviors, product shape, contracts,
     and adjacent lanes the work does not depend on and must leave alone.

2. **Build only to the contract.** Nothing in the contract may be left
   unimplemented, and nothing outside it may be added. If implementation proves
   the contract wrong, update the contract first, then continue.

3. **Make the diff explain itself.** A reviewer reading only the diff should be
   able to tell what changed. Prefer clear names, direct control flow, and tests
   that exercise the real path over clever glue that needs PR-body narration.

4. **Keep scope visible.** Every touched file must map to the stated problem or
   to verification for that problem. If the work uncovers another behavior
   change, either add the shipped behavior to Scope/Mechanism and the PR body,
   or split it out and name only the follow-up work in Deferred.

5. **Fix the upstream cause.** Do not patch the child symptom just because it is
   the line the reviewer or CI noticed. If the root is broader than this slice,
   fix the correct bounded layer and name the remaining upstream work.

6. **Test the behavior, not the story.** Add the happy path and the edge case
   that would expose the bug or contract drift. Use real local adapters when
   they exist; mock only external boundaries such as third-party network APIs,
   wall-clock time, randomness, or paid external services.

7. **Make the PR description a receipt.** The PR body describes exactly what the
   diff does and what the verification proves. Do not claim a full fix when the
   slice only adds a prerequisite, guard, doc rule, or partial path.

8. **Run the cold diff reconstruction before push.** Read your own diff as if
   you did not write it. Cite `file:line` and report every gap before any
   summary:
   - what each change actually does;
   - which Problem-derived contract requirement each change traces to;
   - any contract requirement missing from the diff;
   - any module/behavior touched that the contract said must not change; and
   - any change that does not trace to the contract.

   Do not declare done while any gap stands.

## Self-Check Template

Record this in the PR body under `## Cold diff reconstruction` before opening
or updating a PR:

```text
Problem:
What is the root cause?

Problem-derived contract:
- Correct fix must touch/change:
- Must not change:

Diff:
What did the diff actually change, cited file:line?

Contract match:
Which contract item does each change satisfy?

Gaps:
What is untraced, missing, or forbidden? Use "None" only after checking.
```
