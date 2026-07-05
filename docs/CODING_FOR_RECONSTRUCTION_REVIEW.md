# Coding For Reconstruction Review

These are builder rules. They describe how to write code when the PR will be
reviewed by reconstructing the diff independently instead of trusting the PR
description.

The goal is not to game review. The goal is to make the code, tests, plan, and
PR body tell the same truth.

## Builder Rules

1. **Start from the problem, not the patch.** Before coding, write the correct
   fix shape in the plan: what must become true, what surfaces likely need to
   change, what tests should prove it, and what is explicitly out of scope.

2. **Make the diff explain itself.** A reviewer reading only the diff should be
   able to tell what changed. Prefer clear names, direct control flow, and tests
   that exercise the real path over clever glue that needs PR-body narration.

3. **Keep scope visible.** Every touched file must map to the stated problem or
   to verification for that problem. If the work uncovers another behavior
   change, either add the shipped behavior to Scope/Mechanism and the PR body,
   or split it out and name only the follow-up work in Deferred.

4. **Fix the upstream cause.** Do not patch the child symptom just because it is
   the line the reviewer or CI noticed. If the root is broader than this slice,
   fix the correct bounded layer and name the remaining upstream work.

5. **Test the behavior, not the story.** Add the happy path and the edge case
   that would expose the bug or contract drift. Use real local adapters when
   they exist; mock only external boundaries such as third-party network APIs,
   wall-clock time, randomness, or paid external services.

6. **Make the PR description a receipt.** The PR body describes exactly what the
   diff does and what the verification proves. Do not claim a full fix when the
   slice only adds a prerequisite, guard, doc rule, or partial path.

7. **Run the reconstruction self-check before push.** Compare three things:
   what the diff actually does, what a correct fix for the problem should do,
   and what the PR body claims. Any mismatch becomes a code change, a scope
   split, or a clearer Deferred note before the PR goes up.

## Self-Check Template

Use this mentally before opening or updating a PR:

```text
Problem:
What must be true after this slice?

Correct fix shape:
Which files, contracts, tests, and user-visible behavior should change?

Diff:
What did the diff actually change?

Description:
Does the PR body claim exactly that, no more and no less?

Gaps:
What is incomplete, intentionally deferred, or out of scope?
```
