# PR Reconstruction Protocol

Every PR review **reconstructs the PR independently from the diff**. Never
review a PR against its description. This applies to all reviewers -- human
Claude sessions, Codex, and the CI review action.

## Why

Reviewing off the description reproduces the author's framing and misses where
the diff diverges from a correct fix. The description, commit message, and title
are unverified claims; the code is ground truth.

Exemplar (Atlas #1999, 2026-07-04): a review posted a `BLOCKER` plus "six
blocking findings" read off the PR description and stale bot thread-titles. The
head code had already resolved every one of them. Reconstructing from the diff
caught the error before it stuck.

## The protocol (in order)

1. **Read the diff alone.** State what it actually does, change by change, in
   your own words. Do not read intent off the description, commit message, or
   title -- the code is ground truth, everything else is an unverified claim.

2. **Derive the correct fix from the problem alone.** From the problem the PR
   says it solves, derive what a correct fix would need to touch and change --
   without letting the diff shape the answer.

3. **Compare three things** -- what the diff does, what a correct fix should do,
   and what the description claims -- and report **every** gap between any two:
   - the diff does not match its description;
   - the diff does not match a correct fix (wrong, incomplete, or symptom patch);
   - the diff changes things the description never mentions.

4. **Cite file and line for every claim.** Sort each finding into
   **confirmed / contradicted / could-not-determine**. Never mark a finding
   confirmed without a citation. **Lead with the gaps, not a summary.** Post
   findings inline on the changed line; use the review body (with a `file:line`
   reference) only for out-of-diff lines.

## Composition

- Run this before, and feed it into, the §4a independent-verification pass and
  the `docs/REVIEWER_RULES.md` rule walk (R1-R14 -> BLOCKER / MAJOR / NIT).
- Apply the vertical-progress lens in the same pass: is the slice a vertical
  MVP step, or harness/hardening/polish drift that defers the core?
- This is the strict-review form of "no evidence lifted from prose": no verdict
  claim survives without a checked-out `file:line` behind it.
