# PR Reconstruction Protocol

Every PR review **reconstructs the PR from the diff**. Never review a PR against
its description. This binds builder self-review and Codex connector review:
the description, commit message, and title are unverified claims; the code is
ground truth. Codex applies this protocol in scoped form: changed code, direct
callers/tests/artifacts, required CI, and the PR's Review Contract.

## Why

Reviewing off the description reproduces the author's framing and misses where
the diff diverges from a correct fix. The description, commit message, and title
are unverified claims; the code is ground truth.

Exemplar (Atlas #1999, 2026-07-04): a review posted a `BLOCKER` plus "six
blocking findings" read off the PR description and stale bot thread-titles. The
head code had already resolved every one of them. Reconstructing from the diff
caught the error before it stuck.

## The Protocol

Build two independent reconstructions, then compare them. They are independent,
not sequential: derive the correct-fix reconstruction from the problem before
or separately from the diff so the diff cannot anchor it. The reporting order
below is for the final review write-up, not the investigation order.

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

4. **Cite checkable evidence for every claim.** For code or content claims,
   cite `file:line`. For non-file evidence, cite the named artifact: command
   output, CI run/job, generated artifact, PR metadata, or another inspectable
   source. Sort each finding into **confirmed / contradicted /
   could-not-determine**. Never mark a finding confirmed without checkable
   evidence. **Lead with the gaps, not a summary.** Post findings inline on the
   changed line when the evidence lives in the diff; use the review body for
   out-of-diff lines and non-file evidence.

## Composition

- Run this before, and feed it into, the §4a scoped-verification pass and the
  triggered `docs/REVIEWER_RULES.md` rules (R1-R14 -> BLOCKER / MAJOR / NIT).
- Apply the vertical-progress lens in the same pass: is the slice a vertical
  MVP step, or harness/hardening/polish drift that defers the core?
- This is the strict-review form of "no evidence lifted from prose": no verdict
  claim survives without checkable evidence behind it.
