# PR-Seam-Convergence-Breaker

## Why this slice exists

`AGENTS.md` 3k.2 ("Convergence circuit-breaker") already defines the failure: when
each push closes the reported review threads but the next push opens a comparable
count of same-class findings on the same file, the builder is instance-patching a
shared decision rather than fixing it. The rule is mandatory, names the required
response (a Decision-Seam Analysis), and explicitly bans "another token, regex,
vocabulary row, or oracle fixture" as the next push.

The rule is correct. It has never fired on its own, because it lives in a 73 KB
document and needs a human to notice and count rounds.

Measured on ATLAS #2181 (`feat: advisory warning layer on editorial audits`,
+2,569/-94, 22 commits): 94 bot findings across 20 pushes over 5.6 hours, dead flat
(push 1 produced 5 findings, push 20 produced 5), every round on
`atlas_brain/services/content_factory_copy_verification.py`. Round 1 was "exclude
product names from answer-claim detection"; round 20 was "limit product-term
negation to the governing predicate" -- the same seam at finer grain, because an
open category (does this sentence make an unqualified claim?) was implemented with
an enumerable mechanism. Applying 3k.2's own trip condition to that PR's real data
fires at push 3, with 17 pushes and 76 of the 94 findings still to come.

Size is not the predictor: #2174 (+1,098) and #2175 (+1,064) drew zero review
rounds, and #2133 (+5,641, the largest diff sampled) landed in 2 commits.

### Problem-derived contract

**Root cause:** an existing, correct, mandatory rule has no automatic trigger. The
signal it depends on (findings per push, flat over three pushes, concentrated on one
file) is mechanically computable from the GitHub API, but nothing computes it, so
enforcement depends on a human counting rounds during a loop that pushes every ~13
minutes. This slice fixes the missing trigger, not the rule.

**What a correct fix must touch and change:** one detector that reconstructs
per-push finding counts and the dominant file from live PR data; one CI trigger that
runs it on the same cadence the rule measures (pushes); a surfacing mechanism that
reaches the builder without a human relaying it; and both-direction tests proving it
fires on non-convergence and stays silent on convergence, on breadth, and on size.

**What must not change:** the rule text in `AGENTS.md` and
`docs/GUARD_CLASS_CLOSURE.md` (3k.2 already says the right thing); any existing
workflow or detector; the repository's CI permission posture, which grants no
workflow `pull-requests: write` and posts no comments; and the merge gate -- this
must never block, because capping or blocking counts the symptom and, on surfaces
where the findings are real, would ship defects.

Diff-budget override: the detector is ~350 lines and the remainder is the
both-direction test suite the R13 boundary bar requires, plus this plan.
Splitting the tests from the detector they prove would produce a PR that cannot
be reviewed on its own, and splitting the plan from either would leave the
trip's contract unstated.

## Scope (this PR)

Ownership lane: process-guardrails
Slice phase: workflow/process

1. Add an advisory detector that buckets bot review findings into the push that
   preceded them and trips on a non-converging, single-seam window.
2. Add the CI trigger that runs it per push, mirroring the 3k.1 sibling.
3. Add stdlib-only tests covering both sides of the trip boundary, including a
   replay of the observed ATLAS #2181 shape.
4. Add this plan doc.

Max files: 4

### Files touched

- `.github/workflows/seam_convergence.yml`
- `plans/PR-Seam-Convergence-Breaker.md`
- `scripts/check_seam_convergence.py`
- `tests/test_check_seam_convergence.py`

### Review Contract

Acceptance criteria, checked one by one:

1. The detector trips on ATLAS #2181 and names
   `atlas_brain/services/content_factory_copy_verification.py`. It fires at
   round 6 of 18, not round 3: removing the tuned thresholds traded earliness
   for having no knobs, and Verification below records that cost rather than
   restating the pre-rewrite number.
2. It does not trip on #2174, #2175, or #2133 -- the last proving it keys on
   non-convergence rather than diff size.
3. It trips on #2158 (round 6) and #2161 (round 3), the two other known spirals.
4. It exits 0 on a trip. `--strict` exists but is not wired into CI.
5. A Decision-Seam Analysis suppresses the trip only when it names the tripped
   seam and appears in this PR's own declared plan or body. A marker for
   another seam, an unbound marker, or a marker in an unrelated plan does not
   suppress -- plan docs live on main after merge, so an unbound marker would
   have disabled the breaker for every later PR.
6. A strictly declining run never trips; two rounds never trip; findings scattered
   across files never trip; a window whose last round moved to another file never
   trips; a body that only mentions the phrase never suppresses.
7. The workflow requests only `contents: read` and `pull-requests: read`, uses
   `pull_request` plus the review events rather than `pull_request_target`, and
   therefore needs no entry in `ALLOWED_PULL_REQUEST_TARGET_JOBS`.
8. No existing file is modified.

**Reachability proof:** the real entrypoint is the `seam-convergence` job on
`pull_request`. Observable result: the job runs on this PR, reports success, and
emits no annotation (this PR is not a classifier). The detector's own behavior is
proven against live PR history by the replay commands in Verification, whose output
is the annotation text and per-push table.

Affected surfaces: CI only. No runtime, API, database, billing, delivery, or
user-visible surface. No dependency added -- the detector is stdlib plus the `gh`
CLI already used by `scripts/check_ai_reconciliation_live.py`.

Risk areas: false positives on a legitimately dense review of one file (mitigated by
requiring three consecutive pushes, a non-converging count, and single-file
dominance, and by never blocking); GraphQL pagination on very large PRs (capped at
50 pages, matching the sibling); timestamp comparison (ISO-8601 UTC from the API is
lexicographically ordered, so string comparison is exact and avoids a parsing
dependency).

- Reviewer rules triggered: R1, R2, R10, R12, R13. R12 covers the new GitHub Actions
  workflow and CI job this slice introduces. R10 and R2 come from the evaluator /
  gate-predicate shape of the detector; R13 from its classifier shape (the boundary
  must be probed from both sides); R1 from the repo-wide CI surface.

## Mechanism

`bot_review_rounds` turns bot review submissions into one round per reviewed
commit, carrying the paths they raised findings on. **The reviewed push is the
unit**, which is what 3k.2 counts. Keying on commits directly would split one
push into synthetic empty rounds; keying on submissions overcounts the other
way, because two enrolled bots review the same push and three submissions would
reach the window after only two pushes. Reviews of one commit merge, since the
rule asks what a round argued about rather than which bot said it. A bot review that raises
nothing is kept as an empty round rather than dropped -- it is evidence the loop
converged, and dropping it would splice two non-adjacent rounds together.

`find_trip` slides a 3-round window and trips only on facts, never on a tuned
number: every round in the window raised at least one finding, the same file leads
all three (`leading_path`, plurality, None on a tie), and that file's own finding
count does not decrease from the first round to the last. The count is measured on
the seam, so unrelated findings that happen to share a round cannot hold a
declining seam up.

`recorded_seam_analysis` looks for a line-anchored machine token,
`decision-seam-analysis: fix|waive|rescope`, in the PR body or any plan doc. It
deliberately does not parse prose: deciding whether a paragraph "really" analyses a
seam is itself an open category.

`fetch_reviews` paginates reviews and **raises** if any single review carries more
than one page of inline comments, rather than judging convergence from a truncated
prefix. Exit code 2 is retryable; a silent partial read is not.

On a trip the detector prints a `::warning file=<seam>::` annotation naming the
seam and restating 3k.2's required next action, then returns 0.

## Decision-Seam Analysis

Applied to this PR's own review loop, at round 2, before the breaker could fire on
it. Round 1 raised 5 findings and round 2 raised 6, five of the six on
`scripts/check_seam_convergence.py`. The detector run against this PR reported two
rounds and stayed silent -- it needs three -- so this is voluntary, one round early.

**The seam.** The detector's admit decision, "is this review window
non-convergent?", was implemented as a conjunction of tuned numbers: a convergence
ratio against the window mean, a per-round leading share, a window-wide dominance
majority, and a prose parser deciding whether a paragraph counted as an analysis.

**Why that decision is wrong.** It is an open category answered by enumeration.
For any threshold there is an adjacent case that argues for a different value, so
review can surface correct objections indefinitely -- which is exactly what rounds 1
and 2 did, each legitimately. `AGENTS.md` 3k.3 says to evidence-gate an open
recognizer rather than enumerate it. Building an open-category classifier to detect
open-category classifiers is the same mistake one level up.

**Disposition: fix the seam structurally.** The trip decision now rests on four
facts and no tunable number:

1. three consecutive rounds (quoted from 3k.2, not tuned),
2. none of them empty -- a bot review that raises nothing is convergence evidence,
3. the same file leads all three (plurality, a fact about each round), and
4. that file's finding count does not decrease across them.

The convergence ratio, the mean comparison, the dominance majority and the prose
parser are all deleted. The recorded analysis is now a machine token
(`decision-seam-analysis: fix|waive|rescope`), a closed category, read from the plan
or the PR body -- because judging prose was itself the open-category trap.

**Stated default direction.** The check is advisory and can never block, so its
error costs are lopsided: a false alarm wastes a builder's time and burns trust in
the check, while silence merely reproduces today's status quo, which is no detector
at all. The cheap error is silence. Every ambiguous case -- a tie for the leading
file, an empty round, a declining seam -- resolves to not speaking.

**Cost of the change, stated rather than hidden.** It fires later. On ATLAS #2181 it
trips at round 6 instead of round 3, on #2158 at round 8, on #2161 at round 3. That
is 12, 11 and 7 rounds of warning respectively, against loops that ran 18, 19 and 10
rounds. Trading three rounds of earliness for the removal of every knob is the
intended direction, not a regression.

decision-seam-analysis: fix scripts/check_seam_convergence.py

## Intentional

- **Rounds are bot review submissions, not commits.** A push can carry several
  commits, and commit dates can differ from push times; keying on reviews removes
  both problems at the root rather than reconstructing push boundaries.
- **Review events are enrolled alongside pushes.** A push-only trigger would stay
  silent until a fourth push had already made the patch the breaker exists to
  prevent, and would never fire if the builder stopped after the third review.
- **Advisory, exits 0.** Not a round cap. The operator rejected capping (2026-07-25):
  it counts the symptom, and the preserved 2026-07-10 nuance is that on
  classifier/PII/gate surfaces the findings are real, so capping ships defects. This
  changes the permitted class of fix, not whether the PR may proceed.
- **`pull_request`, not `pull_request_target`.** The measured unit is the push, so the
  natural trigger is also the safe one. It fires one push after the bot comment, which
  on #2181 still means push 3 of 20.
- **Annotations, not a PR comment.** No workflow in this repository holds
  `pull-requests: write` and none posts comments. Matching
  `scripts/check_guard_class_closure.py` keeps that posture intact and still surfaces
  the warning inline on the Files tab.
- **Resolved threads are counted.** Counting only unresolved threads would erase the
  evidence, since the builder resolves each round before pushing again.
- **`--strict` exists but is unused.** Same shape the 3k.1 sibling shipped with, so a
  later promotion is a workflow edit rather than a detector rewrite.
- **String timestamp comparison.** ISO-8601 UTC from the GitHub API sorts
  lexicographically; parsing would add a dependency and no correctness.

## Deferred

- **`--strict` promotion to a failing or required check**, after an advisory-proving
  period. Unlocked by observing it on real PRs without a false trip. Trusted-base
  execution is a named precondition, exactly as `guard_class_closure.yml` records for
  its own promotion.
- **The 3k.3 plan-time gate** in `plan_admission.yml`: ask "can I enumerate the
  complete set of inputs that should trip this?" before code exists. Higher leverage
  than this breaker -- this catches the loop, that prevents it. Separate slice.
- **Port to the EOM repositories.** Codex reviews `eom-timetracker` and the website
  too, but neither carries 3k.2; the rule has to land in their `AUDIT_PROTOCOL.md`
  first.
- **Waived residual, recorded rather than fixed:** the breaker can be defeated by a
  builder who alternates the file they patch, or who draws a clean bot round between
  spirals. Both are silence-side errors and therefore acceptable under the stated
  default direction; closing them would reintroduce the tuned-threshold surface this
  slice just removed.
- Parked hardening: none.

## Verification

Commands run locally, with results:

1. Detector unit tests -- `python -m pytest tests/test_check_seam_convergence.py -q
   --noconftest` -- **44 passed**, including a regression for every finding from both
   review rounds and `pytest.raises` coverage of each error path.
2. Live replay, must trip -- PR 2181 trips at round 6 of 18 on
   `atlas_brain/services/content_factory_copy_verification.py` (seam counts 3, 1, 4);
   PR 2158 at round 8 of 19; PR 2161 at round 3 of 10. All exit 0.
3. Live replay, must stay silent -- PRs 2174, 2175 and 2133 each report "OK: no window
   of 3 consecutive rounds...". 2133 is the largest diff sampled (+5,641), proving the
   detector does not key on size.
4. Self-check -- run against this PR at two review rounds it reports OK, which is
   correct: three are required. The Decision-Seam Analysis above was written
   voluntarily, one round before the breaker could fire.
5. Workflow security posture audit -- passed, with no warning attributed to the new
   workflow.
6. Plan doc audit, plan/code consistency, and reviewer rules triggered -- all pass;
   the full local gauntlet reports 18 checks green.
7. Repository ASCII check for Python files -- passed.
8. Maturity sweep on the scripts lane -- the detector scores 6, matching the 3k.1
   sibling and below the lane's min-score of 8, so no baseline entry is added.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/seam_convergence.yml` | 90 |
| `plans/PR-Seam-Convergence-Breaker.md` | 278 |
| `scripts/check_seam_convergence.py` | 485 |
| `tests/test_check_seam_convergence.py` | 489 |
| **Total** | **1342** |
