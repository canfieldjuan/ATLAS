# Reviewer Rules Pack v1

> The reviewer's job is **not** to "review the code." It is to **disposition the
> review matrix: every acceptance criterion in the PR's Review Contract, and
> every rule below.** Every rule is in the matrix -- the path-trigger table sets
> how DEEPLY a rule is probed, never whether it appears -- so a rule reaches a
> verdict of pass / fail / not-verified / n-a, and `n-a` carries a reason.
> Every review finding cites a rule ID (R1-R14). This pack is the checklist the
> reviewer runs; the
> recurring-lapse list in `docs/SESSION_BOOTSTRAP.md` is the same checklist
> front-loaded into the builder so the repeats stop.
>
> The standard is *the matrix is dispositioned*, not *the code violates no rule*.
> "Violates none of the rules" is a universal negative -- undischargeable on any
> non-trivial diff, so it defines no point at which a review is done. See
> **Review completion** below, which is what makes a review finishable.

This pack sits **under** the existing verdict ladder, it does not replace it:

| Verdict | Meaning |
|---|---|
| **BLOCKER** | A rule below is failed in a way that breaks correctness, security, a contract, or CI. Must fix before merge. |
| **MAJOR** | A rule is at risk: architectural / scope / pattern concern. Fix if small; else discuss. |
| **NIT** | Style / naming / polish. Apply only if 1-line; reviewer marks skip-worthy. |
| **LGTM** | Every rule R1-R14 is Pass or a reasoned N-A (no Not-Verified outstanding), R14 is satisfied, and all AI findings are fixed-or-waived. |

A finding is written as `Rxx (LEVEL) file:line - issue - required fix`.
**Blockers must cite `file:line`.** A bare "LGTM" with no rule matrix and no
independent verification is worse than no comment.

R14 is universal: it applies to every review verdict, even when no changed path
specifically triggers it. A reviewer who has not inspected the checked-out PR
head and relevant codebase evidence cannot issue LGTM.

---

## The Review Contract (authored during planning)

Every non-trivial PR's plan doc (`plans/PR-<Slice>.md`) carries a Review
Contract block inside its **Scope** section. The builder codes against it; the
reviewer reviews against it. No contract, nothing to check against.

```
### Review Contract
- Acceptance criteria:
  - [ ] Behavior A works
  - [ ] Edge case B handled
  - [ ] Existing behavior C unchanged
- Reachability proof: real entrypoint + observable output/state, or N/A with reason
- Affected surfaces: API / DB / auth / frontend / jobs / config / observability / third-party
- Risk areas: data-loss / security / backcompat / performance / concurrency / migration
- Reviewer rules triggered: R1, R2, ... (see path triggers below)
```

The contract is optional for one-off scratch, mandatory for non-trivial PRs
(same threshold as the plan doc itself, per `AGENTS.md`).

---

## The rules

### R1 - Requirements match
The code satisfies the Review Contract's acceptance criteria, solves the stated
problem (not a different one), and contains no unrelated changes.
**Block if:** an acceptance criterion is unimplemented; the implementation
solves a different problem; a newly introduced surface claimed as wired is not
reachable through its real entrypoint; a deferred or N/A reachability decision
lacks a reason or named follow-up; the PR includes scope creep or "while I was
here" cleanups beyond the slice's contract.

### R2 - Test evidence
Every meaningful behavior change has a test, or a documented reason it cannot.
**Block if:** new logic has no direct test; a bug fix has no regression test; a
critical path has only manual testing; tests assert implementation details
instead of behavior; tests cover only a trivial happy path while realistic
negative, edge, malformed, sparse, or varied-input cases remain unexercised.
For detectors/validators/gates, the failure branch is proven to fire
(`AGENTS.md` 3i), not just the happy path.

For any new runtime, workflow, UI, report, billing, delivery, or public
contract surface, unit-only tests are not enough by themselves. The PR needs a
thin reachability proof: exercise the real entrypoint and assert an observable
output, persisted state, rendered UI, emitted artifact, queued job, or gate
result. Unit-only proof is acceptable when the PR introduces no new reachable
surface, or when the plan explicitly defers wiring and names the follow-up.

### R3 - Security and authorization
Any user input, permission check, token, secret, file upload, webhook, or admin
action is reviewed for abuse.
**Block if:** authorization is missing or checked too late; client-side checks
replace server-side checks; secrets/tokens/credentials are exposed or logged;
input is trusted without validation; tenant/user isolation can be bypassed;
per-tenant credentials do not fail **closed** (an unprovisioned tenant must
never borrow shared creds).

### R4 - Data and migration safety
Database, schema, migration, and backfill changes are safe to deploy and roll
back.
**Block if:** a migration is destructive without backup/rollback; a migration
can lock large tables unexpectedly; new non-null columns lack safe
defaults/backfill; code assumes migrated data before the migration is
guaranteed; the rollback plan is missing.

### R5 - Backward compatibility
Public APIs, MCP tool surfaces, events, schemas, and persisted data stay
compatible unless the break is explicitly flagged (BREAKING in the title/plan).
**Block if:** a request/response shape changed silently; old clients will fail;
feature flags or versioning are missing where needed; contract tests are
missing for a changed surface.

### R6 - Error handling and observability
Failures are handled intentionally and are diagnosable.
**Block if:** errors are swallowed; retry behavior can duplicate side effects;
logs omit useful context; sensitive data is logged; metrics/traces are missing
for an important flow. Secondary writes (audit/history/notify after a
charge/send/publish) must be best-effort and must not fail an
already-successful op.

### R7 - Performance and scalability
No obvious performance traps are introduced.
**Block if:** N+1 queries are introduced; large operations run synchronously on
a request path; loops make network/DB calls unnecessarily; caching or
invalidation is broken; pagination or limits are missing.

### R8 - Concurrency and idempotency
Async jobs, webhooks, retries, and state transitions tolerate duplicate or
out-of-order execution.
**Block if:** a retry can double-charge, double-send, or double-create; race
conditions can corrupt state; state transitions are unguarded; idempotency keys
or uniqueness constraints are needed but missing.

### R9 - Frontend behavior
User-facing changes handle real states.
**Block if:** loading / empty / error / success states are incomplete; form
validation is only cosmetic; accessibility basics are broken; important UI
behavior lacks test or manual evidence; responsive/mobile behavior is ignored
where relevant.

### R10 - Maintainability
The code is understandable without summoning the original author.
**Block if:** logic is duplicated instead of centralized; naming hides intent;
complex code lacks structure or tests; dead/debug code or unrelated refactors
are included; the abstraction is larger than the problem.

### R11 - Dependencies and config
New dependencies and config changes are justified.
**Block if:** a dependency is unnecessary or risky; license/security
implications are unclear; configuration is read from raw `os.environ` instead of
a typed `ATLAS_*` field in `atlas_brain/config.py`; env vars lack defaults/docs;
a config change affects production unexpectedly.

### R12 - Deployment safety and CI enrollment
The change is safe to ship incrementally and is actually exercised by CI.
**Block if:** risky behavior has no feature flag; there is no rollback/disable
path; deployment order matters but is undocumented; monitoring is missing for a
high-risk change; **a new or renamed test is not wired into the CI workflow that
runs it** (adding a `test:*` script does not make CI run it - the matching
`run:` step ships in the same PR); a new CI/prod surface is declared but no
real route, workflow, job, or delivery path is enrolled to exercise it.

### R13 - Fix the class, not the example
Review findings that identify a defect class must be fixed at the class level,
not by hardcoding the reviewer's cited strings, values, paths, or examples.
**Block if:** the patch hardcodes the reviewer's example values; the tests reuse
only the examples named in the finding; the mechanism cannot pass a held-out
same-class probe; or the builder claims the class is fixed without showing
fresh same-class cases the reviewer did not provide. Preferred proof is a
property/parametrized test that generates cases; when generation is not
practical, use multiple unseen fixtures plus a short explanation of the
generalized mechanism. Generated or unseen cases must be diverse enough to
exercise the class, not trivial near-duplicates that satisfy the easy path.

**Open-category exception (evidence-gate, do not enumerate).** When the class is
an *open semantic category* the guard cannot enumerate on either side (person
names, senders, intent, language, is-junk), neither a denylist nor a
member-allowlist closes it, and a property test over category members does not
either. Require that the fix meets the open-category form defined canonically in
`docs/GUARD_CLASS_CLOSURE.md` (do not restate it here). **Block if:** a diff answers a same-class
finding with the next member patch (token, regex, vocabulary row, oracle
fixture) while the thread history shows the class is open and not converging --
that is the `AGENTS.md` 3k.2 convergence circuit-breaker, and the next push owes
a Decision-Seam Analysis, not another example.

### R14 - Verify against the codebase, not the PR story
Review verdicts must be based on the checked-out PR head and the current
codebase, not the PR description, issue summary, builder claims, or prior
conversation. Claims used in a verdict are verified by reading the relevant
code, checking at least one relevant caller/test/artifact path, running or
inspecting the relevant command output, or explicitly marking the claim "not
verified" with a reason. **No LGTM from claims alone.**
**Block if:** the verdict accepts a PR claim without checking the codebase; the
review does not name the reviewed head SHA; the reviewer did not inspect the
changed code; a shared-function or contract change lacks a caller/test/artifact
spot-check; or skipped verification is omitted instead of listed as "not
verified."

## Boundary-probe before LGTM on guard-shaped PRs

Before LGTM on any PR whose change is a guard, validator, cap, classifier,
gate, sanitizer, denylist, parser admission rule, or safety checker, run a
boundary probe and state `boundary-probe: <what applied + result>` in the
review.

A guard usually fails on its second side. Check both sides:

- **Both error directions:** test one input that should pass but might be
  rejected, and one input that should fail but might pass.
- **Partial/mixed input:** test some-required-keys-present-some-missing, and
  mixed valid/invalid collections. Do not test only full-valid and empty.
- **Boundary values:** test min-1/min/max/max+1, empty, single-item, and
  large-but-valid where relevant.
- **Falsy/default defeat:** any `x or d`, `x || d`, or `if not x` default on a
  cap, limit, count, permission, or threshold needs probes for `0`, `""`,
  `False`, and past-the-max values. For `??`, probe only null/undefined.
- **Original-vs-sanitized path:** verify downstream code uses the sanitized or
  validated value, not the original raw value after the check.
- **Constructed metadata:** a sanitizer must clean ids, keys, filenames,
  labels, source ids, and derived paths it constructs from input, not only
  field values it copies.
- **Negative test exists:** never LGTM a guard whose tests only prove good
  input passes. Require at least one test proving bad input fails, or record a
  justified waiver.

If the guard protects security, billing, data deletion, customer-visible
output, or CI/release gates, missing boundary proof is BLOCKER. Otherwise it is
at least MAJOR.

**Open-input guards additionally require class-closure.** When the guard's input
space is open -- free text, nested/recursive structures, producer-supplied
keys/values -- boundary probes alone are not enough: they prove the sampled
inputs, not the class. Before LGTM, require and state that the guard meets the
class-closure bar defined canonically in `docs/GUARD_CLASS_CLOSURE.md` (its
open-category exception evidence-gates instead of enumerating; see also R13 and
`AGENTS.md` 3k.2 for a non-converging loop). Block until it holds; the
requirements are not restated here -- that doc is the single source. Confirmed
fail-opens in money/auth/PII/safety guards block regardless of review-round
count.

---

## Path-based rule triggers

Not every PR needs every rule at full depth. The changed paths name which traps
to inspect. The plan's "Reviewer rules triggered" line should list at least
these for the paths it touches:

| Changed path glob | Rules triggered |
|---|---|
| `db/migrations/**`, `*.sql` migrations | R4, R2 (migration/rollback test) |
| `atlas_brain/api/**`, `atlas_brain/mcp/**` | R1, R2, R5 |
| `**/auth/**`, login/token/permission code | R3, R2 (negative permission tests) |
| invoicing / billing / payment code | R3, R8 (idempotency + audit log) |
| `atlas_brain/autonomous/**`, webhooks, jobs | R6, R8 (retry safety) |
| `atlas-*-ui/**`, `*.tsx` | R9, R12 (CI enrollment) |
| `atlas_brain/config.py`, env/config | R11, R12 |
| `scripts/audit_*.py`, `scripts/check_*.py`, evaluators / gate predicates | R2 (failure-branch fixtures per `AGENTS.md` 3h/3i), R10 |
| `extracted_*/` synced files | R1, R10 (manifest sync discipline) |
| Guard, validator, cap, classifier, gate, sanitizer, denylist, parser admission rule, or safety checker changes | R2, R14 (boundary-probe before LGTM) |
| Concurrency, durability, lock/lease, cache-coherence, rotation or retry state-machine, or crash-safe replacement changes | R8, R2 (`AGENTS.md` 3k.4: every such plan names the selected component and which of its guarantees close which part of the seam, states the execution model with the invariants holding for every interleaving it admits and anything uncovered as an explicit assumption, and keeps one execution surface per slice; **only if it hand-rolls** it also names the component it rejected and why) |
| Review comments that name a defect class ("all X", "class of Y", "same failure mode") | R13 (held-out/propertied proof that the class, not only the example, is fixed) |
| All reviewer verdicts | R14 (checked-out PR-head and codebase-backed verification) |

`scripts/audit_review_rules_triggered.py` is the mechanical audit for
machine-matchable path triggers. Local review and trusted-base CI run it for
each changed plan, and it fails when the Review Contract's rule declaration
omits a triggered rule. Prose-only trigger rows remain explicit advisory
findings because no path glob can safely derive them.

---

## Class-defect review framing

When a reviewer finds a class defect, the finding should say so explicitly:
"This is a CLASS defect; the example below is illustrative, not the target."
Name the cited example, name at least one visible same-class probe, and state
that the reviewer may keep a held-out probe for re-review. This prevents the
review comment from becoming a hardcoding target.

The reviewer should reject a "fixed" response that only proves the cited
example. Before LGTM on an R13-triggering finding, verify one of:

- a property/parametrized test generates diverse same-class cases;
- unseen fixtures cover varied cases not listed in the original finding; or
- the reviewer reran a held-out probe and the verdict records it.

---

## Review completion (the stopping rule)

A review is **complete** when its matrix is dispositioned, and the review states
that matrix:

1. **Each acceptance criterion** in the Review Contract: met / not met /
   could-not-determine, with checkable evidence -- `file:line` for a
   code/content claim, or a named non-file artifact (command + output, CI
   run/job, generated artifact, PR metadata) where the criterion is not about
   source. Same evidence forms the binding reviewer workflow already accepts
   (`AGENTS.md` 4a step 4); demanding `file:line` for a CI-status or
   command-output criterion would either stall the matrix or buy a green tick
   with an irrelevant citation.
2. **Every rule, R1-R14**: pass / fail / not-verified / n-a-with-reason. The
   path-trigger table sets how deeply each is probed, **not which appear**. A
   behavior change under a path the table does not list still owes an R2
   verdict; deriving matrix membership from the table would let exactly that
   PR reach LGTM without anyone asking whether the new behavior has tests.
3. **What was not verified**, listed with the reason.

That is the whole standard for *stopping*. Completeness is never "no further
case can be found" -- on an open surface no such point exists, so a reviewer
holding that standard reports forever.

**Complete is not the same as approved.** A matrix with honest `not-verified`
or `could-not-determine` entries is a complete review -- the reviewer may stop
-- but it is **not** an LGTM. LGTM requires every rule to be `pass` or a
reasoned `n-a` (see the ladder above); an unresolved entry is an open question,
so the verdict is *needs verification*. Conflating the two would let a review
that verified almost nothing produce a green merge gate, which is the opposite of
what a stopping rule is for.

**Discharge is per head SHA.** A rule marked *pass* is discharged for **that
head only**. A new head is new code: every rule re-opens on it freely, with no
argument owed, because a defect the builder just introduced was not missed
earlier -- it did not exist earlier. Requiring the reviewer to disown a
previously-correct pass before reporting it would suppress exactly the
regressions a re-review is for.

The argument requirement applies only to re-opening a rule **on the same head**
-- the reviewer revisiting their own verdict on unchanged code. There, state
why the earlier discharge was wrong: the condition that was never actually met,
not one more instance of a decision already reported. A discharge repeatedly
overturned on unchanged code is evidence the rule was never dischargeable as
scoped, which is a finding in itself.

**Recording the gate: an unresolved entry is a BLOCKER-level finding.** The
`claude-review` status has three states and "complete but not approved" is not a
fourth one -- it maps onto the existing `failure`. A `not-verified` rule or a
`could-not-determine` criterion is filed as a BLOCKER (`Rxx (BLOCKER) - not
verified: <what, and why not>`), so `scripts/set_claude_review_status.py` takes
`failure` and `success` is unreachable while anything is unresolved. This does
**not** narrow `success` to LGTM: it keeps its established meaning of "no open
BLOCKER", so a complete review carrying only non-blocking MAJOR/NIT notes is
still `success`, exactly as `docs/REVIEWER_MERGE_GATE.md` and
`scripts/set_claude_review_status.py` already define it. `pending` keeps its
meaning -- a review still in progress -- and is not a parking space for a
finished review with open questions.

This is deliberately fail-closed. Unverified evidence is exactly what a merge
gate exists to stop, so the burden is on resolving the entry or waiving it as a
reasoned `n-a`, never on the gate to guess. The reviewer is still *done*: the
matrix bounds the search, the verdict reports what it found, and neither forces
the other.

**Report the class, not the instance.** R13 obliges the *builder* to fix the
class rather than the cited example. The same duty binds the *reviewer*: when
two or more findings share one underlying decision, file **one** finding that
names the decision, carrying the instances as illustrations. A finding whose own
text opens "fresh evidence beyond the earlier `<X>` finding" is by its own words
another instance of a decision already reported -- merge it into that finding
instead of filing it separately. Where the decision keeps producing instances,
the `AGENTS.md` 3k.2 breaker applies to the reviewer too: file the seam once and
say so, rather than the next adjacent case.

Nothing here licenses withholding a real defect. It changes how defects are
*reported* -- once, at the level the fix has to happen anyway -- not whether.

**Why:** measured on two PRs. #2195 drew 35 findings over 13 rounds; **18 of the
35 (51%) explicitly declared themselves adjacent to an earlier finding**, and 27
collapse into three decisions (source attestation 12, receipt lifecycle 12,
preflight ordering 3). #2184 drew 33 over 14 rounds, collapsing into four
decisions plus four distinct findings. Reported class-first, the same defects
surface in roughly three rounds instead of thirteen, with none lost. Round 6 of
#2195 is the cost of the alternative: it reverses round 5 ("reject ignored
bytecode" became "stop rejecting bytecode created by the CLI itself"), which is
what instance-by-instance boundary-shifting produces.

---

## AI-finding reconciliation (mandatory before LGTM)

External review bots (Codex, Copilot) post raw comments outside the
BLOCKER/MAJOR/NIT taxonomy. They are **advisory inputs to a judgment session,
never auto-applied** - a bot false-positive applied blindly turns correct work
into incorrect work, so there is no "auto-address all comments" loop.

The hard rule: **a reviewer may not issue LGTM until every AI finding is either
fixed or explicitly waived with a reason recorded in the PR body.** The machine
catches mechanical issues; the human owns intent mismatch, product logic,
architecture, risky assumptions, and missing tests. The reviewer compares their
own rule matrix against the AI output and reconciles the difference.

---

## Turning misses into mechanism

Every escaped defect (a bug that shipped past an approved review) is logged in
`REVIEW_MISSES.md` and converted into one durable form so it cannot silently
recur: a new `scripts/audit_*.py`, a new rule ID here, a new path trigger above,
a line in the recurring-lapse checklist, or a Review Contract template change.
**No escaped defect is fixed only once.** This is the reviewer-side mirror of
the builder's `HARDENING.md` + recurring-lapse flywheel.

**The ratchet releases too.** As written this section only ever adds, and the
reviewer's mandate grows with it -- every added rule is another category to
disposition on every future PR, forever, which is a cost paid by every
subsequent review.

**When retirement is considered:** on each addition. Adding a durable mechanism
is the trigger to examine one existing candidate for removal, so the ratchet
self-balances and there is no separate cadence, owner, or audit to maintain.
**Record which candidate you examined** in `REVIEW_MISSES.md` with the outcome,
and take the least-recently-examined mechanism -- when every mechanism has been
examined once the rotation simply starts again, so a retained one is re-checked
against fresh evidence rather than becoming permanently ineligible. Keying
eligibility off "has not fired since it was examined" would exempt exactly the
quiet mechanisms this section is about, and after one full pass only newly added
ones would ever be inspected. An unrecorded pick has the opposite failure: every
addition re-examines the same load-bearing mechanism, re-states why it stays,
and retires nothing while the pack keeps growing. All
**five** mechanism kinds this section creates are in scope -- `scripts/audit_*`
checks, rule IDs, path triggers, recurring-lapse lines, and Review Contract
template additions -- not only the ones that are cheapest to delete.

Retirement terms:

- **Silence is not evidence of absence.** A mechanism that has not fired may be
  the reason the failure stopped. So not-firing is a prompt to examine, never a
  licence to delete: a mechanism that is the sole protection for a defect logged
  in `REVIEW_MISSES.md` may be removed **only** by naming what now covers that
  defect, and the replacement has to be able to **stop the escape** -- a broader
  rule in this pack, a type or structural change that makes the defect
  unrepresentable, or a test that runs in a **required** check. A test that no
  required check executes cannot prevent anything, so it is not a replacement.
  No enforcing replacement named, no removal; state why it stays.
- **Consistently waived** is the stronger signal, because it means the mechanism
  fires and reviewers keep judging it wrong. Re-scope it to what actually
  blocks, or remove it.
- **Removals are atomic across mirrors.** A rule ID, checklist line, or template
  entry is referenced from more than one place -- `AGENTS.md` review guidelines,
  the §2a verification template, the 4d audit checklist, and the completion
  matrix all enumerate rules. A removal updates every mirror in the same change,
  or it leaves the matrix demanding a verdict on a rule that no longer exists.
- **Removals are recorded** in `REVIEW_MISSES.md` beside the defect that
  prompted the addition, with the replacement coverage named, so a removal is a
  decision with a paper trail and not an erosion.

Nothing here weakens the "no escaped defect is fixed only once" rule -- an
escaped defect still converts into mechanism. This governs what happens to that
mechanism afterwards, so the pack stays the size a reviewer can actually apply.
A checklist nobody can finish is the failure mode Review completion exists to
prevent, and an unbounded ratchet recreates it one rule at a time.
