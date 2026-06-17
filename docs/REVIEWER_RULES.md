# Reviewer Rules Pack v1

> The reviewer's job is **not** to "review the code." It is to **prove whether
> the PR satisfies its Review Contract and violates none of the rules below.**
> Every review finding cites a rule ID (R1-R14). This pack is the checklist the
> reviewer runs; the recurring-lapse list in `docs/SESSION_BOOTSTRAP.md` is the
> same checklist front-loaded into the builder so the repeats stop.

This pack sits **under** the existing verdict ladder, it does not replace it:

| Verdict | Meaning |
|---|---|
| **BLOCKER** | A rule below is failed in a way that breaks correctness, security, a contract, or CI. Must fix before merge. |
| **MAJOR** | A rule is at risk: architectural / scope / pattern concern. Fix if small; else discuss. |
| **NIT** | Style / naming / polish. Apply only if 1-line; reviewer marks skip-worthy. |
| **LGTM** | All triggered rules pass, R14 is satisfied, and all AI findings are fixed-or-waived. |

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
solves a different problem; the PR includes scope creep or "while I was here"
cleanups beyond the slice's contract.

### R2 - Test evidence
Every meaningful behavior change has a test, or a documented reason it cannot.
**Block if:** new logic has no direct test; a bug fix has no regression test; a
critical path has only manual testing; tests assert implementation details
instead of behavior; tests cover only a trivial happy path while realistic
negative, edge, malformed, sparse, or varied-input cases remain unexercised.
For detectors/validators/gates, the failure branch is proven to fire
(`AGENTS.md` 3i), not just the happy path.

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
`run:` step ships in the same PR).

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
| Review comments that name a defect class ("all X", "class of Y", "same failure mode") | R13 (held-out/propertied proof that the class, not only the example, is fixed) |
| All reviewer verdicts | R14 (checked-out PR-head and codebase-backed verification) |

Phase 1 of this convention is documentation + reviewer discipline. A later
slice adds a mechanical audit that derives the required rule IDs from the diff
and fails when the plan's triggered-rules line omits one (see
`AGENTS.md` 4 and the workflow-redesign issue).

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
