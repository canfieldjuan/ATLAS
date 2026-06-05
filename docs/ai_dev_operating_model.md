# How Atlas Ships Production Code With AI — Fast and Reliable

> A field description of the operating model behind the Atlas repo: **two builder
> sessions and two reviewer sessions running in parallel, kept safe by mechanical
> gates rather than human vigilance.** Written for anyone trying to improve the way
> they build real software with AI coding agents.

The thesis in one sentence: **make the machine catch every failure mode that has
ever cost a review cycle, so the humans and the AI's judgment are only ever spent
on things a script can't decide.** Everything below is in service of that.

---

## 1. The crew model: 1 coder + 1 reviewer, run twice in parallel

Work is split into **two independent lanes** (e.g. "content generation" and
"deflection/monetization"). Each lane is a pair:

- **Builder session** — writes the plan, writes the code, opens the PR.
- **Reviewer session** — audits that PR *independently* and posts a single
  verdict: **BLOCKER / MAJOR / NIT / LGTM**.

So at full tilt the operator drives **4 sessions** (2 builders + 2 reviewers).
The builder and reviewer are deliberately *different* sessions because a model
reviewing its own work inherits its own blind spots — the reviewer re-derives
claims from scratch instead of trusting them.

The reviewer doesn't rubber-stamp. The contract (`AGENTS.md §4a`) is **independent
verification**: re-run the builder's commands, spot-check the plan's invariants at
the actual `file:line` in the diff, and sweep for missed call sites with grep
patterns *more* reliable than the PR's own claim. "A bare LGTM with no verification
is worse than no comment."

| Verdict | Meaning | Builder action |
|---|---|---|
| **BLOCKER** | Correctness, security, contract break, or CI red | Fix before merge |
| **MAJOR** | Architectural / scope / pattern concern | Fix if small; else discuss |
| **NIT** | Style / naming / polish | Apply only if 1-line; reviewer marks skip-worthy |
| **LGTM** | All gates green | Merge |

---

## 2. The unit of work: a thin, planned, single-purpose PR

Nothing non-trivial is written without a **plan doc first** — `plans/PR-<Slice>.md`,
with seven required sections **in a fixed order**:

1. **Why this slice exists** — what's broken / what audit item it closes
2. **Scope (this PR)** — starts with `Ownership lane:` + `Slice phase:`, then the files touched
3. **Mechanism** — how it works, so the reviewer doesn't reverse-engineer the diff
4. **Intentional** — things that look wrong but aren't (rejected alternatives)
5. **Deferred** — punted work, each naming its future PR
6. **Verification** — exact commands the builder ran + pass counts
7. **Estimated diff size** — LOC budget

Two hard shape rules make slices reviewable:

- **<400 LOC per PR** (soft cap). Over-budget PRs must justify the overage in *Why*.
- **Slice phase** is named and must match the diff: `Vertical slice → Functional
  validation → Robust testing → Production hardening → Product polish` (plus
  `Workflow/process` for tooling changes).

**The plan is the contract; the code is the implementation of the contract.** If
the plan changes mid-build, the plan doc updates *in the same commit*. A plan that
arrives in a follow-up commit is an explicit anti-pattern.

### The thin-slice + HARDENING.md discipline

A slice fixes inline **only what it cannot function without**. Everything else
discovered along the way — error-handling gaps, naming, refactors, edge cases —
gets appended to a root `HARDENING.md` queue (with file, severity `S/M/L`,
category, and the slice it was found in) and **left out of the diff**. This is how
scope creep is structurally prevented: there is an approved place to put "while I
was here" thoughts that *isn't* the current PR.

---

## 3. Drift defense: the machine guards the lanes

Running 4 sessions against one `main` invites three kinds of drift. Each has a
dedicated mechanical check, run locally *and* in CI.

**(a) Session/PR ownership drift** — a session touching a PR that isn't its.
`scripts/check_session_pr_ownership.py` reads a mandatory, git-ignored
`SESSION_STATE.local.md` (owned PR, branch, head SHA, "may touch", "must not
touch") and **fails** if the target PR isn't owned or the branch/SHA don't match.
A PR in the same lane is *not* automatically owned; an abandoned-looking PR is
*not* owned. When in doubt, stop and ask the operator.

**(b) Cross-session file/lane drift** — two PRs colliding on the same files.
`scripts/audit_pr_session_drift.py` parses every open PR's plan, extracts its
`Ownership lane`, and **fails** if two PRs claim the same lane or this branch
touches files the base branch already changed.

**(c) Plan↔code drift** — the diff not matching what the plan claims. Four
audits enforce truth between the plan doc and the actual `git diff`:

| Script | Fails when… |
|---|---|
| `audit_plan_doc.py` | A required section is missing, duplicated, or out of order |
| `audit_plan_doc_files_touched.py` | Plan's "Files touched" ≠ actual diff (missing or extra) |
| `audit_plan_doc_diff_size.py` | Actual LOC drifts >50% from the estimate |
| `audit_plan_code_consistency.py` | A backticked path/function the plan name-drops doesn't exist in the tree |

`scripts/sync_pr_plan.py` rewrites the "Files touched" and diff-size sections
*from the real diff*, with a `--check` mode CI can gate on — so the plan can't
silently lie about what shipped.

---

## 4. Compaction discipline: the single highest-leverage insight

This is the part most teams miss. **Forensic observation in this repo found that
the builder's regressions cluster at conversation-*compaction* boundaries** — when
the agent's context gets summarized mid-task it can drift hard: closing PRs in
other lanes, redoing already-merged work, jumping lanes. (See
`docs/SESSION_BOOTSTRAP.md`: *"the builder's regressions cluster at compaction
boundaries, and a compaction landing close to a PR-open causes hard drift."*)

The countermeasures are entirely about **context hygiene**:

- **Keep sessions short; restart proactively.** A fresh session with a tight
  bootstrap prompt beats one long session that compacts repeatedly. Restarting is
  cheaper than recovering from drift.
- **Stop after opening a PR.** The builder reports the PR URL + the local checks it
  ran, then *hands back to the operator* — it does **not** poll CI or wait for
  review. Waiting is what burns context into a compaction.
- **Read narrow, test scoped.** During iteration: read targeted line ranges of big
  files (not the whole 1.4k-line file), run the single relevant test file (not the
  suite). Run the full gauntlet **once**, right before pushing.
- **Externalize state to disk.** `SESSION_STATE.local.md` is the session's memory
  that *survives* a compaction — current lane, owned PR, last safe action. After
  any compaction/restart the session re-reads it instead of guessing.
- **Two canned prompts** in `docs/SESSION_BOOTSTRAP.md`: a **bootstrap** to seed a
  fresh session fast, and a **drift redirect** to paste the moment a session shows
  post-compaction drift signals ("Stop. You likely just compacted. Do not close,
  merge, or modify any PR…").

The principle to steal: **treat compaction as a known hazard with a known recovery
procedure, and design the workflow so a session is never holding critical state
only in its context window.**

---

## 5. The recurring-lapse list: turning mistakes into mechanism

This is the flywheel that makes speed compound instead of decay. Every time a
failure mode costs a review cycle, it gets converted into one of three durable
forms so it can never silently recur:

1. **A mechanical audit** (`scripts/audit_*.py` / `check_*.py`) — preferred. And
   crucially, **auditors must surface unfamiliar input as drift, never silently
   skip it** (`AGENTS.md §3g`); every new audit ships with fixture tests proving its
   *failure* branch actually fires (`§3h`, `§3i`) — happy-path-only coverage of a
   detector is explicitly not enough.
2. **A line in `AGENTS.md` / `CLAUDE.md` / the bootstrap prompt** — when the rule
   needs judgment a regex can't capture.
3. **A config/typed seam** — e.g. "never read `os.environ`, add a typed
   `ATLAS_*` field" makes the wrong thing structurally harder than the right thing.

The bootstrap's recurring-lapse checklist (`SESSION_BOOTSTRAP.md §1.4`) is *the
same checklist the reviewer runs on every PR* — front-loaded into the builder so
the repeats stop. A sampling of lapses already codified:

- Config goes through `atlas_brain/config.py` typed fields; never raw `os.environ`.
- `extracted-checks` CI runs with **no torch, no asyncpg** — a test importing those
  at module top breaks *collection of the whole suite*. (Test-placement rule.)
- **CI enrollment is part of test authoring, same PR** — adding a `test:*` script
  doesn't make CI run it; you must add the `run:` step. An audit
  (`audit_extracted_pipeline_ci_enrollment.py`) fails on un-enrolled tests so you
  **can't add a test that skips CI**.
- Secondary writes (audit/history/notify after a charge/send/publish) are
  best-effort: wrap so they can't fail an already-successful op.
- Lookup-and-backfill fails safe on ambiguity (0 or >1 matches → don't guess).
- Per-tenant credentials fail **closed** — an unprovisioned tenant never borrows
  shared creds.
- "Passed locally" ≠ green. CI is truth.

---

## 6. The extracted-package sync discipline (advanced, but instructive)

Atlas extracts subsystems into standalone packages (`extracted_*/`) that must stay
byte-identical to their `atlas_brain/` source. The drift control is a **manifest
with two entry types**:

- **Synced** (`source` + `target`): canonical copy lives in `atlas_brain/`; edit
  there and run the sync script. `audit_extracted_manifests.py` **byte-compares**
  and fails on any divergence (`sync drift: … run sync_extracted.sh`).
- **Owned** (`target` only): the package copy is canonical; sync never touches it.

Plus two import guards: `forbid_atlas_reasoning_imports.py` (fail closed on *any*
`atlas_brain.reasoning` import) and `forbid_hard_atlas_imports.py` (an
`atlas_brain` import is only allowed inside a gated `try/except` or env branch, not
at module top). The general lesson: **when you have two copies of the truth, a
byte-level audit is the only thing that keeps them honest.**

---

## 7. The enforcement funnel: four layers, fastest first

The same checks run at progressively more expensive stages, so failures are caught
as early (and cheaply) as possible:

1. **Local bundle** — `bash scripts/local_pr_review.sh` runs the whole mechanical
   suite (plan shape, diff size, file claims, MCP-doc counts, ASCII, plan/code,
   session drift, whitespace) before GitHub runs anything. Plus advisory
   **cross-layer caller hints** (`audit_cross_layer_callers.py`) that surface
   non-diff files referencing your changed symbols, so shared-function changes get
   caller-layer tests.
2. **Pre-push git hook** (optional, installable) — runs the same bundle on `git
   push`. Bypass is explicit (`ATLAS_SKIP_LOCAL_PR_REVIEW=1`), never silent.
3. **A separate local reviewer session** — judgment review of plan + diff *before*
   the GitHub PR opens.
4. **GitHub Actions** — 20 workflows; the same wrapper plus per-product check
   suites, gated by path filters. CI is the *final* enforcement layer, not the
   first reviewer.

And the governing docs the auditor reads before any of it: `BUILD_SPEC.md`
(priorities + definition of done), `CANONICAL.md` (which implementation is the real
one — "one canonical per component, deprecated = don't touch"), `INTEGRATION_MAP.md`
(what's wired to what — "no floating code"), `CONTEXT.md` (known debt).

---

## 8. Within-session economy: judgment in main, lookup to subagents

To stretch the token budget without pushing judgment to a model that can't make it
(`AGENTS.md §5`): **reasoning and synthesis stay in the main session; pure
retrieval fans out to parallel subagents.** "Where is the schema?" → delegate.
"Is this schema right?" → do it yourself. Independent retrievals fire as parallel
subagents in one message so the main session waits once for N answers, not N times.

---

## Why it works — the principles worth stealing

1. **Mechanize every repeated mistake.** A failure that cost a review cycle becomes
   a script with fixture tests, or a line in the contract. Speed compounds because
   the same bug is never paid for twice.
2. **The plan is the contract.** Plan-first + a fixed 7-section shape + a LOC cap
   makes every PR reviewable in minutes and makes "scope creep" a detectable,
   failing condition.
3. **Separate the builder from the reviewer.** Independent re-derivation catches
   what self-review can't.
4. **Design for compaction.** Assume the agent's context will be summarized at the
   worst moment; keep sessions short, stop after milestones, and externalize state
   to disk with a recovery prompt ready.
5. **Fail closed, surface drift.** Audits report unfamiliar input as drift; secrets
   and per-tenant creds fail closed; detectors ship with proof their failure branch
   fires. Silent green is the enemy.
6. **Catch it as early as it's cheap.** The same gate runs local → hook → reviewer
   → CI, so the expensive layers rarely have to.

The net effect: the operator can run four AI sessions in parallel and move *fast*,
because the system mechanically refuses to let any of them break the shipping
product — and every new way one *could* break it gets turned into one more gate.
