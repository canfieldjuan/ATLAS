# AGENTS.md — Atlas multi-agent workflow

Atlas uses **two coordinated Claude Code sessions** for non-trivial
work:

1. **Builder session** — drafts the plan, writes the code, opens the PR.
2. **Reviewer session** — audits each PR independently and posts a
   verdict (BLOCKER / MAJOR / NIT / LGTM).

This file is the contract both sessions work from. The auditor
(prompt at `AUDITOR_PROMPT.md`) handles cross-cutting integration /
canonical / scope checks; this file defines the **PR-shape contract**
that lets the reviewer give the builder a clean LGTM.

---

## 1. PR shape

Every non-trivial change ships as a single PR with the following
artifacts:

### 1a. Plan doc (`plans/PR-<Slice-Name>.md`)

Required sections, in this order:

| Section | Purpose |
|---|---|
| **Why this slice exists** | What's broken / what's missing / what audit item this closes. Tie to a prior plan, audit finding, or a concrete user request. |
| **Scope (this PR)** | The narrow surface this PR touches. Numbered list of intent. List of files in a "Files touched" subsection. |
| **Mechanism** | Short prose (and code stub if helpful) explaining *how* the change works -- enough that the reviewer doesn't have to reverse-engineer it from the diff. |
| **Intentional** | Things that look wrong but aren't -- explicit trade-offs and rejected alternatives ("no `warnings.warn` shim because ..."). Saves reviewer cycles. |
| **Deferred** | Things explicitly punted to a follow-up slice. Each item should name the future PR or describe what would unlock it. |
| **Verification** | The specific commands the builder ran locally + their pass counts. Reviewer reproduces. |
| **Estimated diff size** | LOC budget; flag if approaching 400 LOC. |

### 1b. PR body

Mirror the plan-doc framing in the PR description:

```
Plan: plans/PR-<Slice-Name>.md

<one-paragraph why>

## Intentional
- ...

## Deferred
- ...

## Verification
- ...

## Diff size
N files, +X / -Y
```

### 1c. Commit message

Same `Plan: ...` lead line + Intentional / Deferred sections as the
PR body. Squash-merge collapses to one canonical commit at merge time.

### 1d. Diff budget

Target **<400 LOC** per PR. Soft cap; over-budget PRs ship if the
slice is genuinely indivisible, but the plan doc must justify the
overage in **Why this slice exists**.

### 1e. Branch naming

`claude/pr-<slice-name>` for builder branches.
`claude/<topic>` for non-PR scratch.

### 1f. Open as draft

Open the PR as **draft** until reviewer LGTM. Mark ready for review
just before merge.

---

## 2. Reviewer verdict shape

The reviewer comments **once per push** with a verdict at one of these
levels:

| Level | Meaning | Builder action |
|---|---|---|
| **BLOCKER** | Correctness, security, contract break, or CI red. Must fix before merge. | Fix or push back with rationale. |
| **MAJOR** | Architectural / scope / pattern concern. Strong recommendation but not auto-block. | Fix in this PR if the fix is small; otherwise discuss before deferring. |
| **NIT** | Style, naming, comment polish. Skip-worthy. | Apply if 1-line; skip otherwise. The reviewer should mark NITs as skip-worthy explicitly. |
| **LGTM** | All gates green, no remaining concerns. | Merge. |

### 2a. Reviewer's verification template

The reviewer should produce something like:

```
**Verification (independent):**
1. <claim from PR description> -- verified via <command>
2. <invariant from Mechanism> -- confirmed at <file:line>
3. ...

**Plan-doc compliance:** Why / Scope / Mechanism / Files touched /
Intentional / Deferred / Verification -- matches AGENTS.md framework.

**Defensible trade-offs (no action needed):**
- <decision> -- <why it's the right call>

**<N> NITs (skip-worthy):**
1. ...

LGTM. (or: BLOCKER -- ...)
```

### 2b. CI gate

CI must be green before LGTM. If CI is red on a transient failure
(flaky test, infra), the reviewer can call that out separately and
not block.

---

## 3. Builder workflow

### 3a. Plan first

Open `plans/PR-<Slice-Name>.md` and write the full plan doc **before**
any code change. The plan is the contract; the code is the
implementation of the contract.

If the plan changes mid-implementation (you discovered something the
plan missed), update the plan doc in the same commit. The plan and
code ship together.

### 3b. Per-package guardrails

Touching a package under `extracted_*/` requires the package's audit
gauntlet locally before push. For `extracted_content_pipeline`:

```bash
bash scripts/validate_extracted_content_pipeline.sh
python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
python scripts/audit_extracted_standalone.py --fail-on-debt
bash scripts/check_ascii_python.sh
bash extracted/_shared/scripts/sync_extracted.sh extracted_content_pipeline   # if any synced files changed
```

CI runs `scripts/run_extracted_pipeline_checks.sh`; locally is faster
to triage.

### 3c. Tests

Each PR ships its own tests. Acceptable test patterns:

- **Unit-level**: pure validators, parsers, helpers. Live in
  `tests/test_<package>_<module>.py`.
- **Integration-level**: services + ports with fakes. Live in
  `tests/test_<package>_<service>.py`.
- **Smoke**: thin wrappers that just check imports / wiring.

Locked-in regression tests for deferred follow-ups should name the
future slice in their docstring (e.g. *"after PR-Foo-V2 lands this
test is removed"*) so the test's lifetime is explicit.

### 3d. Working with the manifest

Files listed in `<package>/manifest.json` under `owned` are
package-canonical -- the sync script does not overwrite them. Files
mapped from `atlas_brain/...` are the inverse: edits go to the
`atlas_brain/` source, then the sync script propagates. If unsure
which side a file lives on, run:

```bash
grep -B2 '"target": "<path>"' <package>/manifest.json
```

A `source` line means it's synced; absence (just a `target`) means
it's owned.

---

## 4. Reviewer workflow

### 4a. Independent verification

Don't trust the PR description's claims; reproduce them. The
reviewer should:

1. Re-run the named verification commands.
2. Spot-check the plan's invariants at the actual file:line pointed
   to in the diff.
3. Sweep for missed call sites with grep patterns more reliable than
   the PR's claim (multi-line constructions, kwargs split across
   lines, etc.).

### 4b. Verdict frugality

Post **one** review per push. Don't comment on the PR while CI is
in-flight unless asked. Don't rubber-stamp -- a bare "LGTM" with no
verification is worse than no comment.

### 4c. NIT discipline

NITs should be marked skip-worthy explicitly when they are. The
builder applies 1-line NITs; ignores style/naming/comment NITs that
require a follow-up commit unless the reviewer specifically calls
out "this should be fixed."

### 4d. Audit checklist

Before LGTM, the reviewer confirms:

- [ ] CI green (extracted-checks ✅, Vercel ✅).
- [ ] Plan doc has all 7 required sections.
- [ ] Diff size matches the plan's estimate (or the overage is
      justified in **Why**).
- [ ] No regressions in the named test sweep.
- [ ] No drift from the plan's stated scope (no scope creep, no
      "while I was at it" cleanups beyond the slice's contract).
- [ ] Defensible trade-offs are explained in **Intentional**.
- [ ] Deferred items have a clear next-PR home.

---

## 5. Anti-patterns

Things that should **never** appear in a PR or review:

- **Drive-by formatting changes** unrelated to the slice. Format-only
  diffs ship as their own slice if needed.
- **Plan doc that arrives in a follow-up commit.** Plan and
  implementation ship together.
- **"While I was here..." cleanups** that aren't in the plan. Add a
  Deferred item and move on.
- **Bypassing CI with `--no-verify`** unless the user explicitly
  authorizes.
- **Reviewer running the builder's commands without spot-checking
  the diff.** A green test sweep doesn't prove the diff matches the
  plan.
- **Builder applying every NIT without judgment.** NITs marked
  skip-worthy are skipped. Apply only the 1-line / unambiguous ones.

---

## 6. References

- `AUDITOR_PROMPT.md` -- cross-cutting auditor prompt
  (canonical / integration / scope / debt). Run before any non-trivial
  build session.
- `BUILD_SPEC.md` -- what Atlas is, P0/P1/P2 priorities, definition
  of done.
- `CANONICAL.md` -- which implementation is the real one.
- `INTEGRATION_MAP.md` -- what's wired to what.
- `CONTEXT.md` -- session notes, known debt.
- `CLAUDE.md` -- project-level Claude Code guidance.
- `plans/` -- per-slice plan docs (one per PR).
