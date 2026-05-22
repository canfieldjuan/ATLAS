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

### 1f. Open ready for review

Open the PR as **ready for review** by default. Do not open draft PRs
unless the operator explicitly asks for a draft. Automated review tools
do not review draft PRs, so draft mode burns review time and hides
feedback until the PR is manually marked ready.

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

### 3c. Local review before PR

Before opening or updating a PR, the builder runs the mechanical local
review bundle:

```bash
bash scripts/local_pr_review.sh
```

This is the fast path. It catches plan-shape, diff-size, file-claim,
MCP-doc, ASCII, plan/code, cross-session drift, and whitespace failures
before GitHub has to run anything. It also prints advisory cross-layer
caller hints for changed Python symbols so builder and reviewer can see
non-diff files that may need focused tests or inspection.

To make that automatic for this checkout, install the optional pre-push
hook:

```bash
bash scripts/install_local_pr_hook.sh
```

The installer refuses to overwrite unmanaged hooks unless `--force` is
passed. The installed hook can be bypassed intentionally with
`ATLAS_SKIP_LOCAL_PR_REVIEW=1 git push`.

After the mechanical bundle passes, hand the branch to a separate local
reviewer session for judgment review. The reviewer should read the plan
doc and diff locally, run any focused tests needed to verify the claims,
and return a verdict before the builder opens the GitHub PR.

For logic or shared-function PRs, the builder must read the cross-layer
caller hints from local review and either add focused caller-layer tests
or name why the referenced callers are unaffected. The hints are
advisory rather than blocking because outside references can be valid,
but silently ignoring them recreates the diff-only review gap.

GitHub Actions still runs the same wrapper after the PR opens. Treat CI
as the final enforcement layer, not the first reviewer.

### 3d. Tests

Each PR ships its own tests. Acceptable test patterns:

- **Unit-level**: pure validators, parsers, helpers. Live in
  `tests/test_<package>_<module>.py`.
- **Integration-level**: services + ports with fakes. Live in
  `tests/test_<package>_<service>.py`.
- **Smoke**: thin wrappers that just check imports / wiring.

Locked-in regression tests for deferred follow-ups should name the
future slice in their docstring (e.g. *"after PR-Foo-V2 lands this
test is removed"*) so the test's lifetime is explicit.

### 3e. Working with the manifest

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

### 3f. Auditors must surface, never silently skip

Mechanical audit scripts must report unfamiliar input as drift unless
the skip is explicitly justified in code. Silent skips make the audit
look green while the thing it was supposed to validate disappears from
coverage.

Recent examples this rule is meant to prevent:

| What | Bad shape |
|---|---|
| Unknown `### <Name> MCP Server` headings disappearing from MCP tool-name coverage. | `if name not in HEADER_TO_FILE: continue` |
| Port claims with names not in the normalizer disappearing from MCP port coverage. | `if norm is None: continue` |
| Ports in `MCPConfig` missing from docs without any missing-in-doc check. | Only compare documented rows. |
| Env-var regexes dropping real names with digits, such as `ATLAS_MCP_B2B_CHURN_PORT`. | `[A-Z_]+` without a digit fixture. |

Preferred shape:

```python
norm = NAME_NORMALIZER.get(env_name)
claims.append((line_no, norm or env_name, port, "env"))
# main() then renders unknown names as DRIFT/UNKNOWN, not as skipped.
```

Safe skips are allowed only when the false-positive risk is named:

```python
# Unrelated markdown tables can share this row shape, so admit only
# rows whose first cell normalizes to a known server.
if norm is None:
    continue
```

If the false-positive risk cannot be stated in one sentence, the skip
is probably wrong.

### 3g. Auditors ship with fixture tests

Every new `scripts/audit_*.py` should ship with
`tests/test_audit_<name>.py` in the same slice. The fixture set should
cover:

1. Happy path: known-good input produces the expected OK state.
2. Parser-specific negative case: real-looking input that used to be
   missed, such as `ATLAS_MCP_B2B_CHURN_PORT=8062`.
3. Pathological rejection: absolute path, `..` traversal, malformed
   header, empty section, or an "Out of scope" heading that must not
   satisfy "Scope".

The audit script's `main()` is the contract; fixture tests lock the
parser behavior so a future small regex tweak cannot silently regress
to false-green output.

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
- [ ] For shared-function PRs, cross-layer caller hints were inspected
      and the verdict names any caller-layer tests or unaffected
      references.
- [ ] No drift from the plan's stated scope (no scope creep, no
      "while I was at it" cleanups beyond the slice's contract).
- [ ] Defensible trade-offs are explained in **Intentional**.
- [ ] Deferred items have a clear next-PR home.

---

## 5. Within-session agent routing

**Reasoning stays in main; retrieval goes to a subagent. Synthesis
stays with whoever has to act on the answer (almost always main).**

Applies to both builder and reviewer sessions. The point: stretch
the weekly token budget without pushing judgment work to a model
that can't make judgment calls.

### 5a. The decision

Two questions before opening a file or kicking off a search:

1. *Will I edit this file in-session?* -> Main, direct `Read` (need
   exact line numbers).
2. *Does this need judgment* (quality, design trade-off,
   root-cause)? -> Main only.

If neither, route by shape:

| Shape | Where | Why |
|---|---|---|
| Read-only, >400 lines, no edits planned | `Explore` subagent | Pure retrieval; summary lands in main context, raw file does not |
| Reading 3+ files just to orient | `Explore` subagent | Width without depth -- the subagent's strength |
| "Find every caller of X" / "where is Y defined" | `grep`/`find` via Bash | Regex match, no LLM needed |
| Scaffold multi-file boilerplate (tests, configs, fixtures) | `general-purpose` subagent | Write-capable, separate context window |
| Architectural decision / debugging / refactor plan | Main only | Needs holistic judgment |
| Code review verdict | Main only | Verdict requires judgment, not a summary |

The boundary that matters most: **judgment vs lookup.** "Where is
the displacement edge schema?" -- lookup, delegate. "Is this
displacement edge schema right?" -- judgment, do it yourself.

### 5b. Parallelism

Independent retrievals run as parallel subagents in a single
message -- the main session waits once for N answers instead of N
times for one each. We used this pattern during the CLAUDE.md
refresh (three `Explore` agents in parallel mapped churn signals,
extracted packages, and planned products); without it the same work
would have cost N round-trips of main-context overhead.

### 5c. The Kimi worker model relationship

If a `claude-coworker-model`-style worker LLM is installed locally
(Kimi / DeepSeek / Ollama via OpenRouter), it slots in as a
**cheaper** retrieval channel for cases where an `Explore`
subagent is overkill -- one big file, no reasoning needed, no other
files to cross-reference. The decision table above is unchanged;
just add a row:

| Shape | Where | Why |
|---|---|---|
| Deep retrieval of one large file, no cross-refs | Worker LLM (if installed), else `Explore` | Worker is cheapest; `Explore` is the in-tree fallback |

The worker never replaces `Explore` for multi-file orientation or
the main session for judgment.

### 5d. Routing anti-patterns

- **Asking a subagent for a judgment call** ("is this design
  right?"). The subagent doesn't have full session context and the
  answer is just deferred judgment the main session has to redo.
- **Sequencing N orthogonal `Explore` calls** instead of firing
  them in parallel.
- **Using `Explore` on a <400-line file you're about to edit
  anyway.** Just `Read` it directly.
- **Letting a subagent compose the final user-facing answer.**
  Synthesis is a main-session job.
- **Routing exact-line edits through a worker.** Edits need a
  precise file:line citation; a summary won't have one.

---

## 6. Anti-patterns

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

## 7. References

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

---

## 8. Bootstrapping a fresh reviewer session

When the reviewer session is killed, expired, or otherwise needs to
be re-seeded, paste the block below into a fresh Claude Code session
(adjust the trailing PR list to match what's actually open). The
session will arrive with everything it needs to start auditing
immediately.

```
You are the reviewer Claude session for the canfieldjuan/atlas
repository. A separate builder session opens PRs; you audit them
and post one consolidated review per push with a verdict at:

- BLOCKER: correctness, security, contract break, or CI red.
- MAJOR: architectural / scope / pattern concern.
- NIT: style / naming / comment polish (mark explicitly skip-worthy
  when applicable).
- LGTM: all gates green, no remaining concerns.

Read AGENTS.md at the repo root before your first review. It
defines:
- The required plan-doc shape (Why / Scope / Mechanism / Files
  touched / Intentional / Deferred / Verification) at
  plans/PR-<Slice-Name>.md.
- The PR body / commit message conventions that mirror the plan.
- The reviewer verification template (sections 2a + 4d).
- The 400 LOC diff budget and how to handle overage.
- Anti-patterns that should never appear in a builder PR.

Read AUDITOR_PROMPT.md for the cross-cutting audit checks
(canonical / integration / scope / debt). Apply both lenses.

For each PR you review:

1. Reproduce the named verification commands from the PR body.
   Don't trust claims; re-run them. Spot-check the plan's
   invariants at the actual file:line cited in the diff.
2. Sweep for missed call sites with grep patterns more reliable
   than the PR's claim (multi-line constructions, kwargs split
   across lines).
3. Confirm CI is green (extracted-checks x2 + Vercel) before
   issuing LGTM.
4. Mark NITs as skip-worthy explicitly when they are. The builder
   applies 1-line / unambiguous NITs; ignores style/naming/comment
   NITs unless you specifically call out "this should be fixed."
5. Sign your review with: `_Generated by [Claude Code](
   https://claude.ai/code)_`.

The package under active iteration is `extracted_content_pipeline`.
Its audit gauntlet is:

```bash
bash scripts/validate_extracted_content_pipeline.sh
python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
python scripts/audit_extracted_standalone.py --fail-on-debt
bash scripts/check_ascii_python.sh
bash scripts/run_extracted_pipeline_checks.sh   # full CI mirror
```

The package's manifest at
`extracted_content_pipeline/manifest.json` distinguishes
package-owned files (entries with only a `target`) from
synced-from-`atlas_brain/` files (entries with both `source` and
`target`). Synced files cannot be edited in
`extracted_content_pipeline/` directly; the source lives in
`atlas_brain/`. The sync script propagates.

Recently merged context (so you're not starting cold):
- PR #396 PR-Blog-Topic-Per-Call: per-call topic kwarg threads from
  request.inputs through the dispatcher into the blog skill prompt.
- PR #397 PR-Describe-Control-Surfaces-Cache: GET
  /content-ops/control-surfaces hot path caches the static
  catalog payload at import; only execution flags are computed
  per request.
- PR #398 PR-Campaign-Config-V2 (BREAKING): drops the legacy
  `channel` field from CampaignGenerationConfig + the
  `or (self._config.channel,)` fallback. 4 in-tree wrappers
  normalize `channel` -> `channels` tuple before dataclass
  construction.
- PR #399 PR-Blog-Reasoning-Parity: brings
  BlogPostGenerationService to constructor parity with the other
  4 generators on the CampaignReasoningContextProvider port.
- PR #400 PR-Agents-Md-Framework: this file.

Open PRs as of this bootstrap: <fill in via the GitHub MCP tools or
mcp__github__list_pull_requests at session start>.

Standby for `<github-webhook-activity>` events; investigate each
in turn.
```

If the user prefers to keep AGENTS.md leaner, the reviewer prompt can
also live at `REVIEWER_BOOTSTRAP.md` and be invoked by a one-liner:
*"Read REVIEWER_BOOTSTRAP.md, then standby."* Either shape works;
pick one and keep it.
