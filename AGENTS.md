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

### 3e. Auditors must surface, never silently skip

This rule applies to any mechanical audit script under `scripts/`
(or anywhere else): when the auditor encounters input it doesn't
recognize, the default behavior is **report DRIFT with a clear
message**, not "skip and pretend nothing happened."

Four real cases caught by Copilot reviewers on PRs #483 / #484 /
#485 -- all silent skips that should have been DRIFT:

| What | Why it silently passed |
|---|---|
| `audit_mcp_tool_names_match_docs.py` silently dropped `### <Name> MCP Server` headers whose name wasn't in `HEADER_TO_FILE`. A renamed or newly added server would disappear from coverage. | `if name not in HEADER_TO_FILE: continue` |
| `audit_mcp_port_assignments.py` silently dropped env-var lines whose normalized name wasn't in `NAME_NORMALIZER`. | `if norm is None: continue` |
| `audit_mcp_port_assignments.py` exited 0 even when ports in `MCPConfig` had no claim in CLAUDE.md (missing-in-doc was not surfaced). | No `set(truth) - documented` check at the end. |
| `audit_mcp_port_assignments.py` `ENV_VAR_LINE` regex `[A-Z_]+` rejected the "2" in `ATLAS_MCP_B2B_CHURN_PORT`, silently dropping the entire line. | Regex without a digit-fixture; the class didn't cover real input. |

**Anti-pattern:**

```python
norm = NAME_NORMALIZER.get(env_name)
if norm is None:
    continue        # silently drops a valid claim
```

**Right shape:**

```python
norm = NAME_NORMALIZER.get(env_name)
claims.append((line_no, norm or env_name, port, "env"))
# main() then renders unknown names as DRIFT/UNKNOWN, not as skipped.
```

If you genuinely need a safe-skip (e.g., unrelated markdown tables
that happen to share a row shape with MCP port table rows), say
so in a comment and name the specific false-positive risk:

```python
# Markdown-table style: any "| <text> | <4-5-digit> | ..." row
# could be an unrelated table elsewhere in the doc, so we admit
# only rows whose first cell normalizes to a known server.
if norm is None:
    continue
```

A reviewer should be able to read the comment and decide whether
the skip is intentional. If you can't articulate the false-positive
risk in one sentence, the skip is wrong.

### 3f. Auditors ship with fixture tests

Every `scripts/audit_*.py` ships with a sibling
`tests/test_audit_<name>.py` that exercises at least three cases:

1. **Happy path.** Known-good input, expected exit 0 / OK output.
2. **At least one parser-specific negative case.** The fixture must
   include inputs the parser is supposed to handle but historically
   has not. For `audit_mcp_port_assignments.py` the fixture must
   include `ATLAS_MCP_B2B_CHURN_PORT=8062` (digit in name) and
   assert the auditor matches it. A regex without a digit-fixture
   is a regex that hasn't been thought about.
3. **Pathological input that should be rejected.** Absolute path,
   `..` traversal, malformed header, empty section, "Out of scope"
   heading masquerading as "Scope".

The audit script's `main()` is the contract; the fixture tests
**lock the contract in place** so a future "small tweak" can't
silently regress to silent-skip behavior. Without fixtures, a regex
bug like the `ENV_VAR_LINE` digit miss can ship without anyone
noticing for weeks.

If you're touching an audit script and there isn't a sibling
`test_audit_<name>.py`, add the fixture file in the same slice.

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

---

## 7. Bootstrapping a fresh reviewer session

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
