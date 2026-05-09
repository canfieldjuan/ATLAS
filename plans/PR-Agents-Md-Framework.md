# PR: introduce `AGENTS.md` at the repo root

## Why this slice exists

The reviewer Claude session has been auditing PRs against a
"AGENTS.md framework" with 7 named sections (Why / Scope / Mechanism
/ Files touched / Intentional / Deferred / Verification) and a
verdict shape (BLOCKER / MAJOR / NIT / LGTM). Reviewer comments
across PRs #396, #397, #398 explicitly cite this framework -- but
no `AGENTS.md` exists in the repo. The convention is in both
sessions' heads, not a checked-in spec.

The user flagged this when reviewing PR #399: *"Let's make sure
we're using the agents.md file/prompt for the audits."* Concretely,
the gap is that:

1. The builder session has the plan-doc skeleton in muscle memory
   from prior PRs but no spec to point at when starting fresh.
2. The reviewer session works from convention; if the convention
   drifts (e.g. the user fork-spawns a new reviewer), there's no
   ground truth.
3. Future contributors (or a fresh session after compaction) have
   nothing to read.

This PR makes the convention explicit so both sessions audit
against the same checked-in contract.

## Scope (this PR)

Documentation only. Two files.

### Files touched

1. `AGENTS.md` (new, repo root)
   - Defines the PR-shape contract: plan-doc sections, PR body
     mirror, commit message shape, diff budget, branch naming,
     draft-until-LGTM gate.
   - Defines the reviewer verdict shape (BLOCKER / MAJOR / NIT /
     LGTM) and a verification template.
   - Defines builder + reviewer workflows (plan-first, audit
     gauntlet, manifest discipline, independent verification, NIT
     discipline).
   - Lists anti-patterns ("drive-by formatting changes," "plan in a
     follow-up commit," etc.).
   - Cross-references the existing `AUDITOR_PROMPT.md` (broader
     canonical / integration / scope auditor) so the two prompts are
     complementary, not duplicative.
   - **Section 7 -- Bootstrapping a fresh reviewer session.**
     Copy-pastable prompt block that seeds a new reviewer Claude
     session with the framework, the audit gauntlet, the package's
     manifest discipline, and a recent-PR summary. The user paste
     the block when the prior reviewer session dies; the new session
     arrives audit-ready.

2. `plans/PR-Agents-Md-Framework.md` (this file).

## Mechanism

`AGENTS.md` lives at the repo root. Both Claude Code sessions read
the repo root as part of their bootstrap (the project's `CLAUDE.md`
is also there), so a checked-in `AGENTS.md` is automatically in-scope
for both sessions on the next session start.

The file is descriptive of the workflow we've already converged on
across the OptionA / Consistency / SalesBriefs / LandingPage / Audit
batches and the recent alternatives (`PR-Blog-Topic-Per-Call`,
`PR-Describe-Control-Surfaces-Cache`, `PR-Campaign-Config-V2`). It
codifies the existing pattern; it does not introduce new process.

## Intentional

- **Separate file from `AUDITOR_PROMPT.md`.** That file is the
  cross-cutting auditor prompt (P0/P1/P2 / canonical / integration /
  debt). `AGENTS.md` is the per-PR shape contract. They're
  complementary; one points at the other in its References section.
- **Descriptive, not prescriptive.** The framework documents what
  we do, not what we should do. Keeps the doc honest -- if the
  workflow shifts, the file gets updated, not bypassed.
- **Anti-patterns section listed at the end, not gated up front.**
  The doc is meant to teach the workflow first; the anti-patterns
  are reference material for when something goes wrong.
- **No tooling changes.** This PR doesn't add a CI check that "PR
  body matches AGENTS.md framework." That can come later if the
  pattern slips. Today the human + reviewer-Claude eyeballs are the
  gate.

## Deferred

- A CI check that the PR body has the required sections. Mechanical
  but not load-bearing today.
- A `plans/_template.md` skeleton that builders copy when starting a
  slice. Could land in a follow-up if useful.
- Migration of `AUDITOR_PROMPT.md` content into `AGENTS.md`. The
  two have different scopes (cross-cutting auditor vs. per-PR
  contract); merging would conflate concerns.
- Any change to PR #399 (currently in-flight). PR #399 already
  follows the framework; this slice just makes the framework
  explicit for future PRs.

## Verification

- `cat AGENTS.md` -- file present at repo root, full doc renders.
- `grep -c "^## " AGENTS.md` -- top-level sections present.
- `bash scripts/validate_extracted_content_pipeline.sh` -- clean
  (no Python touched).
- `python scripts/audit_extracted_standalone.py --fail-on-debt` --
  Atlas runtime import findings: 0.
- `bash scripts/check_ascii_python.sh` -- clean (markdown not in
  scope).
- `git diff main --stat` -- 2 files, all additions.

## Estimated diff size

- `AGENTS.md`: ~250 LOC.
- `plans/PR-Agents-Md-Framework.md`: ~110 LOC.

Total: ~360 LOC. Within the 400 LOC PR target. All documentation;
no code change.
