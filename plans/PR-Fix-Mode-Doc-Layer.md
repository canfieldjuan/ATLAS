# PR-Fix-Mode-Doc-Layer

## Why this slice exists

Fix loops -- iterating on red CI or review comments on an already-open PR -- are
where agent sessions (Codex, Claude Code) burn the most time and tokens: broad
exploration, edits to files outside the real failure source, and re-orientation
after every context compaction. Issue #1714 captured a full "PR fix mode" design
(a model-agnostic core plus Codex and Claude Code enforcement layers). This
slice lands only the **doc layer** of that design -- the part that is tracked in
git, benefits both agents immediately, needs no `.gitignore` change, and weakens
no existing gate. The deterministic enforcement layer (`.claude/` hooks,
`/fix-mode` skill, `.gitignore` narrowing, Codex `--max-files`) is deferred to
follow-ups under #1714.

## Scope (this PR)

Ownership lane: process/agent-guidance
Slice phase: Production hardening

Documentation only. Three tracked files gain a fix-loop rule, a reusable baton
block, and a compaction-preservation instruction. No code, no config, no gate
change.

### Files touched

- `AGENTS.md`
- `CLAUDE.md`
- `docs/SESSION_STATE_TEMPLATE.md`
- `plans/PR-Fix-Mode-Doc-Layer.md`

### Review Contract

Acceptance criteria:

- [ ] `AGENTS.md` gains a `### 3l. PR fix mode` subsection between §3k and §4,
      cross-referencing §3k (root-cause) and §4a.1 (no auto-loop) rather than
      restating them.
- [ ] `docs/SESSION_STATE_TEMPLATE.md` gains a `## PR Fix Mode (active fix loop)`
      block inside the template fence with the eight baton fields, plus a
      Resume-Checklist bullet gating edits on the allowed-files set.
- [ ] `CLAUDE.md` gains a `## Compact Instructions` section telling the
      summarizer to preserve the baton fields verbatim across compaction.
- [ ] No code, config, workflow, or `.gitignore` change; no existing gate
      relaxed.

Affected surfaces: builder/agent process docs only.

Risk areas: a documented rule with no enforcement is advisory -- mitigated by
landing it on the read-first docs (AGENTS.md, the SESSION_STATE template loaded
into the gitignored baton, and CLAUDE.md) and tracking the enforcement follow-up
in #1714.

Reviewer rules triggered: R1.

## Mechanism

The baton reuses the existing gitignored `SESSION_STATE.local.md` (mandated by
AGENTS.md §3a.1), so there is no new file or convention -- the template just
grows a `PR Fix Mode` block that builders copy in. AGENTS.md §3l states the
fix-loop discipline (declare failure source + allowed-files + max-files budget
before editing; widening the set is a §3k root-cause decision; disposition
findings in one pass per §4a.1; keep the baton current as the compaction
handoff). CLAUDE.md `## Compact Instructions` is read by Claude Code at
compaction time and documents intent for any agent: preserve the baton fields
verbatim so a post-compaction resume continues instead of re-exploring.

## Intentional

- Doc layer only -- no enforcement code. The value is the shared rule + baton
  shape, usable by both agents today.
- §3l cross-references §3k and §4a.1 instead of duplicating them, to avoid drift
  between the rule and its gates.
- The baton lives in the already-gitignored session-state file, so it never
  enters a PR diff and survives compaction as a read-first doc.

## Deferred

Tracked in issue #1714 (not this PR):

- `.gitignore` narrowing (`.claude/`) to make Claude Code config shareable.
- `.claude/settings.json` PreToolUse deny + SessionStart/PreCompact baton
  re-hydration hooks, and a `/fix-mode` skill entrypoint.
- Codex `--max-files` budget enforcement in
  `audit_plan_doc_files_touched.py`.

## Parked hardening

None.

## Verification

- Run `scripts/audit_plan_doc.py` against this plan -- 7 required sections
  present and ordered.
- Run `scripts/pre_push_audit.sh` -- plan files-touched and diff-size audits
  match the real diff (no MISSING/EXTRA, Total within tolerance).
- Run `scripts/local_pr_review.sh` -- full local review bundle green before
  push.
- Manual read-through: §3l renders between §3k and §4; the template block sits
  inside the fenced template; the CLAUDE.md section is at end of file.

## Estimated diff size

| File | LOC |
|---|---:|
| `AGENTS.md` | 24 |
| `CLAUDE.md` | 14 |
| `docs/SESSION_STATE_TEMPLATE.md` | 15 |
| `plans/PR-Fix-Mode-Doc-Layer.md` | ~110 |
| **Total** | **~163** |
