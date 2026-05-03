# Extraction Coordination

Last updated: 2026-05-03T22:02Z by claude-2026-05-03

State-of-the-world for the multi-product extraction effort. **This file is the index and protocol; live state lives in per-section files under `coordination/`.** Read this end-to-end at session start before doing substantive work, then check each section file you intend to touch.

## Why split

A single shared file generated a merge-conflict every PR. The bottleneck was line-level diff over what was logically per-row state. Sessions touching different *rows* of the same *table* still conflicted because git operates on lines, not table semantics. Splitting by section means edits to different sections no longer conflict at the file level.

## Per-section files

| Section | File | What changes here |
|---|---|---|
| Per-product state | [`coordination/state.md`](coordination/state.md) | Per-product phase, most-recent merged PR, active PRs, next milestone, hot zone. Updated when a PR merges or phase advances. |
| In-flight PRs | [`coordination/inflight.md`](coordination/inflight.md) | Open PRs we are coordinating around. Add a row before opening a PR; drop the row when it merges. |
| Upcoming queue | [`coordination/queue.md`](coordination/queue.md) | Slices not yet started, with dependencies. Set Owner when claiming. |
| Decisions log | [`coordination/decisions.md`](coordination/decisions.md) | Append-only chronological log. Never edit historical entries; supersede with newer entries. |
| Open questions | [`coordination/open_questions.md`](coordination/open_questions.md) | Active questions waiting on an owner or decision. Resolutions move to `decisions.md`. |

The team is one human (`@canfieldjuan`) plus AI sessions. Owner column uses GitHub usernames for human work and agent-stamped session IDs for AI work (`{agent}-YYYY-MM-DD[-suffix]`, e.g. `claude-2026-05-03`, `codex-2026-05-03`). The first session for an agent on a calendar day is unsuffixed; subsequent same-agent sessions claim alphabetical suffixes from `-b` (`claude-2026-05-03-b`, `codex-2026-05-03-b`, ...) in the same commit that claims a slice. Timestamps in coordination files use ISO 8601 UTC (`YYYY-MM-DDTHH:MMZ`).

**Active session aliases (2026-05-03)** -- for conversational shorthand: `A` = `claude-2026-05-03-b` (PR #81 authoring / PR-A0 claim, PR-A1, PR-A2, PR-A1.5), `B` = `codex-2026-05-03` (PR #81 review, PR #82 coordination update, PR-B1 quality-gate audit, PR #88 conflict resolution, PR #89 conflict resolution), `C` = `claude-2026-05-03` (PRs #79, #80, #82, #86, #88, this split). Aliases re-anchor each calendar day. Agent-date IDs remain canonical in all tables; aliases are for in-conversation reference only.
_(The Per-product state, In-flight PRs, Upcoming queue, Decisions log, and Open questions sections that previously lived here have moved to per-section files under `coordination/`. This split lands in the same commit; see the section table above for links. New content from origin/main has been applied to the appropriate per-section files during this merge resolution.)_

---

## Session protocol

1. **At session start**: read this file end-to-end, then skim each `coordination/*.md` file before opening files you intend to touch.
2. **Before opening a PR**: add a row to `coordination/inflight.md` with your owner ID and the files you'll touch.
3. **Before starting code on a queued slice**: claim it in `coordination/queue.md` (set Owner) so a parallel session does not pick the same one.
4. **After a PR merges**: update `coordination/state.md` (most recent PR, next milestone), drop the row from `coordination/inflight.md`, log any decisions made during review in `coordination/decisions.md`.
5. **When a decision lands**: append to `coordination/decisions.md` with the date. Never edit historical entries; supersede with a newer entry instead.
6. **Update the "Last updated" stamp** on every file you touch (each `coordination/*.md` carries its own stamp; this file carries the index stamp). ISO 8601 UTC: `YYYY-MM-DDTHH:MMZ`. Stamps must be monotonic relative to the previous value: write `max(now, last_stamp + 1 minute)`. If the prior stamp is in the future relative to your real clock (clock drift, estimation), still bump past it -- the audit log must never regress.
7. **Tie-breaker on simultaneous claims**: if two sessions claim the same slice within minutes, last commit to that file wins; the loser pivots to a different slice or negotiates in PR comments before opening a competing PR.
8. **Forgive-and-claim**: if you opened a PR without first adding a row, add the row before requesting review. Skipping the claim once is not punishable; abandoning the protocol is.

---

## Conventions

- **Owner format** -- GitHub username (`@canfieldjuan`) for human work; `{agent}-YYYY-MM-DD[-suffix]` for AI session work, e.g. `claude-2026-05-03`, `codex-2026-05-03-b`.
- **Unknown-owner fallback** -- if an in-flight PR's Owner is `(unknown -- confirm)`, treat its listed file paths as locked until the owner is filled in. Safer default than racing on an unattributed PR.
- **PR title verbs** -- match the established pattern: `Add X`, `Own X`, `Route X through Y`, `Document X`, `Harden X`, `Refresh X`. The verb signals intent (Phase 1 add vs Phase 2 ownership vs Phase 3 decoupling vs docs).
- **Boundary / consolidation audit docs** -- land in `docs/extraction/<slug>_audit_<date>.md` (with optional `_boundary` infix for first-PR boundary audits) BEFORE the first scaffold PR. `<slug>` is the slice or topic, not the full product name (examples: `reasoning_boundary_audit_2026-05-03.md`, `quality_gate_boundary_audit_2026-05-03.md`, `cost_closure_audit_2026-05-03.md`, `evidence_temporal_archetypes_audit_2026-05-03.md`). PR #79 is the template.
- **Per-product status** -- STATUS.md inside each `extracted_*/` folder is the product-internal state. The `coordination/state.md` file is the cross-product state. Don't duplicate detail; link.

## What this doc is NOT for

- Detailed product roadmaps -- those live in each product's `STATUS.md` or boundary audit doc.
- Architecture decisions specific to one product -- capture those in the relevant boundary audit or README.
- A real-time PR mirror -- `gh pr list` is the source of truth for what's open. The coordination files track intent and ownership for in-flight work we're coordinating around.
- Long discussion threads -- keep this scannable. Conversations belong in PR descriptions and review comments; only the *outcome* lands in `coordination/decisions.md`.
