# PR-Agents-Md-Routing-Section

## Why this slice exists

The user is hitting Claude Pro weekly token limits ~4 days in. After
reviewing `imkunal007219/claude-coworker-model` (a toolkit that
delegates bulk reads + boilerplate to a cheap worker LLM), the
takeaway was that *the routing principle* is the leverage, not the
specific toolkit -- Claude Code already ships `Agent` subagents
that do the same shape for free.

`AGENTS.md` is the contract both builder and reviewer sessions
work from. It already encodes plan-doc shape, reviewer verdict
shape, and the multi-session workflow, but it does not say
**within a session, when to route to a subagent vs the main
model.** A future session reading `AGENTS.md` cold has no
explicit guidance, so the routing decisions are ad hoc and
inconsistent across sessions -- which directly correlates with
token burn.

This slice adds that routing guidance as a new top-level section.

## Scope (this PR)

1. Add `## 5. Within-session agent routing` to `AGENTS.md` with:
   - The two-question decision heuristic (in-session edit? needs
     judgment? -> main).
   - A six-row shape table covering the common cases.
   - A parallelism rule (orthogonal retrievals fire concurrently).
   - The Kimi-worker relationship: if the worker is ever installed,
     it slots in as a cheaper retrieval channel for single-file
     deep reads; everything else is unchanged.
   - Routing anti-patterns (don't ask subagents for judgment;
     don't sequence orthogonal Explore calls; don't Explore a
     file you're about to edit).
2. Renumber existing sections: 5 -> 6 (Anti-patterns), 6 -> 7
   (References), 7 -> 8 (Bootstrap).
3. File this plan doc per the AGENTS.md contract this PR is
   extending.

### Files touched

- `AGENTS.md`  (+~75 / -~3)
- `plans/PR-Agents-Md-Routing-Section.md`  (this file)

## Mechanism

The new section sits between **4. Reviewer workflow** and (formerly)
**5. Anti-patterns**, now **6**, because the routing rule applies to
both workflows and the anti-patterns section can then naturally
extend with routing-specific don'ts (added to the new section,
not duplicated in 6).

Only one internal cross-reference exists in the file (the
bootstrap block citing "sections 2a + 4d"). Both anchors are below
the insertion point but at section numbers <= 4, so the renumber
does not touch them.

## Intentional

- **New section gets its own number rather than slotting under
  Builder workflow (3) or Reviewer workflow (4).** The routing
  rule applies to both, so attaching it to one would mislead a
  reader of the other. A standalone section is the honest shape.
- **Kept the section short.** The user asked for an *intelligent*
  routing rule, not a textbook. The decision table is six rows;
  the section is ~75 lines including code fences. A longer
  treatise would make it less likely to be re-read.
- **Did not add routing guidance to CLAUDE.md as well.** The user
  explicitly said "add to our agents.md file." If we want a
  pointer in CLAUDE.md later, that's a follow-up slice.
- **Kept the Kimi-worker subsection forward-looking, not
  prescriptive.** The user hasn't decided yet whether to install
  the worker. The section says "if installed, here's how it slots
  in" rather than "go install Kimi." Reviewer-bait if we'd done
  the latter.
- **Renumbered 5->6, 6->7, 7->8 rather than inserting as 4.5 or
  using a sub-section under 4.** Whole-number sections are the
  existing convention.

## Deferred

- **Pointer from `CLAUDE.md` to AGENTS.md section 5.** Worth
  adding once the user decides if they want to commit to the
  routing rule; doing it now is presumptuous.
- **Kimi worker install + CLAUDE.md.template adoption.** The user
  said "just review for now" on the toolkit; this PR encodes the
  routing principle without committing to the toolkit.
- **Updating the bootstrap block** in section 8 to mention the
  new routing rule. Skipped to keep the diff small; the bootstrap
  prompt already says "Read AGENTS.md at the repo root" which
  picks up the new section automatically.

## Verification

1. `grep -nE "^## [0-9]" AGENTS.md` shows sections 1-8 in order,
   no gaps.
2. `grep -nE "section[s]? [0-9]" AGENTS.md` shows the one existing
   cross-ref ("sections 2a + 4d") is unchanged and still valid.
3. The new section 5 references no other section by number, so
   future renumbers won't break it.
4. CI: Vercel preview deploy (the only configured check on doc
   PRs) is expected green.

## Estimated diff size

`AGENTS.md`: ~+75 / -3 across one section insert + three line-level
renumbers.
`plans/PR-Agents-Md-Routing-Section.md`: ~+115 (this file).

Well under the 400 LOC soft cap.
