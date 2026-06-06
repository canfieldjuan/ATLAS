# PR-Refresh-CLAUDE-Md

## Why this slice exists

`CLAUDE.md` was the entry-point Claude Code session-prep doc, but it
had drifted significantly from the codebase. The stale framing was
causing future sessions to arrive with a wrong mental model of what
Atlas actually is and what's shipped.

Concrete drift items found during the refresh:

- The vision section led with a home-automation pitch ("Hey Atlas
  turns off the TV") while the headline products are now B2B Churn
  Intelligence and Content Ops.
- The MCP server count was listed as 7; the repo has 9 (memory and
  scraper servers are missing from the doc).
- The architecture map covered ~5 packages; `atlas_brain/` has 35.
- Several tool counts were wrong on the headline products (corrected
  in the second push — see *Verification*).
- The 19-source review enum was listed as 16.
- The `AGENTS.md` multi-session PR workflow was not referenced from
  `CLAUDE.md` at all.
- The six `extracted_*` packages had no per-package wiring detail.
- There was no "Planned / In-Flight Work" section, so a fresh session
  couldn't tell what's shipped vs what's `BUILD_SPEC.md` P0.
- The Codex bot caught an additional `.mcp.json`-points-at-an-
  incomplete-personal-config issue on the second push; that's fixed
  in this PR by softening the claim rather than rewriting the file
  (see *Deferred*).

Codex (P2) also flagged that this slice itself ships without a plan
doc despite the new guidance requiring one. This file closes that
gap retroactively.

## Scope (this PR)

1. Rewrite `CLAUDE.md` so the opening framing matches what's shipped
   today.
2. Add deep sections for the headline products: B2B Churn
   Intelligence Pipeline, Content Ops, and Planned / In-Flight Work.
3. Expand the Extracted Packages section with per-package LOC,
   status, public surface, and wiring detail.
4. Document the `AGENTS.md` multi-session PR workflow, the testing
   conventions, the per-`extracted_*` validation gauntlets, the
   sub-projects, and the key conventions.
5. Correct numeric counts everywhere they appear (MCP tool counts,
   review-source count).
6. Soften the `.mcp.json` claim so agents don't follow an incomplete
   personal config.

### Files touched

- `CLAUDE.md`  (+~620 / −~135 across three commits)
- `plans/PR-Refresh-CLAUDE-Md.md`  (this file)

## Mechanism

Two parallel `Explore` agents mapped the actual repo: one for the
B2B churn pipeline (scrape → enrich → signals → displacement →
reports → webhooks; schema; calibration; parser-version telemetry;
the 17 b2b/ modules; the atlas-intel-ui routes) and one for the
extracted packages (LOC, status, public surface, wiring).  A third
agent surveyed `BUILD_SPEC.md`, `plans/`, `docs/*roadmap*.md`, and
`CONTEXT.md` to separate shipped from in-flight from planned-only.

Numeric counts came from direct `grep -c "@mcp.tool"` on each
server module after Codex's review caught miscounts in commit 2
(see *Verification*).

## Intentional

- **Doc-only PR; no production code changes.** The `.mcp.json` file
  is broken (personal hardcoded path, only 3 servers) but fixing it
  is a substantive change to a checked-in config file and is out of
  scope for a `CLAUDE.md` refresh.  Handled by softening the
  CLAUDE.md claim instead.  See *Deferred*.
- **Diff over 400 LOC.** The slice is genuinely indivisible — every
  staleness item is a small edit but they're concentrated in one
  file, and splitting "fix counts" / "add product sections" /
  "add AGENTS.md reference" into separate PRs would force readers
  through three half-states of an entrypoint doc.  Net diff is
  ~485 lines; the soft cap is per `AGENTS.md`.
- **Plan doc filed retroactively** rather than rewriting history.
  The work was a multi-session conversational refresh; squashing
  to add a plan doc up front would lose the Codex review trail
  that proved the count corrections (commit 3) and the
  `.mcp.json` correction (commit 4).
- **Kept the original "Design Principles" framing** rather than
  rewriting it product-first, because those principles still hold
  across all products (extensibility, provider-agnostic, single
  source of truth, plan-first, local processing, privacy).
- **Kept the existing per-server MCP detail sections.** They are
  still accurate; the fix was the tool counts in the header lines.

## Deferred

- **Rewrite `.mcp.json`** as a complete portable 9-server config.
  Owner: a follow-up PR.  Blocker for that PR: confirm with the
  human owner that the file is meant to be a checked-in template
  rather than a personal scratch.  Today's CLAUDE.md softens the
  claim so no agent follows the broken file.
- **Diagram updates** in CLAUDE.md.  The existing ASCII diagram is
  in the "What Atlas Is (Today)" preamble (kept) but is not yet
  product-flow-shaped.  A B2B-pipeline-shaped diagram exists in
  the new B2B Churn Intelligence section; an analogous one for
  Content Ops is deferred until E4 generators land.
- **Cross-checking every link** in CLAUDE.md against the actual
  files referenced.  The new doc cites `docs/intelligence_platform_roadmap.md`,
  `docs/consumer_intelligence_roadmap.md`, `docs/progress/gui_camera_audit.md`,
  `INTEGRATION_MAP.md`, `CANONICAL.md`, `BUILD_SPEC.md`,
  `CONTEXT.md`, and `AGENTS.md` — all confirmed to exist via
  `ls`, but their content was not re-verified line-by-line.
- **Auto-regenerating MCP tool counts** from the actual decorator
  grep so the doc can't drift.  Worth a small `scripts/audit_mcp_tool_counts.py`
  follow-up if drift recurs.

## Verification

The doc cannot be unit-tested, so verification was empirical:

1. **MCP tool counts.** After Codex's P2 catch on commit 2, every
   count was grep-verified:

   ```bash
   for f in atlas_brain/mcp/*_server.py; do
     printf '%-30s %s\n' "$(basename $f)" "$(grep -c '@mcp.tool' $f)"
   done
   # =>
   #   b2b_churn_server.py             0   (all under mcp/b2b/)
   #   calendar_server.py              8
   #   crm_server.py                  10
   #   email_server.py                 9
   #   intelligence_server.py         33
   #   invoicing_server.py            18
   #   memory_server.py               15
   #   scraper_server.py               5
   #   twilio_server.py               10

   for f in atlas_brain/mcp/b2b/*.py; do grep -c "@mcp.tool" $f; done \
     | awk '{s+=$1} END {print s}'
   # => 83
   ```

   All counts in `CLAUDE.md` post-commit-3 match.

2. **Review-source count.** `atlas_brain/services/scraping/sources.py`
   `ReviewSource` enum has 19 members.  CLAUDE.md says 19.

3. **MCPConfig port numbers.** Every port in the architecture map
   (8056–8064) matches `atlas_brain/config.py` `MCPConfig`.

4. **Module path existence.** All 9 paths in the MCP table
   (`atlas_brain.mcp.<name>_server`) and the 17 paths in the b2b
   table (`atlas_brain.mcp.b2b.<name>`) exist on disk.

5. **Extracted package LOC + file counts.** Verified by `find
   <pkg> -name '*.py' | wc -l` and `find <pkg> -name '*.py' | xargs
   wc -l | tail -1`; numbers match the table.

6. **Roadmap completion claims.** `docs/intelligence_platform_roadmap.md`
   and `docs/consumer_intelligence_roadmap.md` both have phase
   headers marked complete consistent with the doc's claims.

7. **CI.** Vercel preview deploy (the only CI configured on docs
   changes) is green at every push.

## Estimated diff size

- Commit 1 (`2125229`): +345 / −47   — initial refresh
- Commit 2 (`5f243a8`): +291 / −106  — churn-signal + extracted + planned-products focus
- Commit 3 (`55b00fd`): +16  / −13   — MCP tool count fix (Codex P2)
- Commit 4 (this push): +4   / −2    — `.mcp.json` claim softening (Codex P2) + this plan doc

Net diff vs `main`: ~+656 / −168 across one file (`CLAUDE.md`) plus
this plan doc.

The CLAUDE.md diff exceeds the 400 LOC soft cap; rationale in
*Intentional*.
