# PR-Tier-1-Audits

## Why this slice exists

PR #483 shipped the first three mechanical pre-push audits
(CLAUDE.md MCP tool counts, plan-doc shape, wrapper). The
remaining Tier-1 audits identified during PR #457's review pass --
the ones that would have caught real drift Codex caught me on --
are not yet in the repo. This slice closes that.

Three audits, each catching a different shape of drift seen this
session:

1. **Tool-name inventory drift.** Codex caught me when the
   `### Invoicing MCP Server (18 tools)` header was correct but
   the `Tools: ...` inventory list immediately below it still
   named only 15. PR #483's count auditor would not catch this
   -- it only verifies the header count.

2. **Manifest sync drift.** `sync_extracted.sh` blindly trusts
   `<pkg>/manifest.json`. Today nothing audits that every
   `source` path actually exists in `atlas_brain/`, every
   `target` exists in the package, or that synced pairs are
   byte-identical. Drift here is silent until a sync fails.

3. **Enum-count drift.** Codex caught me on "16 review sources"
   when the `ReviewSource` enum has 19 members. Same shape as
   the MCP tool count audit but for a different enum claim.
   Generalizable pattern.

This is recommendation #2 from the PR #457 token-savings analysis
(*"bash scripts for mechanical audits, zero LLM, biggest absolute
saver"*), Tier 1.

## Scope (this PR)

1. `scripts/audit_mcp_tool_names_match_docs.py` -- for each
   `### <Name> MCP Server` section in CLAUDE.md, gather all
   backticked snake_case identifiers (the claimed tool
   inventory), compare to actual `@mcp.tool` function names in
   the matching `atlas_brain/mcp/*_server.py` (sum
   `atlas_brain/mcp/b2b/*.py` for `b2b_churn`). Report names
   missing-from-doc and extra-in-doc per server.

2. `scripts/audit_extracted_manifests.py` -- for each
   `extracted_*/manifest.json`:
   - Every `mappings[i].source` exists in `atlas_brain/`.
   - Every `mappings[i].target` exists in the package.
   - Every `owned[i].target` exists in the package.
   - Synced pairs (mappings entries) are byte-identical between
     source and target.
   Report per-package drift; exit 1 on any fail.

3. `scripts/audit_review_source_count.py` -- parse
   `atlas_brain/services/scraping/sources.py` via `ast`, count
   `ReviewSource` enum members. Scan CLAUDE.md for "N review
   sources" / "N review sites" claims. Compare; exit 1 on
   drift.

### Files touched

- `scripts/audit_mcp_tool_names_match_docs.py` (new)
- `scripts/audit_extracted_manifests.py` (new)
- `scripts/audit_review_source_count.py` (new)
- `plans/PR-Tier-1-Audits.md` (this file, new)

No edits to existing files. `scripts/pre_push_audit.sh` does NOT
get updated in this slice -- that wiring is deferred to a follow-up
that lands after PR #483 merges (see *Deferred*).

## Mechanism

### `audit_mcp_tool_names_match_docs.py`

```
parse_doc_claims(claude_md_text) -> dict[server, set[tool_name]]
    For each "### <Name> MCP Server" header in CLAUDE.md:
        Slice the section text up to the next "### " or "## ".
        Within the slice, find all backticked snake_case
        identifiers via re.findall(r"`([a-z][a-z0-9_]+)`").
        Store the set under HEADER_TO_KEY[name].

actual_tool_names() -> dict[server, set[tool_name]]
    For each atlas_brain/mcp/*_server.py:
        Walk the file via `ast`; for each FunctionDef preceded by
        an @mcp.tool decorator, collect the function name.
    For b2b: do the same across atlas_brain/mcp/b2b/*.py.

Compare per server: missing (in code but not doc) and extra (in
doc but not code). Report; exit 1 on any drift.
```

False-positive risk: a section might mention a non-tool
snake_case identifier in backticks (e.g., a generic helper name
inside a code block). Acceptable because (a) the actual MCP
inventory uses this exact shape consistently and (b) noise shows
up as "extra in doc" which is easy to interpret and fix.

### `audit_extracted_manifests.py`

```
for each <pkg>/manifest.json:
    parse manifest
    for entry in manifest["mappings"]:
        check (REPO_ROOT / entry["source"]).exists()
        check (REPO_ROOT / entry["target"]).exists()
        check files are byte-identical
            (using pathlib.Path.read_bytes())
    for entry in manifest["owned"]:
        check (REPO_ROOT / entry["target"]).exists()

Report a single line per failure with package + reason.
Exit 1 on any fail.
```

### `audit_review_source_count.py`

```
# Truth side: parse the enum
import ast
tree = ast.parse(sources_py_text)
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == "ReviewSource":
        member_count = sum(
            1 for n in node.body if isinstance(n, ast.Assign)
        )

# Claim side: scan CLAUDE.md
pattern = r"(\d+)\s+review\s+(?:source[s]?|site[s]?)"
claims = re.findall(pattern, claude_md_text, re.IGNORECASE)
# Each claim should equal member_count.
```

Report each claim with its line number, expected vs actual, OK
or DRIFT.

## Intentional

- **Off `main`, NOT stacked on `claude/pr-pre-push-audit-scripts`
  (PR #483).** Each audit script is independently useful;
  wrapper integration is a 5-line follow-up after #483 lands.
  Stacking would couple this PR's mergeability to #483's review
  cycle.
- **No update to `scripts/pre_push_audit.sh` in this slice.** The
  wrapper lives on the PR #483 branch and on main once #483
  merges. Updating it now means either (a) duplicating PR #483's
  wrapper here or (b) basing this off #483's branch (stacked).
  Both add coupling. The clean shape is: ship the three scripts
  here, ship the wrapper update + AGENTS.md workflow doc in a
  follow-up `PR-Pre-Merge-Workflow-And-Wrapper` after #483
  merges.
- **`audit_extracted_manifests.py` walks `extracted_*/manifest.json`
  globs, not a hardcoded package list.** Survives future
  package additions or removals. Note: `extracted_reasoning_core/`
  does not have a manifest today; CLAUDE.md claims it does. That
  is a real drift the auditor surfaces, but fixing CLAUDE.md is
  Deferred.
- **`audit_mcp_tool_names_match_docs.py` uses backtick-snake_case
  heuristic, not "lines starting with `Tools:`".** The actual
  inventory in CLAUDE.md spans multiple continuation lines and
  has multiple `Tools (Group, N): ...` rows per server. A
  line-prefix heuristic misses the continuation lines; the
  whole-section heuristic does not.
- **`audit_review_source_count.py` uses `ast` to count enum
  members, not regex.** Regex is brittle for Python source;
  `ast` survives multi-line members, comments, and decorators.
- **All three scripts ASCII-only.** Matches the
  `scripts/check_ascii_python.sh` policy.

## Deferred

- **`scripts/pre_push_audit.sh` wiring.** Follow-up:
  `PR-Pre-Merge-Workflow-And-Wrapper` (after PR #483 merges).
  That PR also adds the AGENTS.md pre-merge gate workflow section
  (mechanical-then-model order) the user asked for.
- **AGENTS.md reference to the new auditors.** Same follow-up.
- **Fix CLAUDE.md's claim that `extracted_reasoning_core` has a
  manifest.** The auditor will surface this when run; the actual
  CLAUDE.md edit is a doc fix outside this slice's scope.
- **`audit_pr_claims.py`** -- the Tier-3 broader claim verifier.
  Still useful but lower ROI; deferred to a later slice if a
  specific claim drift recurs.
- **CI integration (GitHub Actions).** Eventual goal: mechanical
  audits run automatically on every push so a Claude session
  isn't needed for the deterministic part. Phase 2 of the
  pre-merge gate plan.
- **`audit_mcp_tool_names_match_docs.py` per-tool grouping.**
  Currently treats the inventory as a set; doesn't verify the
  `Tools (Strategic, 8): ...` group-count sub-claims. Follow-up.

## Verification

Local commands the builder ran (reviewer should reproduce):

```bash
# 1. Tool-name inventory auditor on current main (CLAUDE.md is
#    stale on main; PR #457 is the fix). Expected: drift across
#    several servers because PR #457 hasn't merged.
python scripts/audit_mcp_tool_names_match_docs.py
echo "exit: $?"

# 2. Manifest auditor on current main. Expected: pass for all 5
#    extracted_* packages OR surface real drift (any non-existent
#    source path, any drift in synced pairs).
python scripts/audit_extracted_manifests.py
echo "exit: $?"

# 3. Review-source-count auditor on current main. Expected:
#    DRIFT because CLAUDE.md on main says "16 review sites" but
#    actual enum has 19 members. (PR #457 fixes this.)
python scripts/audit_review_source_count.py
echo "exit: $?"
```

## Estimated diff size

| File | LOC (est) |
|---|---|
| `scripts/audit_mcp_tool_names_match_docs.py` | ~110 |
| `scripts/audit_extracted_manifests.py` | ~90 |
| `scripts/audit_review_source_count.py` | ~75 |
| `plans/PR-Tier-1-Audits.md` | ~175 |
| **Total** | **~450** |

50 LOC over the 400 soft cap; same shape as PR #483 (plan doc
~40% of the total). Slice is genuinely indivisible (three audit
scripts + plan ship together); splitting one audit into its own
PR would three plan docs each ~150 LOC, net worse for the
reviewer.
