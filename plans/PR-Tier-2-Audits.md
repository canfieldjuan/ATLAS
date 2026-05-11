# PR-Tier-2-Audits

## Why this slice exists

PR #483 and PR #484 shipped Tier 1: mechanical audits that catch
drift Codex actually flagged on this session's PRs (MCP tool
counts, plan-doc shape, manifest sync, review-source enum, MCP
tool inventory). Tier 2 enforces the AGENTS.md contract itself,
not just doc/code consistency:

1. **Files-touched scope drift.** A plan doc's *Scope -> Files
   touched* subsection is the contract. AGENTS.md anti-pattern
   list calls out scope creep ("while I was at it cleanups") as
   a thing that should never appear in a PR. Today nothing
   audits the actual diff against that contract.

2. **Diff-size estimate drift.** Plan docs claim "~310 LOC"; the
   actual diff might be ~544. AGENTS.md soft cap is 400 LOC,
   with overage requiring justification in *Why this slice
   exists*. Today nothing audits the estimate against reality
   -- which means builders (myself included) can drift past the
   cap without realizing it.

3. **MCP port assignment drift.** `atlas_brain/config.py`
   `MCPConfig` is the source of truth for port numbers 8056-8064.
   CLAUDE.md duplicates those numbers in env-var blocks and (in
   PR #457) a markdown table. Today nothing audits these. A port
   renumber in config without a doc update would silently break
   every developer's Claude Desktop config.

This is recommendation #2 from the PR #457 token-savings analysis
(*bash scripts for mechanical audits, zero LLM, biggest absolute
saver*), Tier 2.

## Scope (this PR)

1. `scripts/audit_plan_doc_files_touched.py PATH [BASE_REF]` --
   parse the plan doc's *Files touched* subsection (backticked
   paths under a "### Files touched" or "**Files touched**"
   sub-heading), run `git diff --name-only <BASE_REF>...HEAD`
   for the actual changes, report missing-in-doc and
   extra-in-doc. Default BASE_REF = `origin/main`.

2. `scripts/audit_plan_doc_diff_size.py PATH [BASE_REF]` --
   parse the plan doc's *Estimated diff size* section for the
   "Total" LOC number, run `git diff --shortstat
   <BASE_REF>...HEAD` for the actual additions+deletions,
   compare. WARN if drift > 25%; FAIL if > 50%.

3. `scripts/audit_mcp_port_assignments.py` -- parse
   `atlas_brain/config.py` `MCPConfig` class via `ast`, extract
   each `<name>_port: int = Field(default=N, ...)` assignment.
   Scan CLAUDE.md for env-var-style lines (`ATLAS_MCP_<NAME>_PORT=N`)
   and table-style rows (`| <Name> | <port> | ... |`). Compare
   per-port; report drift.

### Files touched

- `scripts/audit_plan_doc_files_touched.py` (new)
- `scripts/audit_plan_doc_diff_size.py` (new)
- `scripts/audit_mcp_port_assignments.py` (new)
- `plans/PR-Tier-2-Audits.md` (this file, new)

No edits to existing files. Wrapper integration deferred (same
shape as PR #484).

## Mechanism

### `audit_plan_doc_files_touched.py`

```
parse_files_touched(plan_text) -> set[str]
    Locate the "Files touched" sub-heading (any of:
      "### Files touched", "**Files touched**", "## Files touched").
    Slice the subsection up to the next "##"/"###" heading or the
    final line.
    Within the slice, find backticked paths via regex
      `([^\`]+\.[a-z]+)`
    Return as a set of repo-relative paths.

actual_files_changed(base_ref) -> set[str]
    git diff --name-only <base_ref>...HEAD
    Return as a set.

Report:
    missing-in-doc  -> changed in git, not named in plan doc
                       (could be scope creep)
    extra-in-doc    -> named in plan doc, not changed in git
                       (could be a plan/code mismatch)
Exit 1 on any drift.
```

### `audit_plan_doc_diff_size.py`

```
parse_estimate(plan_text) -> int
    Find a "**Total**" row in the estimate table with a "~N"
    number, e.g. "| **Total** | **~310** |". Strip the tilde.

actual_diff_size(base_ref) -> int
    git diff --shortstat <base_ref>...HEAD
    Parse output like:
      "4 files changed, 441 insertions(+), 13 deletions(-)"
    Return insertions + deletions.

Compare:
    drift_pct = abs(actual - estimate) / estimate
    drift_pct <= 0.25 -> OK
    0.25 < drift_pct <= 0.50 -> WARN (exit 0, but print warning)
    drift_pct > 0.50 -> FAIL (exit 1)
```

### `audit_mcp_port_assignments.py`

```
config_ports() -> dict[str, int]
    ast-walk atlas_brain/config.py, find class MCPConfig.
    For each AnnAssign whose target.id matches "<name>_port":
        Extract default int from the Field(default=N, ...) call.
    Return {name: port}.

doc_ports(claude_md_text) -> dict[str, list[(line, port)]]
    Two scan passes:
      A. Env-var style: r"^ATLAS_MCP_([A-Z_]+)_PORT\s*=\s*(\d+)"
      B. Table-row style: r"^\| ([A-Z][\w +]+?) \| (\d{4,5}) \|"
    Normalize to dict[name -> [(line, port)]].

Compare each claim site to config_ports[name]. Report DRIFT lines.
```

## Intentional

- **Three standalone scripts, not one mega-script.** Each does
  one thing; each is easier to read, test, and wire into a CI
  job separately. The wrapper composes them.
- **`audit_plan_doc_files_touched.py` accepts an optional
  `BASE_REF` arg.** Defaults to `origin/main` but lets the user
  override for stacked PRs (e.g., `origin/claude/pr-tier-1-audits`).
- **`audit_plan_doc_diff_size.py` uses a soft + hard threshold.**
  25% tolerance is OK (estimates are estimates); 50% is a fail
  because at that point the plan doc isn't a useful contract.
- **Port auditor parses both doc formats** -- env-var-style
  (present on `main`) and markdown-table-style (added in PR #457).
  Either is canonical; both should match config.py.
- **All ASCII-only `.py`.** Continues policy.
- **Not extending `scripts/audit_plan_doc.py` (from PR #483).**
  Separate scripts because PR #483 hasn't merged; basing this
  off main means the file doesn't exist locally. Extension is
  a clean follow-up after #483 lands.

## Deferred

- **Wrapper integration into `scripts/pre_push_audit.sh`.**
  Follow-up after PR #483 merges (carries the wrapper to main).
- **AGENTS.md reference to Tier 2.** Same follow-up.
- **Per-skill / per-domain skill audits.** Tier 3 candidate.
- **Auditor for "Tools (Group, N): ..." sub-group counts.**
  Deferred from PR #484.
- **`audit_pr_claims.py`** -- broad PR body claim verifier.
  Still useful, still Tier 3.
- **CI integration (GitHub Actions).** Phase 2 of pre-merge gate.

## Verification

Local commands the builder ran (reviewer should reproduce):

```bash
# 1. Files-touched audit on this slice's own plan doc.
#    Expected: OK once the plan doc and scripts are committed.
python scripts/audit_plan_doc_files_touched.py \
    plans/PR-Tier-2-Audits.md origin/main
echo "exit: $?"

# 2. Diff-size audit on this slice's own plan doc.
#    Expected: drift within 25% of the estimate -> OK.
python scripts/audit_plan_doc_diff_size.py \
    plans/PR-Tier-2-Audits.md origin/main
echo "exit: $?"

# 3. Port-assignment audit on current state.
#    Expected: all 9 ports match between config.py and CLAUDE.md
#    on main (the env-var block is accurate even though PR #457
#    isn't merged).
python scripts/audit_mcp_port_assignments.py
echo "exit: $?"

# 4. Known-bad cases:
#    Plan doc with a Files-touched list that omits a real file:
#      -> 1 missing-in-doc line, exit 1.
#    Plan doc with an estimate < half of actual:
#      -> FAIL (exit 1).
```

## Estimated diff size

| File | LOC (est) |
|---|---|
| `scripts/audit_plan_doc_files_touched.py` | ~85 |
| `scripts/audit_plan_doc_diff_size.py` | ~75 |
| `scripts/audit_mcp_port_assignments.py` | ~100 |
| `plans/PR-Tier-2-Audits.md` | ~200 |
| **Total** | **~460** |

60 LOC over the 400 soft cap; same shape as PRs #483 and #484
(plan doc is ~43% of the total). Slice is indivisible; splitting
each script into its own PR would require three plan docs each
~150 LOC, net worse for review.
