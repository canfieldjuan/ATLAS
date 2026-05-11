# PR-Pre-Push-Audit-Scripts

## Why this slice exists

PR #457 (the CLAUDE.md / AGENTS.md refresh) was caught three times
by the Codex bot for stale numeric claims in `CLAUDE.md`:

1. `email_server` listed as 8 tools; actual is 9.
2. `invoicing_server` listed as 15 tools; actual is 18.
3. `intelligence_server` listed as 17 tools; actual is 33.
4. `b2b_churn_server` headline listed as "60+" / "66"; actual sum
   across `mcp/b2b/*.py` is 83.
5. `MCP Servers` section intro still said "Seven MCP servers" after
   nine were documented above it.
6. Invoicing tool inventory listed 15 names below an "18 tools"
   header.

Each catch cost a Codex review round-trip, a CLAUDE.md edit, a
commit, and a push. The drift would never have shipped if a
mechanical pre-push check had grep-counted `@mcp.tool` decorators
and diffed against CLAUDE.md's claims.

Companion analysis in PR #457's discussion concluded: *bash scripts
beat subagents for mechanical audits*. Subagents save parent-context
tokens but cost system-prompt overhead per invocation; bash scripts
cost zero LLM tokens. The biggest absolute-token savings on the
table for our PR workflow are the mechanical audits we don't yet
have.

This slice ships those audits.

## Scope (this PR)

1. `scripts/audit_claude_md_claims.py` — parse `CLAUDE.md` for
   numeric MCP tool-count claims, count actual `@mcp.tool`
   decorators in `atlas_brain/mcp/*_server.py` (and the
   `atlas_brain/mcp/b2b/` sub-module split for `b2b_churn_server`),
   report drift, exit 1 if any.
2. `scripts/audit_plan_doc.py` — given a path to a plan doc,
   verify it has the 7 required AGENTS.md sections in order
   (Why / Scope / Mechanism / Intentional / Deferred /
   Verification / Estimated diff size). Exit 1 if any missing or
   out of order.
3. `scripts/pre_push_audit.sh` — convenience wrapper that runs
   both above + existing `scripts/check_ascii_python.sh`. One
   command for the builder to run before opening a PR.

### Files touched

- `scripts/audit_claude_md_claims.py` (new)
- `scripts/audit_plan_doc.py` (new)
- `scripts/pre_push_audit.sh` (new)
- `plans/PR-Pre-Push-Audit-Scripts.md` (this file, new)

No edits to existing files. No wiring into git hooks, CI, or
`AGENTS.md` (all deferred — see *Deferred*).

## Mechanism

### `audit_claude_md_claims.py`

Two functions:

```
actual_counts() -> dict[str, int]
    For each atlas_brain/mcp/*_server.py, count grep -c "@mcp.tool".
    For atlas_brain/mcp/b2b/, sum @mcp.tool across all *.py files
    and report as "b2b_churn".

claimed_counts() -> dict[str, int]
    Parse the MCP servers table in CLAUDE.md (the "### atlas_brain/mcp/"
    table). Match rows shaped like "| Email | 8057 | 9 | ..." and
    extract (server name, tool count).
```

Report per-server: claimed, actual, status (OK or DRIFT). Exit 1
if any DRIFT.

### `audit_plan_doc.py`

Reads the plan doc; for each of the 7 required section titles,
scan for a line starting with `## ` containing that title (case
insensitive). Track order. Report each section as `OK <heading>`
or `MISSING ## <title>`. Exit 1 if any missing or out of order.

### `pre_push_audit.sh`

Bash wrapper that runs the above two + the existing ASCII check.
Reports a summary line per check. Exit 1 if any check failed.

## Intentional

- **Python, not pure bash, for the parsers.** `audit_claude_md_claims`
  needs regex matching across markdown tables; `audit_plan_doc`
  needs ordered-section validation. Both stay simple in Python
  (~80 LOC each) vs gnarly in bash. The wrapper is bash for the
  "run everything" entry point.
- **ASCII output only** — matches `scripts/check_ascii_python.sh`
  policy for `.py` files; using OK / DRIFT / MISSING text labels
  instead of ✓ / ✗ unicode.
- **No new dependencies.** Pure stdlib (`pathlib`, `re`, `sys`).
  Runs in any venv that has Python 3.10+.
- **Auditors report drift; they don't enforce a specific CLAUDE.md
  state.** On `main` today, `audit_claude_md_claims.py` will
  report drift because PR #457 hasn't merged. When #457 merges,
  drift goes to zero. The auditor's job is to *tell you*, not to
  presume which side is correct.
- **No git hook installation.** Wiring `pre_push_audit.sh` into
  `.git/hooks/pre-push` is a per-developer choice; some prefer to
  run it manually. Scripted as a runnable file under `scripts/`,
  not as a hook.
- **`audit_plan_doc.py` does not yet check "Files touched"
  matches `git diff --name-only`.** That requires a `--base-ref`
  argument and git plumbing; punted to a follow-up so this slice
  stays small and reviewable.

## Deferred

- **`scripts/audit_pr_claims.py`** — a broader claim verifier that
  parses arbitrary PR body / CLAUDE.md claims (file paths exist,
  "wired into X" claims, package LOC claims) and verifies each.
  Would have caught the `.mcp.json` claim Codex flagged in PR #457.
  Follow-up slice: `PR-Audit-PR-Claims`.
- **`audit_plan_doc.py --base-ref` flag** to verify the
  *Files touched* subsection matches `git diff --name-only
  <base-ref>...HEAD`. Follow-up slice once the base auditor is in
  use.
- **Git pre-push hook wiring.** A `scripts/install_pre_push_hook.sh`
  installer that creates `.git/hooks/pre-push` calling
  `scripts/pre_push_audit.sh`. Follow-up if/when we want enforcement.
- **CI integration.** Adding the auditors to the GitHub Actions
  pipeline. Follow-up: `PR-CI-Wire-Audit-Scripts`.
- **AGENTS.md reference to the auditors.** A bullet under
  "Builder workflow" that says "run `bash scripts/pre_push_audit.sh`
  before pushing." Follow-up so this slice can land without
  AGENTS.md churn.
- **`pre_push_audit.sh` running per-`extracted_*` validators.**
  The existing `extracted/_shared/scripts/validate_extracted.sh`
  is one call away; could be wrapped. Follow-up to keep this PR's
  surface narrow.

## Verification

Local commands the builder ran (reviewer should reproduce):

```bash
# 1. The MCP claim auditor reports the drift that PR #457 fixes,
#    on main today. (Expected: non-zero exit with 4+ DRIFT lines.)
python scripts/audit_claude_md_claims.py
echo "exit: $?"
# Expected lines include:
#   email             8       9   DRIFT
#   invoicing        15      18   DRIFT
#   intelligence     17      33   DRIFT
#   b2b_churn        60+     83   DRIFT (or similar; depends on how
#                                  CLAUDE.md is parsed)

# 2. The plan-doc auditor accepts this slice's own plan doc.
python scripts/audit_plan_doc.py plans/PR-Pre-Push-Audit-Scripts.md
echo "exit: $?"
# Expected: all 7 sections OK, exit 0.

# 3. The plan-doc auditor rejects a deliberately incomplete plan.
echo "# bad plan\n\n## Why this slice exists\n\nfoo" > /tmp/bad-plan.md
python scripts/audit_plan_doc.py /tmp/bad-plan.md
echo "exit: $?"
# Expected: 6 MISSING lines, exit 1.

# 4. The wrapper runs all three checks.
bash scripts/pre_push_audit.sh
echo "exit: $?"
# Expected (on main, today): exit 1 due to CLAUDE.md drift.
```

## Estimated diff size

| File | LOC (approx) |
|---|---|
| `scripts/audit_claude_md_claims.py` | ~80 |
| `scripts/audit_plan_doc.py` | ~50 |
| `scripts/pre_push_audit.sh` | ~30 |
| `plans/PR-Pre-Push-Audit-Scripts.md` | ~150 |
| **Total** | **~310** |

Well under the 400 LOC soft cap.
