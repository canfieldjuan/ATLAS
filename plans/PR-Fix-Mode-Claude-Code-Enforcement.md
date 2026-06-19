# PR-Fix-Mode-Claude-Code-Enforcement

## Why this slice exists

The PR fix-mode doc layer (#1715) and the Codex push-time `Max files:` budget
(#1716) are merged. This slice adds the Claude Code *pre-edit* half from issue
#1714: a `PreToolUse` hook that denies an edit to a file outside the declared
failure source **before tokens are spent**, a `SessionStart` hook that
re-injects the fix baton after compaction, and a `/fix-mode` skill to arm it.
The push-time audit only catches strays after the work is done; this stops them
at the edit.

This PR is over the 400-LOC soft cap (~513). The overage is intentional and
indivisible: the gitignore enablement, the two hooks, the `/fix-mode` skill, and
the fail-open test matrix are one unit -- a half-wired hook set (e.g. a deny hook
with no skill to arm it, or a skill writing a baton no hook reads) would be worse
than one coherent slice. Roughly half the diff is the plan doc plus the
fail-open/deny test coverage, not new behavior.

## Scope (this PR)

Ownership lane: process/agent-guidance
Slice phase: Production hardening
Max files: 7

The hooks are committed and active in `.claude/settings.json` (team-wide) but
**fail open**: they do nothing unless a gitignored fix-mode baton exists, and any
error / missing / malformed input exits 0 (allow). Normal sessions and any
unarmed checkout are never blocked.

### Files touched

- `.gitignore`
- `.claude/settings.json`
- `.claude/hooks/check_edit_budget.py`
- `.claude/hooks/inject_fix_mode.py`
- `.claude/skills/fix-mode/SKILL.md`
- `tests/test_fix_mode_hook.py`
- `plans/PR-Fix-Mode-Claude-Code-Enforcement.md`

### Review Contract

Acceptance criteria:

- [ ] With no / inactive / malformed / empty-allowed baton, `.claude/hooks/check_edit_budget.py`
      exits 0 and emits no deny -- the committed hook is inert for normal sessions.
- [ ] With an active baton, an edit outside the `allowed` globs returns
      `permissionDecision: "deny"` with a reason; an edit inside is allowed;
      MultiEdit denies if any target is outside; absolute paths are relativized
      before matching.
- [ ] `.claude/hooks/inject_fix_mode.py` emits the baton as `additionalContext` when active and
      nothing otherwise.
- [ ] `.gitignore` tracks only `.claude/settings.json`, `.claude/hooks/`, and
      `.claude/skills/`; `.claude/settings.local.json` and the baton stay ignored.
- [ ] No secrets or absolute/home paths in any committed `.claude/` file.

Affected surfaces: Claude Code session tooling only; no application code.

Risk areas: a committed active `PreToolUse` hook can block edits for everyone who
pulls the repo. Mitigated by exhaustive fail-open (every error path exits 0) and
tests covering the missing / inactive / malformed / empty-allowed baton paths.

Reviewer rules triggered: R1, R14, R2.

## Mechanism

`.claude/hooks/check_edit_budget.py` reads the PreToolUse payload from stdin, collects target
paths (`file_path` for Edit/Write, `edits[].file_path` for MultiEdit), and reads
`.claude/fix-mode-state.json` under `CLAUDE_PROJECT_DIR`. If the baton is absent,
inactive, malformed, or has no `allowed` globs, it returns 0 with no output. With
an active baton it relativizes each target and denies (via `permissionDecision`)
any that no `allowed` glob matches (`fnmatch`). Every error path exits 0, so the
hook cannot wedge a session. It enforces the allowed *set* only; the merged
`scripts/audit_plan_doc_files_touched.py` budget enforces the file *count* at
pre-push. `.claude/hooks/inject_fix_mode.py` re-emits the baton fields as `additionalContext`
on `SessionStart` (`startup|resume|compact`). The `/fix-mode` skill writes the
baton; `.gitignore` uses an allowlist so only the shared, scrubbed config is
tracked.

## Intentional

- Committed and active, but fail-open, so enforcement only bites inside an armed
  fix loop and normal work is untouched.
- Dedicated machine-readable baton `.claude/fix-mode-state.json` for the hooks,
  alongside the human `## PR Fix Mode` block already in `SESSION_STATE.local.md`
  (#1715); the skill writes both.
- Hooks in `python3` (repo convention for JSON in scripts; no `jq` dependency),
  ASCII-only.
- The allowed-set lives in the hook; the count budget stays in the #1716
  pre-push audit -- two composable layers, no stateful counting in the hook.

## Deferred

- A `PreCompact` snapshot hook: `SessionStart` `matcher: compact` already
  re-injects the on-disk baton after auto-compaction, so PreCompact adds little;
  revisit if volatile in-context findings need snapshotting first. (Tracked in
  #1714.)
- An optional pre-commit secret-scan over staged `.claude/` files. (Tracked in
  #1714.)

## Parked hardening

None.

## Verification

- Run the hook tests in `tests/test_fix_mode_hook.py` -- 10 cases pass (fail-open
  for no/inactive/malformed/empty baton; allow inside; deny outside; MultiEdit;
  absolute-path relativization; SessionStart inject present/absent).
- Validate `.claude/settings.json` parses as JSON.
- Fail-open proof: with no baton, pipe an Edit payload to
  `.claude/hooks/check_edit_budget.py` -- exit 0, no output.
- Deny proof: with an active baton `allowed: ["scripts/*"]`, pipe an Edit to a
  `tests/` path -- deny JSON.
- Confirm `git check-ignore` keeps `.claude/settings.local.json` and the baton
  ignored while the four shared files are trackable.
- Run `scripts/local_pr_review.sh` -- full bundle green, including this plan
  passing its own `Max files: 7` budget.

## Estimated diff size

| File | LOC |
|---|---:|
| `.gitignore` | 8 |
| `.claude/settings.json` | 26 |
| `.claude/hooks/check_edit_budget.py` | 105 |
| `.claude/hooks/inject_fix_mode.py` | 70 |
| `.claude/skills/fix-mode/SKILL.md` | 50 |
| `tests/test_fix_mode_hook.py` | 130 |
| `plans/PR-Fix-Mode-Claude-Code-Enforcement.md` | ~124 |
| **Total** | **~513** |
