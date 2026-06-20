---
name: fix-mode
description: Arm or clear PR fix mode -- confine edits to an allowed-files set and a max-files budget during a red-CI / review-comment fix loop. Operator-invoked.
disable-model-invocation: true
argument-hint: "[allowed-glob,allowed-glob,...] [max-files] | off"
---

# /fix-mode -- constrain a PR fix loop

Writes the machine-readable baton `.claude/fix-mode-state.json` (gitignored) that
the committed `PreToolUse` and `SessionStart` hooks read. When active, edits to
files outside the allowed set are denied before they happen, and the baton is
re-injected after compaction. See AGENTS.md 3l.

## Arm

`/fix-mode "scripts/*,tests/test_x.py" 3`

Write `.claude/fix-mode-state.json` with:

```json
{
  "active": true,
  "pr": "<#number from the open PR>",
  "branch": "<git rev-parse --abbrev-ref HEAD>",
  "latest_commit": "<git rev-parse --short HEAD>",
  "allowed": ["scripts/*", "tests/test_x.py"],
  "max_files": 3,
  "failing_check": "<the red check / review thread>",
  "last_finding": "<the line that localized it>",
  "next_action": "<one sentence>",
  "do_not_redo": "<paths ruled out, checks already green, dead ends>"
}
```

`allowed` entries are `fnmatch` globs against repo-relative paths. Set them to the
failure source only, not "everything the symptom touches." Also mirror these
fields into the `## PR Fix Mode` block of `SESSION_STATE.local.md` (the human
baton).

## Widen (root-cause decision)

If the fix genuinely needs an upstream file, add it to `allowed` **and** record
the upstream reason in the baton and the plan first (AGENTS.md 3k). Do not grow
the set silently.

## Clear

`/fix-mode off` -> set `"active": false` (or delete the file). The hooks then
fail open and stop constraining edits.
