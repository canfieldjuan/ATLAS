# MCP Connector Session Handoff

## Why this slice exists

The invoicing MCP connector rollout advanced from read-only OAuth to a working
ChatGPT draft-writer connector with live write smoke and OAuth state-file
persistence. The next session should not have to reconstruct which PRs landed,
which server is running, which smokes are authoritative, or which risks remain.

This docs-only slice records the current operational state and the next-step
queue for MCP connector work.

## Scope (this PR)

1. Add a current-state handoff section to the ChatGPT OAuth rollout runbook.
2. Add a concrete draft-writer operational-state section to the invoicing write
   guardrails.
3. Capture the restart/state-file requirement, smoke commands, and known limits.

### Files touched

- `docs/MCP_CHATGPT_OAUTH_ROLLOUT_RUNBOOK.md`
- `docs/INVOICING_MCP_WRITE_ACCESS_GUARDRAILS.md`
- `plans/PR-MCP-Connector-Session-Handoff.md`

## Mechanism

The runbook gets a dated handoff section immediately after the proven outcome,
covering merged PRs, current public URL, runtime state, required smokes, and the
recommended next MCP connector slices. The invoicing guardrails get a narrower
section focused on the live draft-writer connector and its local state files.

## Intentional

- Docs only. No server, tool, OAuth, or route behavior changes.
- The handoff records current local operation but does not claim the server is
  daemonized; it is still foreground/persistent-session operation.
- The handoff names Copilot/Codex review limitations from the session so future
  agents know why local review mattered.

## Deferred

- A durable process manager/systemd user service remains a future operational
  slice if we want the MCP connector to survive terminal/session closure.
- Applying the same OAuth pattern to CRM/email/calendar/etc. remains per-server
  rollout work.

## Verification

Commands run:

    git diff --check
    bash scripts/local_pr_review.sh

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Rollout runbook handoff | ~80 |
| Invoicing guardrails handoff | ~40 |
| Plan doc | ~60 |
| **Total** | ~180 |
