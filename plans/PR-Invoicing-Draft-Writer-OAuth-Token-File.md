## Why this slice exists

The draft-writer invoicing OAuth launcher merged with the correct fail-closed
approval-token validation, but the first operator dry-run immediately fails on
a fresh checkout unless the approval token is already exported in the shell.

Passing the token as a raw CLI value would be convenient, but it would expose a
write-approval secret through shell history and process listings. This slice
adds a safer operator path: store the approval token in a local file and pass
the file path to the launcher and e2e smoke.

## Scope (this PR)

1. Add `--approval-token-file` to the draft-writer OAuth server launcher.
2. Add `--approval-token-file` to the draft-writer OAuth e2e smoke.
3. Keep existing env-var behavior unchanged.
4. Document a `0600` local-token-file setup flow and keep `.secrets/` ignored.
5. Add tests that prove file tokens are loaded and never printed.

### Files touched

- `scripts/start_invoicing_draft_writer_oauth_server.py`
- `scripts/check_invoicing_draft_writer_oauth_e2e.py`
- `tests/test_start_invoicing_draft_writer_oauth_server.py`
- `tests/test_check_invoicing_draft_writer_oauth_e2e.py`
- `docs/INVOICING_MCP_WRITE_ACCESS_GUARDRAILS.md`
- `docs/MCP_CHATGPT_OAUTH_ROLLOUT_RUNBOOK.md`
- `.gitignore`
- `plans/PR-Invoicing-Draft-Writer-OAuth-Token-File.md`

## Mechanism

Both scripts accept an optional `--approval-token-file <path>`. If provided,
the script reads the first non-empty file contents, strips surrounding
whitespace, and uses that value as
`ATLAS_MCP_INVOICING_DRAFT_WRITER_OAUTH_APPROVAL_TOKEN` for validation/runtime.

The env-var path remains the default, so existing operator and CI flows keep
working.

## Intentional

- There is no raw `--approval-token` flag on the server launcher. Avoiding
  shell-history/process-list exposure for the long-running process is worth one
  extra local file.
- The e2e smoke keeps its existing raw `--approval-token` option for direct
  test compatibility, but docs route operators through `--approval-token-file`.
- The token-file path is explicit. The scripts do not auto-discover secret
  files because silent secret selection is harder to audit.
- The launcher still refuses short tokens. File-based loading changes only how
  the token is supplied, not the security contract.

## Deferred

- Automatic token rotation is deferred until the connector is run as an
  unattended service.
- Persisted OAuth client/token storage remains deferred; the current flow is
  still local operator approval.

## Verification

Planned commands:

```bash
.venv/bin/python -m py_compile scripts/start_invoicing_draft_writer_oauth_server.py scripts/check_invoicing_draft_writer_oauth_e2e.py tests/test_start_invoicing_draft_writer_oauth_server.py tests/test_check_invoicing_draft_writer_oauth_e2e.py
.venv/bin/pytest tests/test_start_invoicing_draft_writer_oauth_server.py tests/test_check_invoicing_draft_writer_oauth_e2e.py -q
bash scripts/local_pr_review.sh --allow-dirty
git diff --check
```

## Estimated diff size

| Area | LOC churn (approx) |
|---|---:|
| Scripts | ~45 |
| Tests | ~70 |
| Docs/plan | ~90 |
| Git ignore | ~1 |
| **Total** | **~206** |
