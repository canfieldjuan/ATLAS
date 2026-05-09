# PR: Content Ops Reasoning Smoke Docs

## Goal

Document the `--with-reasoning` execution smoke option added for host installs.

## Scope

- Add the reasoning smoke command to the extracted content pipeline README.
- Add the same command to the control-surface API documentation.
- Add host-install runbook guidance for validating the reasoning handoff before
  mounting real generated-asset services.

## Non-Goals

- Do not change runtime code.
- Do not add new smoke behavior.
- Do not document real provider configuration beyond the existing runbook
  sections.

## Verification

- `python scripts/smoke_extracted_content_ops_execution.py --outputs email_campaign,landing_page --with-reasoning --json`
- `git diff --check`
