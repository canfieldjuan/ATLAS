# PR-Content-Ops-Reasoning-DB-Check

## Why this slice exists

PR #463 added the DB-backed reasoning provider and PR #469 proved the provider
contract through an offline fixture. Operators still need a small live-DSN
check after applying migration 277 and loading `campaign_reasoning_contexts`
rows, without running a full generation job.

## Scope (this PR)

1. Add `scripts/check_extracted_campaign_reasoning_postgres.py`.
2. Use the existing `PostgresCampaignReasoningContextRepository` read path.
3. Return JSON or concise text and exit non-zero when no context matches.
4. Add focused unit tests and include them in the extracted gauntlet.
5. Document the check in README and the host install runbook.

### Files touched

- `scripts/check_extracted_campaign_reasoning_postgres.py`
- `tests/test_extracted_campaign_reasoning_postgres_check.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/docs/host_install_runbook.md`
- `docs/extraction/coordination/inflight.md`

## Mechanism

The script accepts the same DSN convention as other extracted DB CLIs:
`--database-url`, `EXTRACTED_DATABASE_URL`, or `DATABASE_URL`. It builds a
`TenantScope`, passes target selectors to
`PostgresCampaignReasoningContextRepository.read_campaign_reasoning_context`,
and emits:

- `status="ok"` plus the normalized context payload when found.
- `status="missing"` with exit code 1 when no context matches.

The check is read-only and does not create, mutate, or seed rows.

## Intentional

- No generation, sender, LLM, or network-provider handles are opened.
- No production API behavior changes.
- No new table shape; the script only uses migration 277's table through the
  existing repository.

## Deferred

- Full end-to-end live generation smoke from DB reasoning row to persisted
  generated asset.
- Admin UI surfacing for provider health.

## Verification

- Focused test file.
- `python -m py_compile` on the new script.
- Full extracted pipeline check.
- `git diff --check`.
- ASCII byte check on edited Python/test files.

## Estimated diff size

7 files, roughly +190 / -0. Under the 400 LOC soft review budget.
