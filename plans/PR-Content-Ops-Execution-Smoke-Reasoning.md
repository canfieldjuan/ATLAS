# PR: Content Ops Execution Smoke Reasoning

## Goal

Make the host-facing Content Ops execution smoke validate the reasoning
provider path and the new reasoning usage audit fields.

## Scope

- Add `--with-reasoning` to `scripts/smoke_extracted_content_ops_execution.py`.
- Use reasoning-aware fake generated-asset services when the flag is set.
- Assert the JSON smoke output includes both
  `result.reasoning_contexts_used` and `reasoning.contexts_used`.

## Non-Goals

- Do not call an LLM, database, network, or real reasoning provider.
- Do not change the default smoke behavior.
- Do not change Content Ops execution semantics.

## Verification

- `python -m py_compile scripts/smoke_extracted_content_ops_execution.py`
- `python -m pytest tests/test_extracted_content_ops_execution_smoke.py -q`
- `git diff --check`
