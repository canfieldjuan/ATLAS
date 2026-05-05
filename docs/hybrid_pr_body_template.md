# Hybrid Extraction PR Body Template

Use this template for every PR in the current hybrid extraction wave.

## Execution-board mapping
- Slice: `PR-<n>` / `<slice name>`
- Board reference: `docs/hybrid_extraction_execution_board.md`

## Summary
- 
- 

## Behavior-change statement
- `No behavior change` OR `Compatible additive change: ...`

## Contract impact
- One of: `none` / `additive` / `breaking`
- Details:
  - 

## Scope check
- In-scope rationale:
  - 
- Explicitly not changed:
  - 

## Rollback plan
- Revert files:
  - 
- Revert command:
  - `git revert <commit>`

## Testing
- ✅ Compatibility matrix:
  - `./scripts/run_reasoning_provider_port_compat_checks.sh`
- ✅ Scoped pytest matrix (env-aware):
  - `./scripts/run_reasoning_provider_port_tests.sh`
- Additional checks:
  - 

- Optional markdown summary from JSON report:
  - `./scripts/render_hybrid_reasoning_report_summary.py --report <path>`
