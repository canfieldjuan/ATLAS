# In-Flight PRs

Last updated: 2026-05-05T03:46Z by claude-2026-05-03

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (PR-C4e1, in flight) | PR-C4e1: pure graph helpers + plan_actions to core (PR 6 fifth slice, 1/3 of graph extraction) | NEW: `extracted_reasoning_core/graph_helpers.py` (regex constants, `clean_summary_text`, `build_notification_fallback`, `has_suspicious_trailing_fragment`, `sanitize_notification_summary`, `parse_llm_json`, `valid_uuid_str`, `plan_actions`). EDIT: `atlas_brain/reasoning/graph.py` (re-export the helpers under their existing private names so reflection.py and other internal callers don't break). NEW: `tests/test_extracted_reasoning_core_graph_helpers.py` (unit coverage on each helper -- regex idempotence, fallback message shape, plan_actions filtering on SAFE_ACTIONS + confidence). EDIT: `scripts/run_extracted_pipeline_checks.sh` (wire new test). First of three PR-C4e sub-slices: pure logic move, no new ports, no behavior change. PR-C4e2 covers LLMClient port alignment + triage/reason/synthesize nodes; PR-C4e3 covers the orchestrator + atlas-coupled node adapters. | claude-2026-05-03 | `extracted_reasoning_core/graph_helpers.py`; `atlas_brain/reasoning/graph.py`; `tests/test_extracted_reasoning_core_graph_helpers.py`; `scripts/run_extracted_pipeline_checks.sh` |
| #164 | docs: log cross-product standalone % audit | `docs/extraction/cross_product_audit_2026-05-04.md` | canfieldjuan | Avoid editing the cross-product audit doc until PR #164 lands |
| #209 | Add seller category intelligence refresh seam | NEW: `extracted_content_pipeline/campaign_postgres_seller_category_intelligence.py`; NEW: `scripts/refresh_extracted_seller_category_intelligence.py`; NEW: `tests/test_extracted_campaign_postgres_seller_category_intelligence.py`; EDIT: extracted content pipeline manifest/import smoke/checks/docs/status/coordination. No Atlas source edits. | codex-2026-05-05 | Avoid seller category intelligence refresh files until this PR lands |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
