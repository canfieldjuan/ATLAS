# In-Flight PRs

Last updated: 2026-05-05T01:33Z by codex-content-seller-opportunities-worker

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #210 | Own Competitive Intelligence Anthropic batch helper boundary | `extracted_competitive_intelligence/autonomous/tasks/_b2b_batch_utils.py`; `extracted_competitive_intelligence/manifest.json`; `extracted_competitive_intelligence/README.md`; `extracted_competitive_intelligence/STATUS.md`; `scripts/run_extracted_competitive_intelligence_checks.sh`; `scripts/smoke_extracted_competitive_intelligence_standalone.py`; `tests/test_extracted_competitive_batch_utils.py`; `tests/test_extracted_competitive_manifest.py` | codex-2026-05-05 | Competitive Intelligence batch helper seam only; avoid Atlas reasoning state, extracted pipeline workflow, and cross-product audit files |
| (PR-C4b, in flight) | PR-C4b: Atlas reasoning state inherits core's ReasoningAgentState (PR 6 second slice) | EDIT: `atlas_brain/reasoning/state.py` (atlas's `ReasoningAgentState` becomes a `TypedDict` subclass of `extracted_reasoning_core.state.ReasoningAgentState` -- atlas keeps its 10 atlas-specific context fields (`crm_context`, `email_history`, `voice_turns`, `calendar_events`, `sms_messages`, `graph_facts`, `recent_events`, `market_context`, `news_context`, `b2b_churn`); core fields are inherited). NEW: `tests/test_atlas_reasoning_state_inherits_core.py` (subclass relationship + subtype acceptance + token-tracking smoke). Pure typing seam -- no runtime behavior changes; `agent.py` / `graph.py` import sites unchanged. Forward-looking note: PR-C4d/e will decide whether atlas's granular context fields stay atlas-side or migrate up; this slice only establishes the structural is-a. | claude-2026-05-03 | `atlas_brain/reasoning/state.py`; `tests/test_atlas_reasoning_state_inherits_core.py` |
| #164 | docs: log cross-product standalone % audit | `docs/extraction/cross_product_audit_2026-05-04.md` | canfieldjuan | Avoid editing the cross-product audit doc until PR #164 lands |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
