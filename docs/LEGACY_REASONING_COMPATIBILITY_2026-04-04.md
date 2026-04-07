# Legacy Reasoning Compatibility

Date: 2026-04-04

## Decision

Legacy reasoning from `b2b_churn_signals` is no longer part of the active
delivery path for vendor or cross-vendor reasoning.

The remaining compatibility code exists only as an explicit opt-in safety
path during burn-in.

## Burn-In Removal Target

- Burn-in window ends: `2026-04-18`
- Runtime kill switch: `ATLAS_B2B_CHURN_LEGACY_REASONING_FALLBACK_ENABLED`
- If no production incidents require legacy reasoning before that date:
  - delete `reconstruct_reasoning_lookup()`
  - delete `reconstruct_cross_vendor_lookup()`
  - delete legacy fallback branches in `_b2b_synthesis_reader.py`
  - delete legacy fallback branch in `_b2b_cross_vendor_synthesis.py`

## Remaining `b2b_churn_signals` Usage

### Keep

These are deterministic metric reads and should remain:

- `atlas_brain/autonomous/tasks/_b2b_shared.py`
- `atlas_brain/mcp/b2b/signals.py`
- `atlas_brain/autonomous/tasks/b2b_vendor_briefing.py`
- `atlas_brain/api/b2b_dashboard.py`
- `atlas_brain/api/b2b_tenant_dashboard.py`
- `atlas_brain/api/b2b_win_loss.py`
- `atlas_brain/autonomous/tasks/b2b_blog_post_generation.py`
- `atlas_brain/autonomous/tasks/b2b_keyword_signal.py`

### Compatibility Only

These should be deleted after burn-in:

- `atlas_brain/autonomous/tasks/b2b_churn_intelligence.py`
  - `reconstruct_reasoning_lookup()`
  - `reconstruct_cross_vendor_lookup()`
- `atlas_brain/autonomous/tasks/_b2b_synthesis_reader.py`
  - `allow_legacy_fallback=True` path
  - `include_legacy=True` discovery path
- `atlas_brain/autonomous/tasks/_b2b_cross_vendor_synthesis.py`
  - `allow_legacy_fallback=True` path

### Already Removed As Active Fallbacks

- `b2b_battle_cards`
- `b2b_churn_reports`
- `b2b_accounts_in_motion`
- `b2b_article_correlation`
- `b2b_challenger_brief`
- `mcp/b2b/write_intelligence`
- cross-vendor battle-card merge path

## Removal Gate

Delete the compatibility paths only after verifying:

1. No user-facing consumer calls legacy fallback code in production logs.
2. Synthesis coverage remains sufficient for active vendors/categories.
3. Cross-vendor synthesis continues to populate the required battle/council outputs.
4. No operator workflow depends on the old compatibility readers.
