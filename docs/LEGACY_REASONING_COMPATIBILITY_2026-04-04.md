# Legacy Reasoning Compatibility Removal

Date: 2026-04-04

## Decision

Legacy reasoning from `b2b_churn_signals` is no longer part of the active
delivery path for vendor or cross-vendor reasoning.

The legacy compatibility code has been removed. Vendor and cross-vendor
reasoning are now synthesis-only.

## Status

- Removed on: `2026-04-07`
- Runtime kill switch: deleted
- Legacy fallback branches: deleted
- Legacy reconstructors: deleted

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

### Removed Compatibility Paths

- `atlas_brain/autonomous/tasks/b2b_churn_intelligence.py`
  - `reconstruct_reasoning_lookup()`
  - `reconstruct_cross_vendor_lookup()`
- `atlas_brain/autonomous/tasks/_b2b_synthesis_reader.py`
  - legacy vendor fallback path
  - legacy discovery path
- `atlas_brain/autonomous/tasks/_b2b_cross_vendor_synthesis.py`
  - legacy cross-vendor fallback path

### Already Removed As Active Fallbacks

- `b2b_battle_cards`
- `b2b_churn_reports`
- `b2b_accounts_in_motion`
- `b2b_article_correlation`
- `b2b_challenger_brief`
- `mcp/b2b/write_intelligence`
- cross-vendor battle-card merge path

## Verification

Removal was completed only after verifying:

1. No active runtime callers still requested legacy fallback.
2. Readiness remained green with fresh `v2` synthesis rows.
3. Competitive-set scoped synthesis still hash-reused and reran correctly.
4. User-facing consumers were already synthesis-only.
