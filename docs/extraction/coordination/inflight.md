# In-Flight PRs

Last updated: 2026-05-04T05:07Z by codex-2026-05-04

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (PR-C1j, in flight) | PR-C1j: Route `extracted_content_pipeline/reasoning/temporal.py` through reasoning core wrapper | EDIT: `extracted_content_pipeline/reasoning/temporal.py` (drop ~466-line drifted fork; replace with thin re-export wrapper from `extracted_reasoning_core.temporal` + `extracted_reasoning_core.types`). EDIT: `extracted_reasoning_core/temporal.py` (fix latent frozen-dataclass mutations in `analyze_vendor` / `_compute_long_term_trends` activated by PR-C1c; drift-forward `_coerce_date` / `_days_between` / `_volatility` / `_percentiles_from_rows` helpers; replace atlas-coupled `_compute_percentiles` with self-contained SQL; drop dead `_infer_category`). Existing `tests/test_extracted_reasoning_temporal.py` keeps green against the wrapper. | claude-2026-05-03 | `extracted_content_pipeline/reasoning/temporal.py`; `extracted_reasoning_core/temporal.py`; `tests/test_extracted_reasoning_temporal.py` |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
