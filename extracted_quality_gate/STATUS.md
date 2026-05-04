# Extracted Quality Gate Status

Date: 2026-05-03

## Current Slice

PR-B5c: source-quality pack.

- Deterministic core (`source_quality_pack.py`):
  - `apply_witness_render_gate(row, *, policy)` lifted verbatim
    from `atlas_brain/services/b2b/witness_render_gate.py` (161
    LOC). Optional `policy` argument is additive; default matches
    legacy atlas constants.
  - `compute_coverage_ratio`, `row_count`,
    `build_non_empty_text_check` lifted from `source_impact.py`
    (the legacy underscore-prefixed versions).
  - Pack-contract entry point `evaluate_source_quality(input, *,
    policy)` returns a `QualityReport` whose findings enumerate
    suppressed witness rows; decision is BLOCK when zero rows
    render, WARN on partial suppression, PASS otherwise.
- Atlas-side `atlas_brain/services/b2b/witness_render_gate.py` is a
  thin re-export wrapper. The `_compute_coverage_ratio`,
  `_row_count`, `_build_non_empty_text_check` aliases in
  `source_impact.py` now resolve to the pack functions.
- Out of scope (kept atlas-side): `build_source_impact_ledger`,
  `summarize_source_field_baseline`, `get_consumer_wiring_baseline`,
  the `_SOURCE_IMPACT_PROFILES` registry data.
- PR-B5b regression fix: re-exported `_contains_term` and
  `_normalize_text` from `_b2b_specificity.py` for
  `services/blog_quality.py`.

PR-B5a (merged via #130): evidence-coverage gate.

- Deterministic core (`evidence_pack.py`) -- legacy entry point
  `audit_witness_evidence_coverage(pool, *, vendor_name, source_review_ids,
  min_pain_confidence, valid_status, coverage_precision)` lifted
  verbatim from `atlas_brain/services/b2b/evidence_gate.py`. The
  three original kwargs keep their signatures and defaults; two new
  kwargs (`valid_status`, `coverage_precision`) are additive.
- Pack-contract entry point `evaluate_evidence_coverage(pool, input,
  *, policy)` returns a standard `QualityReport`. Decision is driven
  by `coverage_block_threshold` / `coverage_warn_threshold` thresholds
  on `QualityPolicy.thresholds`; defaults preserve the current
  shadow-mode posture (block_threshold=0.0 means never block).
- Atlas-side `atlas_brain/services/b2b/evidence_gate.py` is a thin
  re-export wrapper. The single existing caller in
  `atlas_brain/autonomous/tasks/b2b_campaign_generation.py:1232`
  continues to work without change.

PR-B5b (merged via #125): witness specificity pack.

- Deterministic core (`witness_pack.py`) -- six legacy entry points
  (`surface_specificity_context`, `merge_specificity_contexts`,
  `specificity_signal_terms`, `evaluate_specificity_support`,
  `specificity_audit_snapshot`, `campaign_proof_terms_from_audit`)
  plus a pack-contract entry point
  `evaluate_witness_specificity(input, *, policy)` that returns a
  standard `QualityReport`.
- Atlas-side `atlas_brain/autonomous/tasks/_b2b_specificity.py` is now
  a thin re-export wrapper (~210 LOC, was 755). It keeps
  `campaign_policy_audit_snapshot` (the PR-B4b adapter) plus
  metadata helpers `latest_specificity_audit` and
  `specificity_quality_summary` that read pre-computed audit dicts
  out of row metadata and are not deterministic validators.
- All existing import paths (`from atlas_brain.autonomous.tasks._b2b_specificity import ...`)
  keep working without external code changes.

PR-B4b (merged via #120): campaign quality pack.

- Deterministic core (`campaign_pack.py`) -- `evaluate_campaign(input, *, policy)`
  returning a `QualityReport`. Validates proof-term coverage,
  report-tier banned language, forbidden competitor / incumbent names
  in cold email (channel + target_mode gated), and private account-name
  leak detection. Specificity findings are passed through from the
  Atlas-side audit so the report carries one merged blocking-issues
  list.
- Atlas-side wrapper (`atlas_brain/autonomous/tasks/_b2b_specificity.py:campaign_policy_audit_snapshot`)
  now runs the specificity audit, resolves proof terms, builds a
  `QualityInput`/`QualityPolicy`, calls the pack, and merges the pack
  findings with the audit dict so callers see the legacy shape.

PR-B4a (merged via #118): blog quality pack.

- Deterministic core (`blog_pack.py`) -- `evaluate_blog_post(input, *, policy)`
  returning a `QualityReport`. Validates word count, chart placeholders,
  unresolved tokens, quote count, review-period mention, methodology
  disclaimer, required vendors, placeholder/internal links, title/vendor
  match, category-outcome support, ungrounded data claims (two-strategy
  scan: known-vendor lookup + multi-word capitalized-name regex with
  configurable skip words), chart-scope ambiguity, numeric consistency,
  migration-direction drift.
- Atlas-side wrapper (`atlas_brain/autonomous/tasks/b2b_blog_post_generation.py:_apply_blog_quality_gate`)
  now sanitizes the body, builds a `QualityInput`/`QualityPolicy`, calls
  the pack, then layers atlas-side specificity findings on top.

PR-B3 (merged via #114): split safety-gate primitives.

- Deterministic core (`safety_gate.py`) -- `check_content` + `assess_risk`
- Atlas-side wrapper (`atlas_brain/services/safety_gate.py`) now delegates
  the pure functions to this package; approvals + audit-log + DB writes
  remain Atlas-side behind the existing `ApprovalStore` and `AuditLog`
  port protocols.

PR-B2 (merged via #85) is the prior slice:

- `product_claim.py`
- `api.py`
- `types.py`
- `ports.py`

The module is deterministic and imports without Atlas.

## Included

- Product claim envelope
- Evidence posture derivation
- Confidence derivation
- Render/report gate decisions
- Claim policy registry
- Generic quality report types
- Integration port protocols
- Safety gate (deterministic core: `check_content` + `assess_risk`) -- PR-B3
- Blog quality pack (deterministic core: `evaluate_blog_post`) -- PR-B4a
- Campaign quality pack (deterministic core: `evaluate_campaign`) -- PR-B4b
- Witness specificity pack (deterministic core: `evaluate_witness_specificity` + 6 legacy entry points) -- PR-B5b
- Evidence-coverage gate (deterministic core: `evaluate_evidence_coverage` + lifted `audit_witness_evidence_coverage`) -- PR-B5a
- Source-quality pack (deterministic core: `apply_witness_render_gate`, coverage helpers, `evaluate_source_quality`) -- PR-B5c

## Not Yet Included

- Safety gate Atlas adapter wrapper (approvals + audit log + DB stay
  in `atlas_brain/services/safety_gate.py`; the deterministic core
  is now in `extracted_quality_gate/safety_gate.py` and the wrapper
  delegates to it -- but the wrapper itself is not yet extracted)
- `build_source_impact_ledger` / `summarize_source_field_baseline` /
  `get_consumer_wiring_baseline` / `_SOURCE_IMPACT_PROFILES` registry
  remain atlas-side (atlas-coupled settings + schema).
- Memory quality pack
