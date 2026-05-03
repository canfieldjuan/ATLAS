# Extracted Quality Gate Status

Date: 2026-05-03

## Current Slice

PR-B4b: campaign quality pack.

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

## Not Yet Included

- Safety gate Atlas adapter wrapper (approvals + audit log + DB stay
  in `atlas_brain/services/safety_gate.py`; the deterministic core
  is now in `extracted_quality_gate/safety_gate.py` and the wrapper
  delegates to it -- but the wrapper itself is not yet extracted)
- Witness render policy pack (PR-B5)
- Evidence-claim coverage pack (PR-B5)
- Source-quality ingest pack (PR-B5)
- Memory quality pack
